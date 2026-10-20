//! Monitor layout for multi-monitor capture and coordinate input.
//!
//! On GNOME the layout comes from Mutter (`GetCurrentState` on
//! `org.gnome.Mutter.DisplayConfig`). On any other desktop, or when that call
//! fails, the whole capture counts as one monitor.
//!
//! Coordinates in a [`Monitor`] are layout coordinates. A screenshot, and every
//! coordinate the click tools accept, uses capture pixels. The two match at
//! scale 1.0. [`monitor_capture_rect`] converts one into the other.

use crate::diagnostics::hydrate_session_bus_env;
use crate::screenshot_impl::{
    capture_screenshot_raw, prepare_screenshot_payload, run_image_task, screenshot_pipeline_permit,
    RawScreenshotCapture, ScreenshotCapture, ScreenshotPayloadOptions,
};
use anyhow::{anyhow, bail, Context, Result};
use schemars::JsonSchema;
use serde::{
    de::{self, Visitor},
    Deserialize, Deserializer, Serialize,
};
use std::{collections::HashMap, fmt, io::Cursor, time::Duration};
use zbus::zvariant::OwnedValue;

const DISPLAY_CONFIG_SERVICE: &str = "org.gnome.Mutter.DisplayConfig";
const DISPLAY_CONFIG_PATH: &str = "/org/gnome/Mutter/DisplayConfig";
const DISPLAY_CONFIG_TIMEOUT: Duration = Duration::from_secs(3);
/// `layout-mode` in `GetCurrentState`: 1 is logical, 2 is physical.
const LAYOUT_MODE_LOGICAL: u32 = 1;
/// Connector name of the single monitor used when no layout is available.
pub(crate) const FULL_CAPTURE_CONNECTOR: &str = "screen";
/// Largest relative difference between the horizontal and vertical scale of a
/// capture before it no longer matches the monitor layout.
const SCALE_TOLERANCE: f64 = 0.01;
const LAYOUT_NOTE: &str = "x, y, width, and height are desktop coordinates. Pass monitor (index, connector, or \"primary\") to screenshot, get_app_state, click, drag, and scroll.";

/// One monitor of the desktop.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub(crate) struct Monitor {
    /// Position in the list. Index 0 is the left-most monitor.
    pub index: u32,
    /// Connector name, for example `HDMI-1` or `DP-2`.
    pub connector: String,
    /// Left edge in desktop coordinates.
    pub x: i32,
    /// Top edge in desktop coordinates.
    pub y: i32,
    /// Width in desktop coordinates.
    pub width: u32,
    /// Height in desktop coordinates.
    pub height: u32,
    /// Scale factor of the monitor.
    pub scale: f64,
    /// True for the primary monitor.
    pub primary: bool,
}

/// Where a [`MonitorLayout`] came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, JsonSchema)]
#[serde(rename_all = "kebab-case")]
pub(crate) enum LayoutSource {
    /// The GNOME compositor reported the layout.
    GnomeMutter,
    /// No layout was available, so the whole capture counts as one monitor.
    FullCapture,
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
pub(crate) struct MonitorLayout {
    /// `gnome-mutter` when the compositor reported the layout. `full-capture`
    /// when the whole screen counts as one monitor.
    pub source: LayoutSource,
    /// Monitors sorted by x, then y. Index 0 is the left-most monitor.
    pub monitors: Vec<Monitor>,
    /// Why the compositor layout was not used, when that is the case.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    pub note: String,
}

impl MonitorLayout {
    fn new(source: LayoutSource, monitors: Vec<Monitor>, error: Option<String>) -> Self {
        Self {
            source,
            monitors,
            error,
            note: LAYOUT_NOTE.to_string(),
        }
    }
}

/// Which monitor a tool call means: an index, a connector name, or `primary`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(untagged)]
pub(crate) enum MonitorSelector {
    Index(u32),
    Name(String),
}

impl MonitorSelector {
    /// Read a command-line or string value. Digits mean an index.
    pub(crate) fn from_text(text: &str) -> Result<Self, String> {
        let text = text.trim();
        if text.is_empty() {
            return Err("monitor must not be empty".to_string());
        }
        if text.bytes().all(|byte| byte.is_ascii_digit()) {
            return text
                .parse::<u32>()
                .map(Self::Index)
                .map_err(|_| "monitor index is too large".to_string());
        }
        Ok(Self::Name(text.to_string()))
    }

    /// Find the selected monitor. The error text lists the monitors that exist.
    pub(crate) fn resolve<'a>(&self, monitors: &'a [Monitor]) -> Result<&'a Monitor, String> {
        let found = match self {
            Self::Index(index) => monitors.iter().find(|monitor| monitor.index == *index),
            Self::Name(name) if name.eq_ignore_ascii_case("primary") => {
                monitors.iter().find(|monitor| monitor.primary)
            }
            Self::Name(name) => monitors
                .iter()
                .find(|monitor| monitor.connector.eq_ignore_ascii_case(name)),
        };
        found.ok_or_else(|| {
            format!(
                "monitor {self} was not found. Available monitors: {}",
                describe_monitors(monitors)
            )
        })
    }
}

impl fmt::Display for MonitorSelector {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Index(index) => write!(formatter, "{index}"),
            Self::Name(name) => write!(formatter, "{name:?}"),
        }
    }
}

fn describe_monitors(monitors: &[Monitor]) -> String {
    if monitors.is_empty() {
        return "none".to_string();
    }
    monitors
        .iter()
        .map(|monitor| {
            format!(
                "{} {}{}",
                monitor.index,
                monitor.connector,
                if monitor.primary { " (primary)" } else { "" }
            )
        })
        .collect::<Vec<_>>()
        .join(", ")
}

impl<'de> Deserialize<'de> for MonitorSelector {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct SelectorVisitor;

        impl Visitor<'_> for SelectorVisitor {
            type Value = MonitorSelector;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str(
                    "a monitor index (0, 1, ...), a connector name such as \"HDMI-1\", or \"primary\"",
                )
            }

            fn visit_u64<E: de::Error>(self, value: u64) -> Result<Self::Value, E> {
                u32::try_from(value)
                    .map(MonitorSelector::Index)
                    .map_err(|_| E::custom("monitor index is too large"))
            }

            fn visit_i64<E: de::Error>(self, value: i64) -> Result<Self::Value, E> {
                u64::try_from(value)
                    .map_err(|_| E::custom("monitor index must not be negative"))
                    .and_then(|value| self.visit_u64(value))
            }

            fn visit_f64<E: de::Error>(self, value: f64) -> Result<Self::Value, E> {
                if value.is_finite() && value.fract() == 0.0 && value >= 0.0 {
                    self.visit_u64(value as u64)
                } else {
                    Err(E::custom("monitor index must be a whole number"))
                }
            }

            fn visit_str<E: de::Error>(self, value: &str) -> Result<Self::Value, E> {
                MonitorSelector::from_text(value).map_err(E::custom)
            }
        }

        deserializer.deserialize_any(SelectorVisitor)
    }
}

fn selector_schema(description: &str) -> schemars::Schema {
    schemars::json_schema!({
        "description": description,
        "anyOf": [
            { "type": "integer", "minimum": 0 },
            { "type": "string" },
            { "type": "null" }
        ]
    })
}

/// JSON schema of `monitor` on `screenshot` and `get_app_state`.
pub(crate) fn screenshot_monitor_schema(_gen: &mut schemars::SchemaGenerator) -> schemars::Schema {
    selector_schema(
        "Capture one monitor at native size instead of the whole desktop: an index from list_monitors (0 is the left-most), a connector name such as \"HDMI-1\", or \"primary\". Cannot be combined with a window target. Metadata then includes monitor, origin_x, and origin_y; desktop x = origin_x + preview x / scale.",
    )
}

/// JSON schema of `monitor` on the tools that accept post-action capture.
pub(crate) fn post_action_monitor_schema(_gen: &mut schemars::SchemaGenerator) -> schemars::Schema {
    selector_schema(
        "Monitor to use: an index from list_monitors (0 is the left-most), a connector name such as \"HDMI-1\", or \"primary\". With screenshot_after true or \"screen\", only this monitor is captured; \"window\" ignores it. On click, drag, and scroll, x/y are measured from the top-left corner of this monitor and must lie inside it. Cannot be combined with a window target or relative: true.",
    )
}

/// Which monitor a capture was cropped to, and where its top-left corner sits
/// in the desktop capture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct MonitorCrop {
    pub connector: String,
    pub origin_x: i32,
    pub origin_y: i32,
}

/// A rectangle in capture pixels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CaptureRect {
    pub x: i32,
    pub y: i32,
    pub width: u32,
    pub height: u32,
}

// `GetCurrentState` replies with `(ua((ssss)a(siiddada{sv})a{sv})a(iiduba(ssss)a{sv})a{sv})`.
// The property type is generic so one parser serves the D-Bus reply
// (`OwnedValue`) and the JSON fixture in the tests (`serde_json::Value`).
type ConnectorInfo = (String, String, String, String);
type ModeEntry<P> = (String, i32, i32, f64, f64, Vec<f64>, HashMap<String, P>);
type PhysicalMonitorEntry<P> = (ConnectorInfo, Vec<ModeEntry<P>>, HashMap<String, P>);
type LogicalMonitorEntry<P> = (
    i32,
    i32,
    f64,
    u32,
    bool,
    Vec<ConnectorInfo>,
    HashMap<String, P>,
);
type CurrentState<P> = (
    u32,
    Vec<PhysicalMonitorEntry<P>>,
    Vec<LogicalMonitorEntry<P>>,
    HashMap<String, P>,
);

/// Read the few typed values this module needs from a property dictionary.
trait StateProperty {
    fn as_bool(&self) -> Option<bool>;
    fn as_u32(&self) -> Option<u32>;
}

impl StateProperty for OwnedValue {
    fn as_bool(&self) -> Option<bool> {
        self.try_clone()
            .ok()
            .and_then(|value| bool::try_from(value).ok())
    }

    fn as_u32(&self) -> Option<u32> {
        self.try_clone()
            .ok()
            .and_then(|value| u32::try_from(value).ok())
    }
}

#[cfg(test)]
impl StateProperty for serde_json::Value {
    fn as_bool(&self) -> Option<bool> {
        serde_json::Value::as_bool(self)
    }

    fn as_u32(&self) -> Option<u32> {
        serde_json::Value::as_u64(self).and_then(|value| u32::try_from(value).ok())
    }
}

/// Size in pixels of the mode a physical monitor runs now.
fn current_mode_size<P: StateProperty>(
    physical: &[PhysicalMonitorEntry<P>],
    connector: &str,
) -> Option<(u32, u32)> {
    let (_, modes, _) = physical
        .iter()
        .find(|((name, _, _, _), _, _)| name == connector)?;
    modes
        .iter()
        .find_map(|(_, width, height, _, _, _, properties)| {
            let is_current = properties
                .get("is-current")
                .and_then(StateProperty::as_bool);
            (is_current == Some(true)).then_some(())?;
            Some((u32::try_from(*width).ok()?, u32::try_from(*height).ok()?))
        })
}

/// Turn a `GetCurrentState` reply into monitors sorted by `(x, y)`.
fn parse_current_state<P: StateProperty>(state: &CurrentState<P>) -> Result<Vec<Monitor>> {
    let (_serial, physical, logical, properties) = state;
    let logical_layout = properties
        .get("layout-mode")
        .and_then(StateProperty::as_u32)
        == Some(LAYOUT_MODE_LOGICAL);

    let mut monitors = Vec::new();
    for (x, y, scale, transform, primary, connectors, _) in logical {
        // Mirrored monitors share one rectangle. The first connector names it.
        let Some((connector, _, _, _)) = connectors.first() else {
            continue;
        };
        let Some((mode_width, mode_height)) = current_mode_size(physical, connector) else {
            continue;
        };
        let scale = if scale.is_finite() && *scale > 0.0 {
            *scale
        } else {
            1.0
        };
        // Transforms 1, 3, 5, and 7 turn the monitor by 90 or 270 degrees.
        let (mut width, mut height) = if transform % 2 == 1 {
            (f64::from(mode_height), f64::from(mode_width))
        } else {
            (f64::from(mode_width), f64::from(mode_height))
        };
        if logical_layout {
            width = (width / scale).round();
            height = (height / scale).round();
        }
        if width < 1.0 || height < 1.0 {
            continue;
        }
        monitors.push(Monitor {
            index: 0,
            connector: connector.clone(),
            x: *x,
            y: *y,
            width: width as u32,
            height: height as u32,
            scale,
            primary: *primary,
        });
    }
    if monitors.is_empty() {
        bail!("the display configuration has no monitor with a current mode");
    }
    monitors.sort_by_key(|monitor| (monitor.x, monitor.y));
    for (index, monitor) in monitors.iter_mut().enumerate() {
        monitor.index = index as u32;
    }
    Ok(monitors)
}

async fn read_mutter_layout() -> Result<Vec<Monitor>> {
    hydrate_session_bus_env();
    let connection = zbus::Connection::session()
        .await
        .context("failed to connect to session bus")?;
    let reply = tokio::time::timeout(
        DISPLAY_CONFIG_TIMEOUT,
        connection.call_method(
            Some(DISPLAY_CONFIG_SERVICE),
            DISPLAY_CONFIG_PATH,
            Some(DISPLAY_CONFIG_SERVICE),
            "GetCurrentState",
            &(),
        ),
    )
    .await
    .map_err(|_| anyhow!("the display configuration did not answer within 3 seconds"))?
    .context("failed to read the display configuration")?;
    let state: CurrentState<OwnedValue> = reply
        .body()
        .deserialize()
        .context("the display configuration reply has an unexpected shape")?;
    parse_current_state(&state)
}

/// One monitor that covers a whole capture.
fn full_capture_monitors(width: u32, height: u32) -> Vec<Monitor> {
    vec![Monitor {
        index: 0,
        connector: FULL_CAPTURE_CONNECTOR.to_string(),
        x: 0,
        y: 0,
        width,
        height,
        scale: 1.0,
        primary: true,
    }]
}

async fn measure_capture() -> Result<(u32, u32)> {
    let _pipeline = screenshot_pipeline_permit().await?;
    let raw = capture_screenshot_raw().await?;
    Ok((raw.width, raw.height))
}

/// Current monitor layout. `capture_size` is the size of a capture the caller
/// already holds. Without it, and without a compositor layout, one capture
/// measures the screen. Never call this with the screenshot pipeline held and
/// no `capture_size`.
pub(crate) async fn current_layout(capture_size: Option<(u32, u32)>) -> MonitorLayout {
    let reason = match read_mutter_layout().await {
        Ok(monitors) => return MonitorLayout::new(LayoutSource::GnomeMutter, monitors, None),
        Err(error) => format!("{error:#}"),
    };
    let size = match capture_size {
        Some(size) => Ok(size),
        None => measure_capture().await,
    };
    match size {
        Ok((width, height)) => MonitorLayout::new(
            LayoutSource::FullCapture,
            full_capture_monitors(width, height),
            Some(reason),
        ),
        Err(error) => MonitorLayout::new(
            LayoutSource::FullCapture,
            Vec::new(),
            Some(format!("{reason}; the screen size is unknown: {error:#}")),
        ),
    }
}

/// Rectangle of `monitor` inside a capture of `capture_width` x `capture_height`
/// pixels that shows the whole desktop in `monitors`.
pub(crate) fn monitor_capture_rect(
    monitor: &Monitor,
    monitors: &[Monitor],
    capture_width: u32,
    capture_height: u32,
) -> Result<CaptureRect> {
    if capture_width == 0 || capture_height == 0 {
        bail!("the screenshot is empty");
    }
    let edges = |monitor: &Monitor| {
        (
            i64::from(monitor.x),
            i64::from(monitor.y),
            i64::from(monitor.x) + i64::from(monitor.width),
            i64::from(monitor.y) + i64::from(monitor.height),
        )
    };
    let (min_x, min_y, max_x, max_y) = monitors
        .iter()
        .chain(std::iter::once(monitor))
        .map(edges)
        .reduce(|(a, b, c, d), (e, f, g, h)| (a.min(e), b.min(f), c.max(g), d.max(h)))
        .ok_or_else(|| anyhow!("there is no monitor layout"))?;
    let (layout_width, layout_height) = (max_x - min_x, max_y - min_y);
    let scale_x = f64::from(capture_width) / layout_width as f64;
    let scale_y = f64::from(capture_height) / layout_height as f64;
    if !scale_x.is_finite()
        || !scale_y.is_finite()
        || (scale_x - scale_y).abs() > SCALE_TOLERANCE * scale_x.max(scale_y)
    {
        bail!(
            "the {capture_width}x{capture_height} screenshot does not match the {layout_width}x{layout_height} monitor layout"
        );
    }

    // The epsilon keeps floating-point noise from moving an edge by one pixel.
    let (left, top, right, bottom) = edges(monitor);
    let to_capture = |value: i64, origin: i64, scale: f64| (value - origin) as f64 * scale;
    let left = (to_capture(left, min_x, scale_x) + 1e-6).floor() as i64;
    let top = (to_capture(top, min_y, scale_y) + 1e-6).floor() as i64;
    let right = (to_capture(right, min_x, scale_x) - 1e-6).ceil() as i64;
    let bottom = (to_capture(bottom, min_y, scale_y) - 1e-6).ceil() as i64;

    let left = left.max(0);
    let top = top.max(0);
    let right = right.min(i64::from(capture_width));
    let bottom = bottom.min(i64::from(capture_height));
    if right <= left || bottom <= top {
        bail!(
            "monitor {} is outside the {capture_width}x{capture_height} screenshot",
            monitor.connector
        );
    }
    Ok(CaptureRect {
        x: i32::try_from(left)?,
        y: i32::try_from(top)?,
        width: u32::try_from(right - left)?,
        height: u32::try_from(bottom - top)?,
    })
}

/// Turn a point measured from the monitor's top-left corner into a desktop
/// capture point. Rejects a point outside the monitor.
pub(crate) fn monitor_point_to_desktop(
    connector: &str,
    rect: CaptureRect,
    x: i32,
    y: i32,
) -> Result<(i32, i32), String> {
    if x < 0 || y < 0 || x as u32 >= rect.width || y as u32 >= rect.height {
        return Err(format!(
            "Coordinate ({x}, {y}) is outside monitor {connector} ({}x{}). Coordinates are measured from the top-left corner of the monitor.",
            rect.width, rect.height
        ));
    }
    let desktop_x = rect
        .x
        .checked_add(x)
        .ok_or_else(|| "The x coordinate overflowed.".to_string())?;
    let desktop_y = rect
        .y
        .checked_add(y)
        .ok_or_else(|| "The y coordinate overflowed.".to_string())?;
    Ok((desktop_x, desktop_y))
}

fn crop_raw_capture(raw: RawScreenshotCapture, rect: CaptureRect) -> Result<RawScreenshotCapture> {
    let image = image::load_from_memory_with_format(&raw.bytes, image::ImageFormat::Png)
        .context("failed to decode the screenshot for the monitor crop")?;
    let x = u32::try_from(rect.x).context("monitor crop x is negative")?;
    let y = u32::try_from(rect.y).context("monitor crop y is negative")?;
    let width = rect.width.min(image.width().saturating_sub(x));
    let height = rect.height.min(image.height().saturating_sub(y));
    if width == 0 || height == 0 {
        bail!("the monitor crop is outside the screenshot");
    }
    let mut bytes = Vec::new();
    image
        .crop_imm(x, y, width, height)
        .write_to(&mut Cursor::new(&mut bytes), image::ImageFormat::Png)
        .context("failed to encode the monitor crop")?;
    Ok(RawScreenshotCapture {
        mime_type: raw.mime_type,
        bytes,
        source: raw.source,
        width,
        height,
    })
}

/// Cut a full-desktop capture down to the selected monitor.
pub(crate) async fn crop_to_monitor(
    selector: &MonitorSelector,
    raw: RawScreenshotCapture,
) -> Result<(RawScreenshotCapture, MonitorCrop)> {
    let layout = current_layout(Some((raw.width, raw.height))).await;
    let monitor = selector
        .resolve(&layout.monitors)
        .map_err(anyhow::Error::msg)?;
    let rect = monitor_capture_rect(monitor, &layout.monitors, raw.width, raw.height)?;
    let crop = MonitorCrop {
        connector: monitor.connector.clone(),
        origin_x: rect.x,
        origin_y: rect.y,
    };
    if rect.x == 0 && rect.y == 0 && rect.width == raw.width && rect.height == raw.height {
        return Ok((raw, crop));
    }
    let cropped = run_image_task(move || crop_raw_capture(raw, rect)).await?;
    Ok((cropped, crop))
}

/// Capture the desktop and return the selected monitor as a bounded payload.
pub(crate) async fn capture_monitor_screenshot(
    selector: &MonitorSelector,
    options: ScreenshotPayloadOptions,
) -> Result<(ScreenshotCapture, MonitorCrop)> {
    let _pipeline = screenshot_pipeline_permit().await?;
    let raw = capture_screenshot_raw().await?;
    let (raw, crop) = crop_to_monitor(selector, raw).await?;
    let capture = run_image_task(move || prepare_screenshot_payload(raw, options)).await?;
    Ok((capture, crop))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const THREE_MONITOR_REPLY: &str =
        include_str!("../tests/fixtures/mutter-display-config-three-monitors.json");

    type JsonState = CurrentState<serde_json::Value>;

    fn monitor(index: u32, connector: &str, x: i32, y: i32, width: u32, height: u32) -> Monitor {
        Monitor {
            index,
            connector: connector.to_string(),
            x,
            y,
            width,
            height,
            scale: 1.0,
            primary: index == 0,
        }
    }

    fn three_monitors() -> Vec<Monitor> {
        vec![
            Monitor {
                primary: false,
                ..monitor(0, "DP-2", 0, 0, 1920, 1080)
            },
            Monitor {
                primary: true,
                ..monitor(1, "HDMI-2", 1920, 0, 1920, 1080)
            },
            Monitor {
                primary: false,
                ..monitor(2, "HDMI-1", 3840, 0, 1920, 1080)
            },
        ]
    }

    /// Build a reply from `(connector, current mode width, height)` monitors
    /// and `(x, y, scale, transform, primary, connector)` logical monitors.
    fn state(
        layout_mode: u32,
        physical: &[(&str, i32, i32)],
        logical: &[(i32, i32, f64, u32, bool, &str)],
    ) -> JsonState {
        let physical: Vec<_> = physical
            .iter()
            .map(|(connector, width, height)| {
                json!([
                    [connector, "VND", "MODEL", "serial"],
                    [
                        ["640x480@60.000", 640, 480, 60.0, 1.0, [1.0], {}],
                        [
                            format!("{width}x{height}@60.000"),
                            width,
                            height,
                            60.0,
                            1.0,
                            [1.0, 2.0],
                            {"is-current": true}
                        ]
                    ],
                    {}
                ])
            })
            .collect();
        let logical: Vec<_> = logical
            .iter()
            .map(|(x, y, scale, transform, primary, connector)| {
                json!([
                    x,
                    y,
                    scale,
                    transform,
                    primary,
                    [[connector, "VND", "MODEL", "serial"]],
                    {}
                ])
            })
            .collect();
        serde_json::from_value(json!([
            1,
            physical,
            logical,
            {"layout-mode": layout_mode}
        ]))
        .unwrap()
    }

    #[test]
    fn real_three_monitor_reply_parses_into_sorted_monitors() {
        let reply: JsonState = serde_json::from_str(THREE_MONITOR_REPLY).unwrap();
        let monitors = parse_current_state(&reply).unwrap();
        assert_eq!(monitors, three_monitors());
    }

    #[test]
    fn logical_layout_divides_mode_size_by_scale() {
        let physical = [("eDP-1", 3840, 2160)];
        let logical = [(0, 0, 2.0, 0, true, "eDP-1")];
        let scaled = parse_current_state(&state(1, &physical, &logical)).unwrap();
        assert_eq!((scaled[0].width, scaled[0].height), (1920, 1080));
        assert_eq!(scaled[0].scale, 2.0);

        let fractional = [(0, 0, 1.25, 0, true, "eDP-1")];
        let fractional = parse_current_state(&state(1, &physical, &fractional)).unwrap();
        assert_eq!((fractional[0].width, fractional[0].height), (3072, 1728));

        let physical_mode = parse_current_state(&state(2, &physical, &logical)).unwrap();
        assert_eq!(
            (physical_mode[0].width, physical_mode[0].height),
            (3840, 2160)
        );
    }

    #[test]
    fn rotated_monitor_swaps_width_and_height() {
        let physical = [("DP-1", 1920, 1080), ("DP-2", 1920, 1080)];
        let logical = [
            (0, 0, 1.0, 0, true, "DP-1"),
            (1920, 0, 1.0, 1, false, "DP-2"),
            (3000, 0, 1.0, 5, false, "DP-1"),
        ];
        let monitors = parse_current_state(&state(2, &physical, &logical)).unwrap();
        assert_eq!((monitors[0].width, monitors[0].height), (1920, 1080));
        assert_eq!((monitors[1].width, monitors[1].height), (1080, 1920));
        assert_eq!((monitors[2].width, monitors[2].height), (1080, 1920));
    }

    #[test]
    fn monitors_sort_by_x_then_y_with_negative_origins() {
        let physical = [
            ("A", 1000, 800),
            ("B", 1000, 800),
            ("C", 1000, 800),
            ("D", 1000, 800),
        ];
        let logical = [
            (0, 0, 1.0, 0, true, "A"),
            (1000, 800, 1.0, 0, false, "D"),
            (-1000, 0, 1.0, 0, false, "B"),
            (1000, 0, 1.0, 0, false, "C"),
        ];
        let monitors = parse_current_state(&state(2, &physical, &logical)).unwrap();
        let order: Vec<_> = monitors
            .iter()
            .map(|monitor| (monitor.index, monitor.connector.as_str()))
            .collect();
        assert_eq!(order, [(0, "B"), (1, "A"), (2, "C"), (3, "D")]);
    }

    #[test]
    fn logical_monitor_without_a_current_mode_is_skipped() {
        let physical = [("DP-1", 1920, 1080)];
        let logical = [
            (0, 0, 1.0, 0, true, "DP-1"),
            (1920, 0, 1.0, 0, false, "GHOST"),
        ];
        let monitors = parse_current_state(&state(2, &physical, &logical)).unwrap();
        assert_eq!(monitors.len(), 1);

        let only_ghost = [(0, 0, 1.0, 0, true, "GHOST")];
        assert!(parse_current_state(&state(2, &physical, &only_ghost)).is_err());
        assert!(parse_current_state(&state(2, &[], &[])).is_err());
    }

    #[test]
    fn selector_reads_index_connector_and_primary() {
        let selector = |value: serde_json::Value| -> Result<MonitorSelector, String> {
            serde_json::from_value(value).map_err(|error| error.to_string())
        };
        assert_eq!(selector(json!(0)), Ok(MonitorSelector::Index(0)));
        assert_eq!(selector(json!(2.0)), Ok(MonitorSelector::Index(2)));
        assert_eq!(selector(json!("1")), Ok(MonitorSelector::Index(1)));
        assert_eq!(
            selector(json!(" HDMI-2 ")),
            Ok(MonitorSelector::Name("HDMI-2".to_string()))
        );
        assert_eq!(
            selector(json!("primary")),
            Ok(MonitorSelector::Name("primary".to_string()))
        );
        assert!(selector(json!(-1)).unwrap_err().contains("negative"));
        assert!(selector(json!(1.5)).unwrap_err().contains("whole number"));
        assert!(selector(json!("")).unwrap_err().contains("empty"));
        assert!(selector(json!(true)).is_err());
        assert!(selector(json!(4_294_967_296_u64))
            .unwrap_err()
            .contains("too large"));
    }

    #[test]
    fn selector_resolves_against_the_layout() {
        let monitors = three_monitors();
        let resolve = |selector: MonitorSelector| {
            selector
                .resolve(&monitors)
                .map(|monitor| monitor.connector.clone())
        };
        assert_eq!(resolve(MonitorSelector::Index(0)), Ok("DP-2".to_string()));
        assert_eq!(resolve(MonitorSelector::Index(2)), Ok("HDMI-1".to_string()));
        assert_eq!(
            resolve(MonitorSelector::Name("hdmi-2".to_string())),
            Ok("HDMI-2".to_string())
        );
        assert_eq!(
            resolve(MonitorSelector::Name("Primary".to_string())),
            Ok("HDMI-2".to_string())
        );

        let error = resolve(MonitorSelector::Index(3)).unwrap_err();
        assert!(error.contains("monitor 3 was not found"), "{error}");
        assert!(
            error.contains("0 DP-2, 1 HDMI-2 (primary), 2 HDMI-1"),
            "{error}"
        );
        let error = resolve(MonitorSelector::Name("VGA-1".to_string())).unwrap_err();
        assert!(error.contains("\"VGA-1\""), "{error}");
        assert!(MonitorSelector::Index(0)
            .resolve(&[])
            .unwrap_err()
            .contains("none"));
    }

    #[test]
    fn selector_without_a_primary_monitor_reports_it() {
        let monitors = vec![Monitor {
            primary: false,
            ..monitor(0, "DP-1", 0, 0, 1920, 1080)
        }];
        let error = MonitorSelector::Name("primary".to_string())
            .resolve(&monitors)
            .unwrap_err();
        assert!(error.contains("was not found"), "{error}");
    }

    #[test]
    fn capture_rect_cuts_each_monitor_out_of_the_desktop() {
        let monitors = three_monitors();
        let rects: Vec<_> = monitors
            .iter()
            .map(|monitor| monitor_capture_rect(monitor, &monitors, 5760, 1080).unwrap())
            .collect();
        let expected = [(0, 0), (1920, 0), (3840, 0)].map(|(x, y)| CaptureRect {
            x,
            y,
            width: 1920,
            height: 1080,
        });
        assert_eq!(rects, expected);
    }

    #[test]
    fn capture_rect_follows_the_capture_scale() {
        let monitors = three_monitors();
        // The capture is 1.25 times the layout.
        let rect = monitor_capture_rect(&monitors[1], &monitors, 7200, 1350).unwrap();
        assert_eq!(
            rect,
            CaptureRect {
                x: 2400,
                y: 0,
                width: 2400,
                height: 1350
            }
        );
    }

    #[test]
    fn capture_rect_starts_at_the_left_edge_of_the_layout() {
        let monitors = vec![
            monitor(0, "LEFT", -1920, 0, 1920, 1080),
            monitor(1, "RIGHT", 0, 0, 2560, 1440),
        ];
        let left = monitor_capture_rect(&monitors[0], &monitors, 4480, 1440).unwrap();
        assert_eq!(
            (left.x, left.y, left.width, left.height),
            (0, 0, 1920, 1080)
        );
        let right = monitor_capture_rect(&monitors[1], &monitors, 4480, 1440).unwrap();
        assert_eq!(
            (right.x, right.y, right.width, right.height),
            (1920, 0, 2560, 1440)
        );
    }

    #[test]
    fn capture_rect_handles_stacked_monitors() {
        let monitors = vec![
            monitor(0, "TOP", 0, -1080, 1920, 1080),
            monitor(1, "BOTTOM", 0, 0, 1920, 1080),
        ];
        let bottom = monitor_capture_rect(&monitors[1], &monitors, 1920, 2160).unwrap();
        assert_eq!(
            (bottom.x, bottom.y, bottom.width, bottom.height),
            (0, 1080, 1920, 1080)
        );
    }

    #[test]
    fn capture_rect_of_a_full_capture_monitor_is_the_whole_capture() {
        let monitors = full_capture_monitors(2560, 1440);
        let rect = monitor_capture_rect(&monitors[0], &monitors, 2560, 1440).unwrap();
        assert_eq!(
            rect,
            CaptureRect {
                x: 0,
                y: 0,
                width: 2560,
                height: 1440
            }
        );
    }

    #[test]
    fn capture_rect_rejects_a_capture_that_does_not_match_the_layout() {
        let monitors = three_monitors();
        // One monitor wide: the layout is three times wider than the capture.
        let error = monitor_capture_rect(&monitors[0], &monitors, 1920, 1080).unwrap_err();
        assert!(error.to_string().contains("does not match"), "{error}");
        assert!(monitor_capture_rect(&monitors[0], &monitors, 0, 1080).is_err());
    }

    #[test]
    fn monitor_point_converts_to_desktop_coordinates() {
        let rect = CaptureRect {
            x: 1920,
            y: 0,
            width: 1920,
            height: 1080,
        };
        assert_eq!(
            monitor_point_to_desktop("HDMI-2", rect, 0, 0),
            Ok((1920, 0))
        );
        assert_eq!(
            monitor_point_to_desktop("HDMI-2", rect, 100, 50),
            Ok((2020, 50))
        );
        assert_eq!(
            monitor_point_to_desktop("HDMI-2", rect, 1919, 1079),
            Ok((3839, 1079))
        );
    }

    #[test]
    fn monitor_point_rejects_coordinates_outside_the_monitor() {
        let rect = CaptureRect {
            x: 1920,
            y: 0,
            width: 1920,
            height: 1080,
        };
        for (x, y) in [(-1, 0), (0, -1), (1920, 0), (0, 1080), (5000, 5000)] {
            let error = monitor_point_to_desktop("HDMI-2", rect, x, y).unwrap_err();
            assert!(
                error.contains("outside monitor HDMI-2 (1920x1080)"),
                "{error}"
            );
        }
        let edge = CaptureRect {
            x: i32::MAX,
            y: 0,
            width: 10,
            height: 10,
        };
        assert!(monitor_point_to_desktop("X", edge, 5, 0)
            .unwrap_err()
            .contains("overflowed"));
    }

    #[test]
    fn crop_keeps_only_the_selected_pixels() {
        // 6x2 image: the pixel at (x, y) is (x * 10, y * 10, 7).
        let image =
            image::RgbImage::from_fn(6, 2, |x, y| image::Rgb([(x * 10) as u8, (y * 10) as u8, 7]));
        let mut bytes = Vec::new();
        image::DynamicImage::ImageRgb8(image)
            .write_to(&mut Cursor::new(&mut bytes), image::ImageFormat::Png)
            .unwrap();
        let raw = RawScreenshotCapture {
            mime_type: "image/png".to_string(),
            bytes,
            source: "test".to_string(),
            width: 6,
            height: 2,
        };
        let cropped = crop_raw_capture(
            raw,
            CaptureRect {
                x: 2,
                y: 1,
                width: 3,
                height: 1,
            },
        )
        .unwrap();
        assert_eq!((cropped.width, cropped.height), (3, 1));
        assert_eq!(cropped.source, "test");
        let decoded = image::load_from_memory_with_format(&cropped.bytes, image::ImageFormat::Png)
            .unwrap()
            .to_rgb8();
        assert_eq!(decoded.dimensions(), (3, 1));
        assert_eq!(decoded.get_pixel(0, 0).0, [20, 10, 7]);
        assert_eq!(decoded.get_pixel(2, 0).0, [40, 10, 7]);
    }

    #[test]
    fn crop_rejects_a_rectangle_outside_the_image() {
        let mut bytes = Vec::new();
        image::DynamicImage::ImageRgb8(image::RgbImage::new(4, 4))
            .write_to(&mut Cursor::new(&mut bytes), image::ImageFormat::Png)
            .unwrap();
        let raw = RawScreenshotCapture {
            mime_type: "image/png".to_string(),
            bytes,
            source: "test".to_string(),
            width: 4,
            height: 4,
        };
        let outside = CaptureRect {
            x: 4,
            y: 0,
            width: 2,
            height: 2,
        };
        assert!(crop_raw_capture(raw, outside).is_err());
    }

    #[test]
    fn full_capture_monitor_covers_the_capture() {
        let monitors = full_capture_monitors(1280, 720);
        assert_eq!(monitors.len(), 1);
        assert_eq!(monitors[0].connector, FULL_CAPTURE_CONNECTOR);
        assert_eq!(
            (
                monitors[0].x,
                monitors[0].y,
                monitors[0].width,
                monitors[0].height
            ),
            (0, 0, 1280, 720)
        );
        assert!(monitors[0].primary);
        assert_eq!(
            MonitorSelector::Name("primary".to_string())
                .resolve(&monitors)
                .unwrap()
                .index,
            0
        );
    }

    #[test]
    fn layout_serializes_the_documented_fields() {
        let layout = MonitorLayout::new(LayoutSource::GnomeMutter, three_monitors(), None);
        let value = serde_json::to_value(&layout).unwrap();
        assert_eq!(value["source"], "gnome-mutter");
        assert!(value.get("error").is_none());
        assert_eq!(
            value["monitors"][1],
            json!({
                "index": 1,
                "connector": "HDMI-2",
                "x": 1920,
                "y": 0,
                "width": 1920,
                "height": 1080,
                "scale": 1.0,
                "primary": true
            })
        );

        let fallback = MonitorLayout::new(
            LayoutSource::FullCapture,
            full_capture_monitors(800, 600),
            Some("no compositor layout".to_string()),
        );
        let value = serde_json::to_value(&fallback).unwrap();
        assert_eq!(value["source"], "full-capture");
        assert_eq!(value["error"], "no compositor layout");
    }

    #[test]
    fn selector_serializes_back_to_its_input() {
        assert_eq!(
            serde_json::to_value(MonitorSelector::Index(2)).unwrap(),
            json!(2)
        );
        assert_eq!(
            serde_json::to_value(MonitorSelector::Name("HDMI-1".to_string())).unwrap(),
            json!("HDMI-1")
        );
    }
}
