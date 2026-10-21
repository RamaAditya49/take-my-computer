//! Safety gate for mutating tools (take-my-computer).
//!
//! MCP annotations are hints only, and many hosts run with approvals off, so
//! the server enforces the gate itself. Each mutating tool calls
//! [`Policy::enter`], reads the desktop facts, and then calls
//! [`Entry::decide`] before it changes focus, shows an indicator event, or
//! sends input. The gate does these things in this order:
//!
//! 1. It refuses every call when the kill switch file exists.
//! 2. It refuses every mutating call in the `observe` mode.
//! 3. It takes an exclusive `flock`. The lock stays until the action and its
//!    audit line are done. No other call can send input while a dialog is open.
//! 4. The caller reads the window, element, and focus facts. The facts are
//!    read after the lock, so they describe the desktop at action time.
//! 5. It rates the call and asks the user to confirm high-impact actions.
//! 6. It writes an audit line with `"phase":"start"`. If this write fails,
//!    the call is refused and no input is sent.
//! 7. After the action, it writes a second line with `"phase":"end"`. A
//!    refused call writes one line with `"phase":"refused"`.
//!
//! No audit line holds typed text: only length and digest.

use anyhow::{Context, Result};
use futures_util::future::BoxFuture;
use rmcp::{
    model::{
        ClientResult, ElicitRequest, ElicitRequestParams, ElicitationAction, ElicitationSchema,
        ServerRequest,
    },
    schemars::JsonSchema,
    Peer, RoleServer,
};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    env,
    fs::{DirBuilder, File, OpenOptions},
    io::{ErrorKind, Write},
    os::{
        fd::AsRawFd,
        unix::fs::{DirBuilderExt, OpenOptionsExt},
    },
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

pub(crate) const POLICY_ENV: &str = "TAKE_MY_COMPUTER_POLICY";
pub(crate) const SEND_APPS_ENV: &str = "TAKE_MY_COMPUTER_SEND_APPS";
pub(crate) const TERMINAL_APPS_ENV: &str = "TAKE_MY_COMPUTER_TERMINAL_APPS";

/// Apps where a bare Enter key sends a message. Override with
/// `TAKE_MY_COMPUTER_SEND_APPS`.
pub(crate) const DEFAULT_SEND_APPS: [&str; 10] = [
    "telegram",
    "whatsapp",
    "slack",
    "discord",
    "signal",
    "element",
    "teams",
    "thunderbird",
    "evolution",
    "geary",
];

/// Terminal apps. Enter in a terminal runs a shell command. Override with
/// `TAKE_MY_COMPUTER_TERMINAL_APPS`.
pub(crate) const DEFAULT_TERMINAL_APPS: [&str; 14] = [
    "gnome-terminal",
    "kgx",
    "ptyxis",
    "console",
    "konsole",
    "kitty",
    "alacritty",
    "foot",
    "wezterm",
    "xterm",
    "tilix",
    "terminator",
    "ghostty",
    "terminal",
];

/// Words on a control that make a click or action high-impact. English first,
/// then Indonesian. Matching is whole-word and case-insensitive.
pub(crate) const COMMIT_WORDS: [&str; 21] = [
    "send",
    "submit",
    "delete",
    "remove",
    "pay",
    "buy",
    "purchase",
    "checkout",
    "publish",
    "post",
    "transfer",
    "confirm",
    "approve",
    "kirim",
    "hapus",
    "bayar",
    "beli",
    "terbitkan",
    "posting",
    "setujui",
    "konfirmasi",
];

const DIALOG_TIMEOUT: Duration = Duration::from_secs(60);
const ELICIT_TIMEOUT: Duration = Duration::from_secs(120);
const INPUT_LOCK_TIMEOUT: Duration = Duration::from_secs(30);
const LOCK_POLL_INTERVAL: Duration = Duration::from_millis(50);
/// Extra time for zenity to exit after its own `--timeout` fires.
const DIALOG_PROCESS_GRACE: Duration = Duration::from_secs(5);
const AUDIT_FIELD_MAX_CHARS: usize = 200;
const PROMPT_FIELD_MAX_CHARS: usize = 80;
/// A shell command up to this length shows in full in the confirmation. A
/// longer command is refused in `confirm` mode, because the user cannot read it.
pub(crate) const RUN_SHELL_PREVIEW_MAX_CHARS: usize = 2000;

// ---------------------------------------------------------------------------
// Mode
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Mode {
    /// Read-only tools run. Low-impact input runs. High-impact actions need confirmation.
    Confirm,
    /// All tools run without confirmation. Audit and kill switch still apply.
    Full,
    /// Only read-only tools run. Every mutating tool is refused.
    Observe,
}

impl Mode {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Mode::Confirm => "confirm",
            Mode::Full => "full",
            Mode::Observe => "observe",
        }
    }

    /// Parse the policy variable. An unknown value falls back to `confirm`
    /// and returns a warning.
    pub(crate) fn parse(value: Option<&str>) -> (Mode, Option<String>) {
        let Some(raw) = value.map(str::trim).filter(|value| !value.is_empty()) else {
            return (Mode::Confirm, None);
        };
        match raw.to_ascii_lowercase().as_str() {
            "confirm" => (Mode::Confirm, None),
            "full" => (Mode::Full, None),
            "observe" => (Mode::Observe, None),
            _ => (
                Mode::Confirm,
                Some(format!(
                    "Unknown {POLICY_ENV} value {raw:?}. Use confirm, full, or observe. Using confirm."
                )),
            ),
        }
    }
}

// ---------------------------------------------------------------------------
// Configuration and paths
// ---------------------------------------------------------------------------

/// Lowercase substrings that the classifier matches against the app id, the
/// wm class, and the window title.
#[derive(Debug, Clone)]
pub(crate) struct AppLists {
    /// Apps where Enter sends a message.
    pub send: Vec<String>,
    /// Terminal apps, where Enter runs a shell command.
    pub terminal: Vec<String>,
}

impl Default for AppLists {
    fn default() -> Self {
        Self {
            send: parse_send_apps(None),
            terminal: parse_terminal_apps(None),
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct PolicyConfig {
    pub mode: Mode,
    pub warning: Option<String>,
    pub state_dir: PathBuf,
    pub lock_path: PathBuf,
    pub apps: AppLists,
    pub lock_timeout: Duration,
    pub dialog_timeout: Duration,
    pub elicit_timeout: Duration,
}

impl PolicyConfig {
    pub(crate) fn from_env() -> Self {
        Self::from_lookup(|name| env::var(name).ok())
    }

    /// Build a config from a variable lookup, so tests do not touch the
    /// process environment.
    pub(crate) fn from_lookup(get: impl Fn(&str) -> Option<String>) -> Self {
        let (mode, warning) = Mode::parse(get(POLICY_ENV).as_deref());
        let state_dir = resolve_state_dir(&get);
        let lock_path = non_empty(get("XDG_RUNTIME_DIR"))
            .map(PathBuf::from)
            .filter(|path| path.is_absolute())
            .map(|dir| dir.join("take-my-computer-input.lock"))
            .unwrap_or_else(|| state_dir.join("input.lock"));
        Self {
            mode,
            warning,
            state_dir,
            lock_path,
            apps: AppLists {
                send: parse_send_apps(get(SEND_APPS_ENV).as_deref()),
                terminal: parse_terminal_apps(get(TERMINAL_APPS_ENV).as_deref()),
            },
            lock_timeout: INPUT_LOCK_TIMEOUT,
            dialog_timeout: DIALOG_TIMEOUT,
            elicit_timeout: ELICIT_TIMEOUT,
        }
    }

    pub(crate) fn stop_file(&self) -> PathBuf {
        self.state_dir.join("STOP")
    }

    pub(crate) fn audit_path(&self) -> PathBuf {
        self.state_dir.join("audit.jsonl")
    }

    /// A dangling symlink still counts: the file name is what the operator created.
    pub(crate) fn kill_switch_active(&self) -> bool {
        std::fs::symlink_metadata(self.stop_file()).is_ok()
    }
}

fn non_empty(value: Option<String>) -> Option<String> {
    value.filter(|value| !value.trim().is_empty())
}

fn resolve_state_dir(get: &impl Fn(&str) -> Option<String>) -> PathBuf {
    let base = non_empty(get("XDG_STATE_HOME"))
        .map(PathBuf::from)
        .filter(|path| path.is_absolute())
        .or_else(|| non_empty(get("HOME")).map(|home| PathBuf::from(home).join(".local/state")))
        // XDG_RUNTIME_DIR is private to the user, so it is safer than the shared temp dir.
        .or_else(|| {
            non_empty(get("XDG_RUNTIME_DIR"))
                .map(PathBuf::from)
                .filter(|path| path.is_absolute())
                .map(|dir| dir.join("take-my-computer-state"))
        })
        // Never fall back to the shared temp dir: other users can plant files
        // and symlinks there. Without a private base the gate fails closed,
        // because it cannot create the state dir and refuses every input.
        .or_else(|| passwd_home().map(|home| home.join(".local/state")))
        .unwrap_or_else(|| PathBuf::from("/nonexistent"));
    base.join("take-my-computer")
}

/// The home directory of the current user from the passwd database.
fn passwd_home() -> Option<PathBuf> {
    use std::os::unix::ffi::OsStrExt;
    let mut buf = vec![0 as libc::c_char; 16 * 1024];
    // SAFETY: passwd is plain data, so all zero bytes is a valid value.
    let mut pwd: libc::passwd = unsafe { std::mem::zeroed() };
    let mut result: *mut libc::passwd = std::ptr::null_mut();
    // SAFETY: every pointer refers to a live local buffer of the stated size,
    // and getuid has no preconditions.
    let rc = unsafe {
        libc::getpwuid_r(
            libc::getuid(),
            &mut pwd,
            buf.as_mut_ptr(),
            buf.len(),
            &mut result,
        )
    };
    if rc != 0 || result.is_null() || pwd.pw_dir.is_null() {
        return None;
    }
    // SAFETY: on success pw_dir points to a NUL-terminated string inside buf.
    let dir = unsafe { std::ffi::CStr::from_ptr(pwd.pw_dir) };
    let path = PathBuf::from(std::ffi::OsStr::from_bytes(dir.to_bytes()));
    path.is_absolute().then_some(path)
}

/// An empty or blank list means "use the default list". That is the safe
/// reading: an empty override must not silently disable the check.
fn parse_app_list(value: Option<&str>, defaults: &[&str]) -> Vec<String> {
    let parsed: Vec<String> = value
        .map(|value| {
            value
                .split(',')
                .map(|item| item.trim().to_lowercase())
                .filter(|item| !item.is_empty())
                .collect()
        })
        .unwrap_or_default();
    if parsed.is_empty() {
        defaults.iter().map(|app| (*app).to_string()).collect()
    } else {
        parsed
    }
}

fn parse_send_apps(value: Option<&str>) -> Vec<String> {
    parse_app_list(value, &DEFAULT_SEND_APPS)
}

fn parse_terminal_apps(value: Option<&str>) -> Vec<String> {
    parse_app_list(value, &DEFAULT_TERMINAL_APPS)
}

/// Create the state directory, then check that the current user owns it and
/// that it is not a symlink. A directory that other users can reach gets mode
/// 0700. The check stops another user from planting the directory first, for
/// example in the shared temp dir.
fn ensure_dir(path: &Path) -> std::io::Result<()> {
    use std::os::unix::fs::{MetadataExt, PermissionsExt};
    DirBuilder::new().recursive(true).mode(0o700).create(path)?;
    let meta = std::fs::symlink_metadata(path)?;
    // SAFETY: getuid has no preconditions and cannot fail.
    let uid = unsafe { libc::getuid() };
    if !meta.is_dir() || meta.uid() != uid {
        return Err(std::io::Error::new(
            ErrorKind::PermissionDenied,
            format!(
                "{} is not a directory that the current user owns",
                path.display()
            ),
        ));
    }
    if meta.mode() & 0o077 != 0 {
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700))?;
    }
    // A symlink in the path as written must also belong to this user or to
    // root. Otherwise its owner could point it at another directory later,
    // and the kill switch check would read the wrong place.
    for part in path.ancestors() {
        if part.as_os_str().is_empty() {
            continue;
        }
        let meta = std::fs::symlink_metadata(part)?;
        if meta.file_type().is_symlink() && meta.uid() != uid && meta.uid() != 0 {
            return Err(std::io::Error::new(
                ErrorKind::PermissionDenied,
                format!("{} is a symlink that another user owns", part.display()),
            ));
        }
    }
    // Another user who owns a parent could swap the directory after this
    // check. Every parent must belong to this user or to root, and a parent
    // that others can write must have the sticky bit, as /tmp has.
    let real = std::fs::canonicalize(path)?;
    for parent in real.ancestors().skip(1) {
        let meta = std::fs::symlink_metadata(parent)?;
        let owner_ok = meta.uid() == uid || meta.uid() == 0;
        let shared = meta.mode() & 0o022 != 0;
        let sticky = meta.mode() & 0o1000 != 0;
        if !owner_ok || (shared && !sticky) {
            return Err(std::io::Error::new(
                ErrorKind::PermissionDenied,
                format!(
                    "{} is not safe for the safety state: another user can change it",
                    parent.display()
                ),
            ));
        }
    }
    Ok(())
}

/// Create the kill switch file. Returns the file path.
pub(crate) fn engage_kill_switch(config: &PolicyConfig) -> Result<PathBuf> {
    ensure_dir(&config.state_dir).with_context(|| {
        format!(
            "failed to create state directory {}",
            config.state_dir.display()
        )
    })?;
    let path = config.stop_file();
    let mut file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .mode(0o600)
        .custom_flags(libc::O_NOFOLLOW)
        .open(&path)
        .with_context(|| format!("failed to create {}", path.display()))?;
    writeln!(file, "{}", rfc3339_utc(SystemTime::now()))
        .with_context(|| format!("failed to write {}", path.display()))?;
    Ok(path)
}

/// Remove the kill switch file. Returns true when a file was removed.
pub(crate) fn release_kill_switch(config: &PolicyConfig) -> Result<bool> {
    let path = config.stop_file();
    match std::fs::remove_file(&path) {
        Ok(()) => Ok(true),
        Err(error) if error.kind() == ErrorKind::NotFound => Ok(false),
        Err(error) => Err(error).with_context(|| format!("failed to remove {}", path.display())),
    }
}

// ---------------------------------------------------------------------------
// Classifier (pure)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Tier {
    Low,
    High,
    /// Refused in `confirm` mode with no way to confirm.
    Blocked,
}

impl Tier {
    fn as_str(self) -> &'static str {
        match self {
            Tier::Low => "low",
            Tier::High => "high",
            Tier::Blocked => "blocked",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Reason {
    CommitWord(&'static str),
    EnterInSendApp,
    EnterInTerminal,
    EnterInUnknownWindow,
    NewlineInSendApp,
    NewlineInTerminal,
    NewlineInUnknownWindow,
    ClickInSendApp,
    PasswordField,
    UnresolvedElement,
    SetupTool,
    ShellCommand,
    /// The command is too long to show in full. Refused in `confirm` mode.
    CommandTooLong,
}

impl Reason {
    /// Stable code for the audit log.
    pub(crate) fn code(&self) -> String {
        match self {
            Reason::CommitWord(word) => format!("commit_word:{word}"),
            Reason::EnterInSendApp => "enter_in_send_app".to_string(),
            Reason::EnterInTerminal => "enter_in_terminal".to_string(),
            Reason::EnterInUnknownWindow => "enter_in_unknown_window".to_string(),
            Reason::NewlineInSendApp => "newline_in_send_app".to_string(),
            Reason::NewlineInTerminal => "newline_in_terminal".to_string(),
            Reason::NewlineInUnknownWindow => "newline_in_unknown_window".to_string(),
            Reason::ClickInSendApp => "click_in_send_app".to_string(),
            Reason::PasswordField => "password_field".to_string(),
            Reason::UnresolvedElement => "unresolved_element".to_string(),
            Reason::SetupTool => "setup_tool".to_string(),
            Reason::ShellCommand => "shell_command".to_string(),
            Reason::CommandTooLong => "command_too_long".to_string(),
        }
    }

    /// Plain sentence for the confirmation dialog and for refusal messages.
    pub(crate) fn describe(&self) -> String {
        match self {
            Reason::CommitWord(word) => {
                format!("The control name or description contains the word \"{word}\".")
            }
            Reason::EnterInSendApp => "Enter can send a message in this app.".to_string(),
            Reason::EnterInTerminal => {
                "Enter in a terminal runs a shell command.".to_string()
            }
            Reason::EnterInUnknownWindow => {
                "The app that receives Enter is unknown. Enter can send a message.".to_string()
            }
            Reason::NewlineInSendApp => {
                "The text has a line break. A line break can send a message in this app."
                    .to_string()
            }
            Reason::NewlineInTerminal => {
                "The text has a line break. A line break in a terminal runs a shell command."
                    .to_string()
            }
            Reason::NewlineInUnknownWindow => {
                "The text has a line break and the receiving app is unknown. A line break can send a message."
                    .to_string()
            }
            Reason::ClickInSendApp => {
                "No known control is at the click point, and this app can send messages."
                    .to_string()
            }
            Reason::PasswordField => "The target is a password field.".to_string(),
            Reason::UnresolvedElement => {
                "The target element is not in the last accessibility snapshot.".to_string()
            }
            Reason::SetupTool => "This tool changes the desktop configuration.".to_string(),
            Reason::ShellCommand => "This tool runs a shell command.".to_string(),
            Reason::CommandTooLong => format!(
                "The shell command is longer than {RUN_SHELL_PREVIEW_MAX_CHARS} characters, so the user cannot read all of it in a confirmation."
            ),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Classification {
    pub tier: Tier,
    pub reason: Option<Reason>,
}

impl Classification {
    fn low() -> Self {
        Self {
            tier: Tier::Low,
            reason: None,
        }
    }

    fn high(reason: Reason) -> Self {
        Self {
            tier: Tier::High,
            reason: Some(reason),
        }
    }

    fn blocked(reason: Reason) -> Self {
        Self {
            tier: Tier::Blocked,
            reason: Some(reason),
        }
    }
}

/// What the gate knows about one call. It holds text only as length and
/// digest, so no typed text can reach a dialog or the audit log.
#[derive(Debug, Clone, Default)]
pub(crate) struct ActionFacts {
    pub tool: &'static str,
    pub app_id: Option<String>,
    pub wm_class: Option<String>,
    pub window_title: Option<String>,
    pub element_role: Option<String>,
    pub element_name: Option<String>,
    pub element_description: Option<String>,
    /// The call names an element that the cached tree does not hold.
    pub element_unresolved: bool,
    /// A click at a point, and no element in the cached snapshot holds the point.
    pub click_point_unresolved: bool,
    pub key: Option<String>,
    pub text_chars: Option<usize>,
    pub text_sha256: Option<String>,
    pub text_has_newline: bool,
    pub command_preview: Option<String>,
}

impl ActionFacts {
    pub(crate) fn new(tool: &'static str) -> Self {
        Self {
            tool,
            ..Self::default()
        }
    }

    pub(crate) fn with_window(
        mut self,
        app_id: Option<&str>,
        wm_class: Option<&str>,
        title: Option<&str>,
    ) -> Self {
        self.app_id = clean(app_id);
        self.wm_class = clean(wm_class);
        self.window_title = clean(title);
        self
    }

    pub(crate) fn with_element(
        mut self,
        role: Option<&str>,
        name: Option<&str>,
        description: Option<&str>,
    ) -> Self {
        self.element_role = clean(role);
        self.element_name = clean(name);
        self.element_description = clean(description);
        self
    }

    pub(crate) fn with_unresolved_element(mut self, unresolved: bool) -> Self {
        self.element_unresolved = unresolved;
        self
    }

    pub(crate) fn with_unresolved_click_point(mut self) -> Self {
        self.click_point_unresolved = true;
        self
    }

    pub(crate) fn with_key(mut self, key: &str) -> Self {
        self.key = Some(key.to_string());
        self
    }

    pub(crate) fn with_text(mut self, text: &str) -> Self {
        self.text_chars = Some(text.chars().count());
        self.text_sha256 = Some(sha256_hex(text));
        self.text_has_newline = text.contains('\n') || text.contains('\r');
        self
    }

    /// The shell command counts as text: the gate keeps length and digest.
    /// The full command is for the confirmation dialog only, and only when it
    /// is short enough to read. The audit log never holds it.
    pub(crate) fn with_command(mut self, command: &str) -> Self {
        let chars = command.chars().count();
        self.text_chars = Some(chars);
        self.text_sha256 = Some(sha256_hex(command));
        self.command_preview = (chars <= RUN_SHELL_PREVIEW_MAX_CHARS).then(|| command.to_string());
        self
    }

    fn has_window(&self) -> bool {
        self.app_id.is_some() || self.wm_class.is_some() || self.window_title.is_some()
    }

    fn app(&self) -> Option<&str> {
        self.app_id.as_deref().or(self.wm_class.as_deref())
    }

    fn element_is_password(&self) -> bool {
        self.element_role
            .as_deref()
            .is_some_and(|role| role.to_ascii_lowercase().contains("password"))
    }
}

fn clean(value: Option<&str>) -> Option<String> {
    value
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
}

pub(crate) fn sha256_hex(text: &str) -> String {
    Sha256::digest(text.as_bytes())
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

/// First commit word in `text`, in text order. Whole words only, so
/// "postman" does not match "post".
pub(crate) fn commit_word(text: &str) -> Option<&'static str> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|token| !token.is_empty())
        .find_map(|token| {
            let lower = token.to_lowercase();
            COMMIT_WORDS.iter().copied().find(|word| *word == lower)
        })
}

/// The parts of a chord as the `press_key` grammar reads them: the parts join
/// with `+`, empty parts drop out, and each part is lowercase without
/// hyphens, underscores, or spaces. The last part is the key.
fn chord_parts(key: &str) -> Vec<String> {
    key.split('+')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(|part| {
            part.to_ascii_lowercase()
                .replace(['-', '_'], "")
                .split_whitespace()
                .collect()
        })
        .collect()
}

/// True when a modifier part is the Control key.
fn is_control_modifier(part: &str) -> bool {
    matches!(part, "ctrl" | "control")
}

/// True for Return, KP_Enter, or Enter with any modifiers. It is also true
/// for `Ctrl+M` and `Ctrl+J` with any extra modifiers: a terminal reads
/// them as a carriage return and a line feed. This follows the key grammar of
/// `press_key` (parts joined with `+`, case-insensitive, hyphens and spaces
/// ignored) and also accepts `KP_Enter` spellings.
pub(crate) fn is_enter_key(key: &str) -> bool {
    let mut parts = chord_parts(key);
    let Some(last) = parts.pop() else {
        return false;
    };
    matches!(
        last.as_str(),
        "enter" | "return" | "kpenter" | "numpadenter"
    ) || (matches!(last.as_str(), "m" | "j") && parts.iter().any(|part| is_control_modifier(part)))
}

/// True for `Ctrl+O` with any extra modifiers. A shell reads it as Enter
/// (it runs the line, then recalls the next one), so it is Enter in a
/// terminal only. In other apps it is "open".
pub(crate) fn is_terminal_enter_key(key: &str) -> bool {
    let mut parts = chord_parts(key);
    parts.pop().is_some_and(|last| last == "o")
        && parts.iter().any(|part| is_control_modifier(part))
}

/// True for the space key with any modifiers. Space presses the focused
/// button, link, or checkbox.
pub(crate) fn is_space_key(key: &str) -> bool {
    chord_parts(key).pop().is_some_and(|last| last == "space")
}

/// Keys that cannot type a character: navigation, editing, and Enter. They
/// stay allowed in a password field.
const PASSWORD_SAFE_KEYS: [&str; 21] = [
    "tab",
    "isolefttab",
    "escape",
    "esc",
    "backspace",
    "delete",
    "del",
    "left",
    "right",
    "up",
    "down",
    "arrowleft",
    "arrowright",
    "arrowup",
    "arrowdown",
    "home",
    "end",
    "return",
    "enter",
    "kpenter",
    "numpadenter",
];

/// True when the key cannot type a character into a password field: a key
/// from [`PASSWORD_SAFE_KEYS`], alone or with Shift only. Any other key can
/// type or paste, so an agent could send a password with it.
pub(crate) fn is_password_safe_key(key: &str) -> bool {
    let mut parts = chord_parts(key);
    let Some(last) = parts.pop() else {
        return false;
    };
    parts.iter().all(|part| part == "shift") && PASSWORD_SAFE_KEYS.contains(&last.as_str())
}

/// True when the gate needs the focused element to rate this key. Enter and
/// space can press a focused control. Every key that is not password-safe
/// can type into a password field, so it needs the role of the focus.
pub(crate) fn key_needs_focus_check(key: &str) -> bool {
    !chord_parts(key).is_empty()
        && (is_enter_key(key) || is_space_key(key) || !is_password_safe_key(key))
}

/// The list entry that matches the window, if any. Entries are lowercase
/// substrings of the app id, the wm class, or the title.
fn app_list_match<'a>(facts: &ActionFacts, list: &'a [String]) -> Option<&'a str> {
    let fields = [
        facts.app_id.as_deref(),
        facts.wm_class.as_deref(),
        facts.window_title.as_deref(),
    ];
    list.iter()
        .find(|entry| {
            fields
                .iter()
                .flatten()
                .any(|field| field.to_lowercase().contains(entry.as_str()))
        })
        .map(String::as_str)
}

/// The send-on-Enter entry that matches the window, if any.
pub(crate) fn send_app_match<'a>(facts: &ActionFacts, send_apps: &'a [String]) -> Option<&'a str> {
    app_list_match(facts, send_apps)
}

/// The terminal entry that matches the window, if any.
pub(crate) fn terminal_app_match<'a>(
    facts: &ActionFacts,
    terminal_apps: &'a [String],
) -> Option<&'a str> {
    app_list_match(facts, terminal_apps)
}

/// First commit word in the name or the description of the element.
fn element_commit_word(facts: &ActionFacts) -> Option<&'static str> {
    [
        facts.element_name.as_deref(),
        facts.element_description.as_deref(),
    ]
    .into_iter()
    .flatten()
    .find_map(commit_word)
}

/// Rate a call that sends Enter, or text with a line break, to the window.
/// A commit word on the focused element comes first. Then a terminal, a send
/// app, and an unknown window give the reasons that the caller passes in.
fn enter_rating(
    facts: &ActionFacts,
    lists: &AppLists,
    in_terminal: Reason,
    in_send_app: Reason,
    in_unknown_window: Reason,
) -> Option<Classification> {
    if let Some(word) = element_commit_word(facts) {
        return Some(Classification::high(Reason::CommitWord(word)));
    }
    if terminal_app_match(facts, &lists.terminal).is_some() {
        return Some(Classification::high(in_terminal));
    }
    if send_app_match(facts, &lists.send).is_some() {
        return Some(Classification::high(in_send_app));
    }
    if !facts.has_window() {
        return Some(Classification::high(in_unknown_window));
    }
    None
}

/// Rate one call. Pure: it reads only `facts` and `lists`.
pub(crate) fn classify(facts: &ActionFacts, lists: &AppLists) -> Classification {
    match facts.tool {
        "setup_accessibility" | "setup_window_targeting" => Classification::high(Reason::SetupTool),
        "run_shell" => {
            if facts
                .text_chars
                .is_some_and(|chars| chars > RUN_SHELL_PREVIEW_MAX_CHARS)
            {
                return Classification::blocked(Reason::CommandTooLong);
            }
            Classification::high(Reason::ShellCommand)
        }
        "click" | "perform_action" => {
            if let Some(word) = element_commit_word(facts) {
                return Classification::high(Reason::CommitWord(word));
            }
            // A click that names a missing element fails inside the tool
            // without input, so only perform_action needs the unresolved rule.
            if facts.tool == "perform_action" && facts.element_unresolved {
                return Classification::high(Reason::UnresolvedElement);
            }
            // A click at a point that no known control holds can still press
            // a send button in a messaging app.
            if facts.tool == "click"
                && facts.click_point_unresolved
                && send_app_match(facts, &lists.send).is_some()
            {
                return Classification::high(Reason::ClickInSendApp);
            }
            Classification::low()
        }
        "set_value" => {
            if facts.element_is_password() {
                return Classification::blocked(Reason::PasswordField);
            }
            if facts.element_unresolved {
                return Classification::high(Reason::UnresolvedElement);
            }
            Classification::low()
        }
        "type_text" => {
            if facts.element_is_password() {
                return Classification::blocked(Reason::PasswordField);
            }
            if facts.text_has_newline {
                if let Some(rating) = enter_rating(
                    facts,
                    lists,
                    Reason::NewlineInTerminal,
                    Reason::NewlineInSendApp,
                    Reason::NewlineInUnknownWindow,
                ) {
                    return rating;
                }
            }
            Classification::low()
        }
        "press_key" => {
            let Some(key) = facts.key.as_deref() else {
                return Classification::low();
            };
            // The role check fails open: with no accessibility data the
            // role is unknown, and the key goes through.
            if facts.element_is_password() && !is_password_safe_key(key) {
                return Classification::blocked(Reason::PasswordField);
            }
            // Ctrl+O is Enter in a terminal. An unknown window may be one.
            let enters = is_enter_key(key)
                || (is_terminal_enter_key(key)
                    && (terminal_app_match(facts, &lists.terminal).is_some()
                        || !facts.has_window()));
            if enters {
                if let Some(rating) = enter_rating(
                    facts,
                    lists,
                    Reason::EnterInTerminal,
                    Reason::EnterInSendApp,
                    Reason::EnterInUnknownWindow,
                ) {
                    return rating;
                }
            } else if is_space_key(key) {
                if let Some(word) = element_commit_word(facts) {
                    return Classification::high(Reason::CommitWord(word));
                }
            }
            Classification::low()
        }
        _ => Classification::low(),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RefuseCause {
    Observe,
    Blocked,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Verdict {
    Allow,
    Confirm,
    Refuse(RefuseCause),
}

/// Mode decision. The kill switch is checked before this and beats every mode.
pub(crate) fn verdict(mode: Mode, tier: Tier) -> Verdict {
    match (mode, tier) {
        (Mode::Observe, _) => Verdict::Refuse(RefuseCause::Observe),
        (Mode::Full, _) => Verdict::Allow,
        (Mode::Confirm, Tier::Low) => Verdict::Allow,
        (Mode::Confirm, Tier::High) => Verdict::Confirm,
        (Mode::Confirm, Tier::Blocked) => Verdict::Refuse(RefuseCause::Blocked),
    }
}

// ---------------------------------------------------------------------------
// Audit log
// ---------------------------------------------------------------------------

/// One audit line. The record holds no typed text: only length and digest.
/// A call writes `start` and `end` lines, or one `refused` line. All lines of
/// one call share the same `call_id`.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct AuditRecord {
    pub ts: String,
    pub call_id: String,
    /// `start`, `end`, or `refused`.
    pub phase: &'static str,
    pub client: String,
    pub tool: String,
    pub mode: &'static str,
    pub tier: &'static str,
    pub decision: &'static str,
    pub app: Option<String>,
    pub window_title: Option<String>,
    pub element_name: Option<String>,
    pub key: Option<String>,
    pub text_len: Option<usize>,
    pub text_sha256: Option<String>,
    /// The result of the action. A `start` line has none.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ok: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

impl AuditRecord {
    /// A record before the call has a rating. The tier is `unknown`.
    fn new(call_id: &str, client: &str, facts: &ActionFacts, mode: Mode) -> Self {
        Self {
            ts: rfc3339_utc(SystemTime::now()),
            call_id: call_id.to_string(),
            phase: "start",
            client: truncate_chars(client, AUDIT_FIELD_MAX_CHARS),
            tool: facts.tool.to_string(),
            mode: mode.as_str(),
            tier: "unknown",
            decision: "pending",
            app: facts
                .app()
                .map(|value| truncate_chars(value, AUDIT_FIELD_MAX_CHARS)),
            window_title: facts
                .window_title
                .as_deref()
                .map(|value| truncate_chars(value, AUDIT_FIELD_MAX_CHARS)),
            element_name: facts
                .element_name
                .as_deref()
                .map(|value| truncate_chars(value, AUDIT_FIELD_MAX_CHARS)),
            key: facts.key.as_deref().map(audit_key),
            text_len: facts.text_chars,
            text_sha256: facts.text_sha256.clone(),
            ok: None,
            reason: None,
        }
    }

    fn classified(mut self, classification: &Classification) -> Self {
        self.tier = classification.tier.as_str();
        self.reason = classification.reason.as_ref().map(Reason::code);
        self
    }

    /// A copy for one line of the log, with its own time.
    fn stamped(&self, phase: &'static str, ok: Option<bool>) -> Self {
        let mut line = self.clone();
        line.ts = rfc3339_utc(SystemTime::now());
        line.phase = phase;
        line.ok = ok;
        line
    }

    fn line(&self) -> String {
        let mut line = serde_json::to_string(self).unwrap_or_else(|_| "{}".to_string());
        line.push('\n');
        line
    }
}

/// An id that is unique for each call, across server processes.
fn new_call_id() -> String {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    format!(
        "{millis:x}-{:x}-{:x}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    )
}

/// A single plain character could be one letter of a password that an agent
/// sends key by key. The audit log keeps modifiers and named keys, and hides
/// the character itself.
fn audit_key(key: &str) -> String {
    let parts: Vec<&str> = key
        .split('+')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .collect();
    let Some((last, modifiers)) = parts.split_last() else {
        return String::new();
    };
    let single = last.chars().count() == 1 && last.chars().all(char::is_alphanumeric);
    if !single {
        return truncate_chars(key, AUDIT_FIELD_MAX_CHARS);
    }
    let mut shown: Vec<&str> = modifiers.to_vec();
    shown.push("<char>");
    shown.join("+")
}

fn append_audit(file: &mut File, record: &AuditRecord) -> std::io::Result<()> {
    // One write call keeps each line whole when two servers append together.
    file.write_all(record.line().as_bytes())
}

fn open_audit(config: &PolicyConfig) -> std::io::Result<File> {
    ensure_dir(&config.state_dir)?;
    OpenOptions::new()
        .create(true)
        .append(true)
        .mode(0o600)
        .custom_flags(libc::O_NOFOLLOW)
        .open(config.audit_path())
}

/// RFC 3339 UTC time with millisecond precision.
pub(crate) fn rfc3339_utc(time: SystemTime) -> String {
    let elapsed = time.duration_since(UNIX_EPOCH).unwrap_or_default();
    let seconds = elapsed.as_secs() as i64;
    let millis = elapsed.subsec_millis();
    let days = seconds.div_euclid(86_400);
    let rest = seconds.rem_euclid(86_400);
    let (year, month, day) = civil_from_days(days);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}.{millis:03}Z",
        rest / 3600,
        rest % 3600 / 60,
        rest % 60
    )
}

/// Days since 1970-01-01 to a civil date (proleptic Gregorian calendar).
fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let day_of_era = z.rem_euclid(146_097);
    let year_of_era =
        (day_of_era - day_of_era / 1460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_index = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * month_index + 2) / 5 + 1;
    let month = if month_index < 10 {
        month_index + 3
    } else {
        month_index - 9
    };
    let year = year_of_era + era * 400 + i64::from(month <= 2);
    (year, month as u32, day as u32)
}

fn truncate_chars(value: &str, max: usize) -> String {
    if value.chars().count() <= max {
        return value.to_string();
    }
    let mut shortened: String = value.chars().take(max).collect();
    shortened.push('…');
    shortened
}

/// One line of display text: control characters become spaces and the length
/// is bounded. Agent-controlled names must not forge extra dialog lines.
fn display_text(value: &str, max: usize) -> String {
    let flat: String = value
        .chars()
        .map(|c| if c.is_control() { ' ' } else { c })
        .collect();
    let collapsed = flat.split_whitespace().collect::<Vec<_>>().join(" ");
    truncate_chars(&collapsed, max)
}

// ---------------------------------------------------------------------------
// Cross-process input lock
// ---------------------------------------------------------------------------

/// Holds the exclusive `flock`. The lock ends when this value drops.
#[derive(Debug)]
pub(crate) struct InputLock {
    _file: File,
}

async fn acquire_input_lock(path: &Path, timeout: Duration) -> Result<InputLock, String> {
    if let Some(parent) = path.parent() {
        // The runtime directory exists already. The state-directory fallback may not.
        let _ = ensure_dir(parent);
    }
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .mode(0o600)
        .custom_flags(libc::O_NOFOLLOW)
        .open(path)
        .map_err(|error| {
            format!(
                "The input lock cannot be opened ({}): {error}. No input was sent.",
                path.display()
            )
        })?;
    let deadline = Instant::now() + timeout;
    loop {
        // SAFETY: flock only reads the descriptor, which `file` keeps open.
        let result = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
        if result == 0 {
            return Ok(InputLock { _file: file });
        }
        let error = std::io::Error::last_os_error();
        if error.kind() != ErrorKind::WouldBlock {
            return Err(format!(
                "The input lock failed ({}): {error}. No input was sent.",
                path.display()
            ));
        }
        if Instant::now() >= deadline {
            return Err(
                "Another take-my-computer session is sending input and holds the input lock. No input was sent. Try again."
                    .to_string(),
            );
        }
        tokio::time::sleep(LOCK_POLL_INTERVAL).await;
    }
}

// ---------------------------------------------------------------------------
// Confirmation channels
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ElicitOutcome {
    Approved,
    Declined,
    NoAnswer,
    /// The request failed. The gate falls back to the desktop dialog.
    Failed(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum DialogOutcome {
    Approved,
    Denied,
    TimedOut,
    Unavailable(String),
}

/// The two ways to ask the user. Tests replace it so they never open a dialog.
pub(crate) trait Prompter: Send + Sync {
    /// True when the MCP client declared the elicitation capability.
    fn can_elicit(&self) -> bool;
    fn elicit<'a>(&'a self, message: &'a str, timeout: Duration) -> BoxFuture<'a, ElicitOutcome>;
    fn dialog<'a>(&'a self, text: &'a str, timeout: Duration) -> BoxFuture<'a, DialogOutcome>;
}

/// Real channels: MCP elicitation through the client, then `zenity`.
pub(crate) struct SystemPrompter {
    peer: Option<Peer<RoleServer>>,
}

impl SystemPrompter {
    pub(crate) fn new(peer: Option<Peer<RoleServer>>) -> Self {
        Self { peer }
    }
}

/// True when the client declared form elicitation. A bare capability object
/// with no mode counts as form, as in the MCP specification.
pub(crate) fn client_supports_elicitation(peer: &Peer<RoleServer>) -> bool {
    peer.peer_info().is_some_and(|info| {
        info.capabilities
            .elicitation
            .as_ref()
            .is_some_and(|capability| capability.form.is_some() || capability.url.is_none())
    })
}

impl Prompter for SystemPrompter {
    fn can_elicit(&self) -> bool {
        self.peer.as_ref().is_some_and(client_supports_elicitation)
    }

    fn elicit<'a>(&'a self, message: &'a str, timeout: Duration) -> BoxFuture<'a, ElicitOutcome> {
        Box::pin(async move {
            let Some(peer) = self.peer.as_ref() else {
                return ElicitOutcome::Failed("no client connection".to_string());
            };
            let params = ElicitRequestParams::FormElicitationParams {
                meta: None,
                message: message.to_string(),
                requested_schema: ElicitationSchema::new(BTreeMap::new()),
            };
            let request = ServerRequest::ElicitRequest(ElicitRequest::new(params));
            match tokio::time::timeout(timeout, peer.send_request(request)).await {
                Err(_) => ElicitOutcome::NoAnswer,
                Ok(Err(error)) => ElicitOutcome::Failed(error.to_string()),
                Ok(Ok(ClientResult::ElicitResult(result))) => match result.action {
                    ElicitationAction::Accept => ElicitOutcome::Approved,
                    _ => ElicitOutcome::Declined,
                },
                Ok(Ok(_)) => ElicitOutcome::Failed("unexpected client response".to_string()),
            }
        })
    }

    fn dialog<'a>(&'a self, text: &'a str, timeout: Duration) -> BoxFuture<'a, DialogOutcome> {
        Box::pin(run_zenity(text, timeout))
    }
}

/// Arguments for `zenity --question`. The cancel button has the default
/// focus, so a stray Enter key denies the action and never allows it.
fn zenity_args(text: &str, timeout: Duration) -> Vec<String> {
    vec![
        "--question".to_string(),
        "--no-markup".to_string(),
        "--default-cancel".to_string(),
        "--title=take-my-computer".to_string(),
        "--ok-label=Allow".to_string(),
        "--cancel-label=Deny".to_string(),
        "--width=440".to_string(),
        format!("--timeout={}", timeout.as_secs().max(1)),
        format!("--text={text}"),
    ]
}

/// Ask with `zenity --question`. Only exit code 0 counts as approval.
async fn run_zenity(text: &str, timeout: Duration) -> DialogOutcome {
    let has_display = ["WAYLAND_DISPLAY", "DISPLAY"]
        .iter()
        .any(|name| env::var(name).is_ok_and(|value| !value.is_empty()));
    if !has_display {
        return DialogOutcome::Unavailable("no display is available for a dialog".to_string());
    }
    let mut command = tokio::process::Command::new("zenity");
    command
        .args(zenity_args(text, timeout))
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .kill_on_drop(true);
    let mut child = match command.spawn() {
        Ok(child) => child,
        Err(error) if error.kind() == ErrorKind::NotFound => {
            return DialogOutcome::Unavailable("zenity is not installed".to_string());
        }
        Err(error) => {
            return DialogOutcome::Unavailable(format!("zenity did not start: {error}"));
        }
    };
    match tokio::time::timeout(timeout + DIALOG_PROCESS_GRACE, child.wait()).await {
        Ok(Ok(status)) => match status.code() {
            Some(0) => DialogOutcome::Approved,
            // zenity exits 5 when its own timeout fires.
            Some(5) => DialogOutcome::TimedOut,
            _ => DialogOutcome::Denied,
        },
        Ok(Err(error)) => DialogOutcome::Unavailable(format!("zenity failed: {error}")),
        Err(_) => {
            let _ = child.kill().await;
            DialogOutcome::TimedOut
        }
    }
}

/// Characters of the Unicode category Cf (format): zero-width marks, soft
/// hyphen, the bidirectional controls, and tag characters. They change how
/// text looks or acts, but they do not show. U+061C and U+180E are in the
/// list on purpose.
fn is_format_char(c: char) -> bool {
    matches!(
        c,
        '\u{00ad}'
            | '\u{0600}'..='\u{0605}'
            | '\u{061c}'
            | '\u{06dd}'
            | '\u{070f}'
            | '\u{0890}'..='\u{0891}'
            | '\u{08e2}'
            | '\u{180e}'
            | '\u{200b}'..='\u{200f}'
            | '\u{202a}'..='\u{202e}'
            | '\u{2060}'..='\u{2064}'
            | '\u{2066}'..='\u{206f}'
            | '\u{feff}'
            | '\u{fff9}'..='\u{fffb}'
            | '\u{110bd}'
            | '\u{110cd}'
            | '\u{13430}'..='\u{1343f}'
            | '\u{1bca0}'..='\u{1bca3}'
            | '\u{1d173}'..='\u{1d17a}'
            | '\u{e0001}'
            | '\u{e0020}'..='\u{e007f}'
    )
}

/// A character that ends a line. These are LF, CR, vertical tab, form feed,
/// NEL, the line separator, and the paragraph separator.
fn is_line_break_char(c: char) -> bool {
    matches!(
        c,
        '\n' | '\r' | '\u{000b}' | '\u{000c}' | '\u{0085}' | '\u{2028}' | '\u{2029}'
    )
}

/// The lines of a shell command for the confirmation. Every Unicode line
/// break starts a new line, so the user sees where each command starts and a
/// hidden break cannot forge a dialog line. CR LF counts as one break. Other
/// control characters and format characters show as `\u{XXXX}` escapes, so
/// nothing in the command is invisible. Normal text shows as it is.
fn command_display_lines(command: &str) -> Vec<String> {
    let mut lines = Vec::new();
    let mut line = String::new();
    let mut chars = command.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\r' && chars.peek() == Some(&'\n') {
            continue;
        }
        if is_line_break_char(c) {
            lines.push(std::mem::take(&mut line));
        } else if c.is_control() || is_format_char(c) {
            line.push_str(&format!("\\u{{{:04x}}}", c as u32));
        } else {
            line.push(c);
        }
    }
    lines.push(line);
    lines
}

fn prompt_text(client: &str, facts: &ActionFacts, classification: &Classification) -> String {
    let field = |value: Option<&str>| {
        value
            .map(|value| display_text(value, PROMPT_FIELD_MAX_CHARS))
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| "unknown".to_string())
    };
    let who = display_text(client, PROMPT_FIELD_MAX_CHARS);
    let who = if who.is_empty() {
        "An agent".to_string()
    } else {
        who
    };
    let mut lines = vec![
        format!("{who} wants to run {}.", facts.tool),
        String::new(),
        format!("App: {}", field(facts.app())),
        format!("Window: {}", field(facts.window_title.as_deref())),
    ];
    if let Some(name) = facts.element_name.as_deref() {
        lines.push(format!(
            "Element: {}",
            display_text(name, PROMPT_FIELD_MAX_CHARS)
        ));
    }
    if let Some(key) = facts.key.as_deref() {
        lines.push(format!(
            "Key: {}",
            display_text(key, PROMPT_FIELD_MAX_CHARS)
        ));
    }
    if let Some(chars) = facts.text_chars {
        lines.push(format!("Text: {chars} characters (content hidden)"));
    }
    if let Some(command) = facts.command_preview.as_deref() {
        // Each line has a prefix, so a line of the command cannot pose as
        // a line of this dialog.
        lines.push("Command:".to_string());
        lines.extend(
            command_display_lines(command)
                .into_iter()
                .map(|line| format!("> {line}")),
        );
    }
    if let Some(reason) = classification.reason.as_ref() {
        lines.push(format!("Why: {}", reason.describe()));
    }
    lines.push(String::new());
    lines.push("Allow this action?".to_string());
    lines.join("\n")
}

// ---------------------------------------------------------------------------
// The gate
// ---------------------------------------------------------------------------

/// A refused call. The message is for the agent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Refusal {
    pub message: String,
}

enum Confirmation {
    Approved,
    Denied(String),
    Unavailable(String),
}

/// Proof that the gate allowed a call. It holds the input lock. Call
/// [`Permit::finish`] with the result to write the `end` audit line. A permit
/// that drops without `finish` (for example a cancelled call) writes
/// `ok: false`.
#[derive(Debug)]
pub(crate) struct Permit {
    _lock: InputLock,
    audit: Option<(File, AuditRecord)>,
}

impl Permit {
    pub(crate) fn finish(mut self, ok: bool) {
        self.write_audit(ok);
    }

    fn write_audit(&mut self, ok: bool) {
        if let Some((mut file, record)) = self.audit.take() {
            // The action is done. A failed write cannot undo it.
            if let Err(error) = append_audit(&mut file, &record.stamped("end", Some(ok))) {
                eprintln!("[take-my-computer] audit log write failed: {error}");
            }
        }
    }
}

impl Drop for Permit {
    fn drop(&mut self) {
        self.write_audit(false);
    }
}

/// A call that holds the input lock and has not been decided yet. The caller
/// reads the desktop facts while it holds this value, then calls
/// [`Entry::decide`]. An entry that drops without a decision (for example a
/// cancelled call) writes a `refused` line.
#[derive(Debug)]
pub(crate) struct Entry {
    policy: Policy,
    lock: Option<InputLock>,
    call_id: String,
    client: String,
    base: ActionFacts,
}

impl Entry {
    /// Rate the call and ask the user when it needs confirmation. On success
    /// the audit `start` line is written and the caller may send input.
    pub(crate) async fn decide(
        mut self,
        facts: ActionFacts,
        prompter: &dyn Prompter,
    ) -> Result<Permit, Refusal> {
        let policy = self.policy.clone();
        let config = policy.config();
        let classification = classify(&facts, &config.apps);
        let mut record = AuditRecord::new(&self.call_id, &self.client, &facts, config.mode)
            .classified(&classification);

        // The kill switch may turn on while the call waits for the lock.
        if config.kill_switch_active() {
            return Err(self.refuse(record, "refused", policy.kill_switch_message()));
        }

        let mut decision = "allowed";
        match verdict(config.mode, classification.tier) {
            Verdict::Allow => {}
            Verdict::Refuse(RefuseCause::Observe) => {
                let message = observe_message(facts.tool);
                return Err(self.refuse(record, "refused", message));
            }
            Verdict::Refuse(RefuseCause::Blocked) => {
                let message = format!(
                    "Refused by the safety policy: {} The mode is confirm. Mode full (TAKE_MY_COMPUTER_POLICY=full) allows it.",
                    classification
                        .reason
                        .as_ref()
                        .map(Reason::describe)
                        .unwrap_or_default()
                );
                return Err(self.refuse(record, "refused", message));
            }
            Verdict::Confirm => {
                match policy
                    .confirm(&facts, &classification, &self.client, prompter)
                    .await
                {
                    Confirmation::Approved => decision = "confirmed",
                    Confirmation::Denied(message) => {
                        return Err(self.refuse(record, "denied", message));
                    }
                    Confirmation::Unavailable(message) => {
                        return Err(self.refuse(record, "refused", message));
                    }
                }
            }
        }
        // The user may press the kill switch while a dialog is open.
        if config.kill_switch_active() {
            return Err(self.refuse(record, "refused", policy.kill_switch_message()));
        }
        record.decision = decision;
        // The start line goes in before any input. If it fails, no input is sent.
        let audit_failure = |error: std::io::Error| {
            format!(
                "Refused: the audit log cannot be written ({}): {error}. No input was sent.",
                config.audit_path().display()
            )
        };
        let mut file = match open_audit(config) {
            Ok(file) => file,
            Err(error) => return Err(self.refuse(record, "refused", audit_failure(error))),
        };
        if let Err(error) = append_audit(&mut file, &record.stamped("start", None)) {
            return Err(self.refuse(record, "refused", audit_failure(error)));
        }
        let lock = self
            .lock
            .take()
            .expect("an entry holds the lock until decided");
        Ok(Permit {
            _lock: lock,
            audit: Some((file, record)),
        })
    }

    /// Write the `refused` line, release the lock, and build the refusal.
    fn refuse(&mut self, record: AuditRecord, decision: &'static str, message: String) -> Refusal {
        self.lock = None;
        self.policy.refuse(record, decision, message)
    }
}

impl Drop for Entry {
    fn drop(&mut self) {
        if self.lock.is_some() {
            let record = AuditRecord::new(
                &self.call_id,
                &self.client,
                &self.base,
                self.policy.config().mode,
            );
            self.policy.write_refused(record, "cancelled");
        }
    }
}

fn observe_message(tool: &str) -> String {
    format!(
        "Refused by the safety policy: the mode is observe, so {tool} does not run. Read-only tools still work. Mode confirm or full (TAKE_MY_COMPUTER_POLICY) allows this tool."
    )
}

#[derive(Debug, Clone)]
pub(crate) struct Policy {
    config: Arc<PolicyConfig>,
}

impl Default for Policy {
    fn default() -> Self {
        #[cfg(test)]
        {
            // Unit tests build servers freely. They must not touch the real
            // state directory, the real input lock, or a real dialog.
            let mut config = PolicyConfig::from_lookup(|_| None);
            config.mode = Mode::Full;
            config.state_dir =
                env::temp_dir().join(format!("tmc-policy-unit-test-{}", std::process::id()));
            config.lock_path = config.state_dir.join("input.lock");
            Self::new(config)
        }
        #[cfg(not(test))]
        {
            Self::new(PolicyConfig::from_env())
        }
    }
}

impl Policy {
    pub(crate) fn new(config: PolicyConfig) -> Self {
        Self {
            config: Arc::new(config),
        }
    }

    pub(crate) fn config(&self) -> &PolicyConfig {
        &self.config
    }

    pub(crate) fn mode(&self) -> Mode {
        self.config.mode
    }

    pub(crate) fn status(&self, elicitation: bool) -> PolicyStatus {
        let mut channels = Vec::new();
        if elicitation {
            channels.push("elicitation".to_string());
        }
        if executable_in_path("zenity") {
            channels.push("zenity".to_string());
        }
        PolicyStatus {
            mode: self.config.mode.as_str().to_string(),
            mode_env: POLICY_ENV.to_string(),
            kill_switch_active: self.config.kill_switch_active(),
            kill_switch_file: self.config.stop_file().display().to_string(),
            send_on_enter_apps: self.config.apps.send.clone(),
            terminal_apps: self.config.apps.terminal.clone(),
            audit_log: self.config.audit_path().display().to_string(),
            confirmation_channels: channels,
            warning: self.config.warning.clone(),
        }
    }

    /// Start one call. This is the first step of the gate. It refuses at once
    /// for the kill switch and the `observe` mode. Otherwise it waits for the
    /// input lock. The caller holds the lock until the action and its audit
    /// line are done, so no other call can send input while a dialog is open.
    /// `base` holds what the caller knows before it reads the desktop. It is
    /// for the audit line of a refusal.
    pub(crate) async fn enter(&self, base: &ActionFacts, client: &str) -> Result<Entry, Refusal> {
        let config = &self.config;
        let call_id = new_call_id();
        let record = AuditRecord::new(&call_id, client, base, config.mode);

        if config.kill_switch_active() {
            return Err(self.refuse(record, "refused", self.kill_switch_message()));
        }
        if config.mode == Mode::Observe {
            return Err(self.refuse(record, "refused", observe_message(base.tool)));
        }
        let lock = match acquire_input_lock(&config.lock_path, config.lock_timeout).await {
            Ok(lock) => lock,
            Err(message) => return Err(self.refuse(record, "refused", message)),
        };
        Ok(Entry {
            policy: self.clone(),
            lock: Some(lock),
            call_id,
            client: client.to_string(),
            base: base.clone(),
        })
    }

    /// Decide a call whose facts are known at once. Unit tests use it. The
    /// server reads desktop facts between [`Policy::enter`] and [`Entry::decide`].
    #[cfg(test)]
    pub(crate) async fn begin(
        &self,
        facts: ActionFacts,
        client: &str,
        prompter: &dyn Prompter,
    ) -> Result<Permit, Refusal> {
        self.enter(&facts, client)
            .await?
            .decide(facts, prompter)
            .await
    }

    fn kill_switch_message(&self) -> String {
        format!(
            "Refused: the kill switch is on, so no input is sent. The operator turns it off with 'computer-use-linux resume' (this removes {}).",
            self.config.stop_file().display()
        )
    }

    /// Write the audit line for a refused call and build the refusal.
    fn refuse(&self, record: AuditRecord, decision: &'static str, message: String) -> Refusal {
        self.write_refused(record, decision);
        Refusal { message }
    }

    fn write_refused(&self, mut record: AuditRecord, decision: &'static str) {
        record.decision = decision;
        let line = record.stamped("refused", Some(false));
        match open_audit(&self.config) {
            Ok(mut file) => {
                if let Err(error) = append_audit(&mut file, &line) {
                    eprintln!("[take-my-computer] audit log write failed: {error}");
                }
            }
            Err(error) => eprintln!("[take-my-computer] audit log open failed: {error}"),
        }
    }

    async fn confirm(
        &self,
        facts: &ActionFacts,
        classification: &Classification,
        client: &str,
        prompter: &dyn Prompter,
    ) -> Confirmation {
        let text = prompt_text(client, facts, classification);
        if prompter.can_elicit() {
            match prompter.elicit(&text, self.config.elicit_timeout).await {
                ElicitOutcome::Approved => return Confirmation::Approved,
                ElicitOutcome::Declined => {
                    return Confirmation::Denied(
                        "The user declined this action. It did not run.".to_string(),
                    );
                }
                ElicitOutcome::NoAnswer => {
                    return Confirmation::Denied(format!(
                        "The user did not answer within {} seconds. The action did not run.",
                        self.config.elicit_timeout.as_secs()
                    ));
                }
                // A failed request falls through to the desktop dialog.
                ElicitOutcome::Failed(_) => {}
            }
        }
        match prompter.dialog(&text, self.config.dialog_timeout).await {
            DialogOutcome::Approved => Confirmation::Approved,
            DialogOutcome::Denied => {
                Confirmation::Denied("The user denied this action. It did not run.".to_string())
            }
            DialogOutcome::TimedOut => Confirmation::Denied(format!(
                "The user did not answer within {} seconds. The action did not run.",
                self.config.dialog_timeout.as_secs()
            )),
            DialogOutcome::Unavailable(why) => Confirmation::Unavailable(format!(
                "This action needs confirmation, and no confirmation channel is available ({why}). {} Mode full (TAKE_MY_COMPUTER_POLICY=full) allows it without confirmation.",
                classification
                    .reason
                    .as_ref()
                    .map(Reason::describe)
                    .unwrap_or_default()
            )),
        }
    }
}

// ---------------------------------------------------------------------------
// Status
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, JsonSchema)]
pub(crate) struct PolicyStatus {
    /// `confirm`, `full`, or `observe`.
    pub mode: String,
    /// Environment variable that sets the mode.
    pub mode_env: String,
    /// True while the kill switch file exists. Every mutating tool refuses.
    pub kill_switch_active: bool,
    pub kill_switch_file: String,
    /// Case-insensitive substrings of app id, wm class, or title.
    pub send_on_enter_apps: Vec<String>,
    /// Terminal apps, where Enter runs a shell command. Same matching.
    pub terminal_apps: Vec<String>,
    pub audit_log: String,
    /// Channels for confirmation, in order of use.
    pub confirmation_channels: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub warning: Option<String>,
}

fn executable_in_path(program: &str) -> bool {
    use std::os::unix::fs::PermissionsExt;
    env::var_os("PATH").is_some_and(|paths| {
        env::split_paths(&paths).any(|dir| {
            std::fs::metadata(dir.join(program))
                .map(|meta| meta.is_file() && meta.permissions().mode() & 0o111 != 0)
                .unwrap_or(false)
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::Mutex;

    fn apps() -> AppLists {
        AppLists::default()
    }

    fn facts(tool: &'static str) -> ActionFacts {
        ActionFacts::new(tool)
    }

    fn in_window(tool: &'static str, app_id: &str, title: &str) -> ActionFacts {
        ActionFacts::new(tool).with_window(Some(app_id), None, Some(title))
    }

    // ----- commit words -----

    #[test]
    fn every_listed_commit_word_matches_in_any_case() {
        for word in COMMIT_WORDS {
            assert_eq!(commit_word(word), Some(word), "{word}");
            assert_eq!(commit_word(&word.to_uppercase()), Some(word), "{word}");
            let mut title_case = word.to_string();
            title_case[..1].make_ascii_uppercase();
            assert_eq!(commit_word(&title_case), Some(word), "{word}");
        }
    }

    #[test]
    fn commit_words_match_inside_longer_labels() {
        assert_eq!(commit_word("Send message"), Some("send"));
        assert_eq!(commit_word("Kirim Pesan"), Some("kirim"));
        assert_eq!(commit_word("Hapus semua"), Some("hapus"));
        assert_eq!(commit_word("Bayar sekarang"), Some("bayar"));
        assert_eq!(commit_word("Konfirmasi pesanan"), Some("konfirmasi"));
        assert_eq!(commit_word("Terbitkan artikel"), Some("terbitkan"));
        assert_eq!(commit_word("Setujui permintaan"), Some("setujui"));
        assert_eq!(commit_word("_Send"), Some("send"));
        assert_eq!(commit_word("Send & close"), Some("send"));
        assert_eq!(commit_word("Pay (USD)"), Some("pay"));
        assert_eq!(commit_word("Cancel or delete"), Some("delete"));
    }

    #[test]
    fn commit_words_do_not_match_inside_other_words() {
        for text in [
            "Postman",
            "Compost",
            "Sending",
            "Resend",
            "Payment",
            "Repay",
            "Transferred",
            "Beliau",
            "Terkirim",
            "Dikirim",
            "Postingan",
            "Cancel",
            "OK",
            "Open file",
            "Check out",
            "",
            "   ",
            "---",
        ] {
            assert_eq!(commit_word(text), None, "{text:?}");
        }
    }

    // ----- Enter keys -----

    #[test]
    fn enter_keys_match_with_any_modifier_and_spelling() {
        for key in [
            "Enter",
            "enter",
            "Return",
            "RETURN",
            "KP_Enter",
            "kp-enter",
            "KPEnter",
            "Ctrl+Enter",
            "ctrl + return",
            "Shift+Return",
            "Ctrl+Shift+Alt+Enter",
            "Super+KP_Enter",
        ] {
            assert!(is_enter_key(key), "{key}");
        }
    }

    #[test]
    fn ctrl_m_and_ctrl_j_are_enter_with_any_modifier_case_and_spacing() {
        for key in [
            "Ctrl+M",
            "ctrl+m",
            "CTRL+M",
            "Control+m",
            "ctrl + m",
            "Ctrl+J",
            "ctrl+j",
            "CONTROL + J",
            "Ctrl+Shift+M",
            "Shift+Ctrl+j",
            "Alt+Ctrl+M",
            "Super+Ctrl+Shift+J",
            "ctrl+alt+m",
            "Ctrl+Ctrl+m",
            "con-trol+m",
            " Ctrl + m ",
        ] {
            assert!(is_enter_key(key), "{key}");
        }
    }

    #[test]
    fn ctrl_m_and_ctrl_j_need_a_focus_check_like_enter() {
        for key in ["Ctrl+M", "Ctrl+J", "Ctrl+Shift+m"] {
            assert!(key_needs_focus_check(key), "{key}");
        }
    }

    #[test]
    fn other_control_chords_are_not_enter() {
        for key in [
            "Ctrl+C",
            "Ctrl+Shift+C",
            "Ctrl+K",
            "Ctrl+L",
            "Ctrl+Shift+N",
            "Ctrl+O",
            "Alt+M",
            "Shift+J",
            "Super+m",
            "m",
            "j",
            "M",
            "J",
            "Ctrl",
            "Ctrl+Tab",
            "m+Ctrl",
        ] {
            assert!(!is_enter_key(key), "{key}");
        }
    }

    #[test]
    fn ctrl_o_is_the_terminal_enter_chord() {
        for key in [
            "Ctrl+O",
            "ctrl+o",
            "Control + o",
            "Ctrl+Shift+O",
            "Alt+Ctrl+o",
        ] {
            assert!(is_terminal_enter_key(key), "{key}");
        }
        for key in [
            "o",
            "O",
            "Alt+O",
            "Shift+o",
            "Ctrl+P",
            "Ctrl+M",
            "Ctrl",
            "Ctrl+o+Tab",
            "",
        ] {
            assert!(!is_terminal_enter_key(key), "{key}");
        }
    }

    #[test]
    fn other_keys_are_not_enter() {
        for key in [
            "Escape",
            "Tab",
            "Space",
            "a",
            "Ctrl+L",
            "Ctrl+Shift+T",
            "F5",
            "Enter+Tab",
            "Entertain",
            "",
            "+",
            "Ctrl",
        ] {
            assert!(!is_enter_key(key), "{key}");
        }
    }

    // ----- classifier -----

    #[test]
    fn click_on_commit_word_is_high() {
        let cls = classify(
            &facts("click").with_element(Some("push button"), Some("Send"), None),
            &apps(),
        );
        assert_eq!(cls.tier, Tier::High);
        assert_eq!(cls.reason, Some(Reason::CommitWord("send")));
    }

    #[test]
    fn commit_word_in_description_counts() {
        let cls = classify(
            &facts("perform_action").with_element(
                Some("push button"),
                Some("Go"),
                Some("Delete the selected file"),
            ),
            &apps(),
        );
        assert_eq!(cls.tier, Tier::High);
        assert_eq!(cls.reason, Some(Reason::CommitWord("delete")));
    }

    #[test]
    fn click_on_harmless_or_lookalike_element_is_low() {
        for name in ["Cancel", "Postman", "Open", "Sending options", ""] {
            let cls = classify(
                &facts("click").with_element(Some("push button"), Some(name), None),
                &apps(),
            );
            assert_eq!(cls.tier, Tier::Low, "{name:?}");
        }
        assert_eq!(classify(&facts("click"), &apps()).tier, Tier::Low);
    }

    #[test]
    fn unresolved_element_is_high_for_perform_action_and_set_value_only() {
        let unresolved = |tool| facts(tool).with_unresolved_element(true);
        assert_eq!(
            classify(&unresolved("perform_action"), &apps()).reason,
            Some(Reason::UnresolvedElement)
        );
        assert_eq!(classify(&unresolved("set_value"), &apps()).tier, Tier::High);
        assert_eq!(classify(&unresolved("click"), &apps()).tier, Tier::Low);
    }

    #[test]
    fn enter_in_send_app_is_high() {
        for (app, title) in [
            ("org.telegram.desktop", "Telegram"),
            ("WhatsApp", "WhatsApp Web"),
            ("slack", "general - Slack"),
            ("com.discordapp.Discord", "Discord"),
            ("org.signal.Signal", "Signal"),
            ("im.riot.Riot", "Element"),
            ("teams-for-linux", "Microsoft Teams"),
            ("thunderbird", "Inbox - Thunderbird"),
            ("org.gnome.Evolution", "Evolution"),
            ("org.gnome.Geary", "Geary"),
        ] {
            for key in ["Enter", "Return", "Ctrl+Enter", "Shift+KP_Enter"] {
                let cls = classify(&in_window("press_key", app, title).with_key(key), &apps());
                assert_eq!(cls.tier, Tier::High, "{app} {key}");
                assert_eq!(cls.reason, Some(Reason::EnterInSendApp));
            }
        }
    }

    #[test]
    fn send_app_matches_wm_class_and_title_substrings() {
        let by_class = ActionFacts::new("press_key")
            .with_window(None, Some("Slack"), Some("untitled"))
            .with_key("Enter");
        assert_eq!(classify(&by_class, &apps()).tier, Tier::High);
        let by_title = ActionFacts::new("press_key")
            .with_window(Some("firefox"), None, Some("Chat - WhatsApp Web - Firefox"))
            .with_key("Enter");
        assert_eq!(classify(&by_title, &apps()).tier, Tier::High);
    }

    #[test]
    fn enter_in_other_app_is_low() {
        let cls = classify(
            &in_window("press_key", "org.gnome.gedit", "notes.txt").with_key("Enter"),
            &apps(),
        );
        assert_eq!(cls.tier, Tier::Low);
        for key in ["Escape", "Tab", "Ctrl+L"] {
            let cls = classify(
                &in_window("press_key", "org.telegram.desktop", "Telegram").with_key(key),
                &apps(),
            );
            assert_eq!(cls.tier, Tier::Low, "{key}");
        }
    }

    #[test]
    fn enter_with_unknown_window_needs_confirmation() {
        let cls = classify(&facts("press_key").with_key("Return"), &apps());
        assert_eq!(cls.tier, Tier::High);
        assert_eq!(cls.reason, Some(Reason::EnterInUnknownWindow));
        assert_eq!(
            classify(&facts("press_key").with_key("Escape"), &apps()).tier,
            Tier::Low
        );
    }

    #[test]
    fn send_app_override_replaces_the_default_list() {
        let custom = AppLists {
            send: parse_send_apps(Some(" Mattermost , ,ZULIP ")),
            ..apps()
        };
        assert_eq!(
            custom.send,
            vec!["mattermost".to_string(), "zulip".to_string()]
        );
        let in_custom = in_window("press_key", "Mattermost", "Town Square").with_key("Enter");
        assert_eq!(classify(&in_custom, &custom).tier, Tier::High);
        let in_old_default = in_window("press_key", "slack", "Slack").with_key("Enter");
        assert_eq!(classify(&in_old_default, &custom).tier, Tier::Low);
    }

    #[test]
    fn empty_send_app_override_keeps_the_default_list() {
        assert_eq!(parse_send_apps(Some("")), apps().send);
        assert_eq!(parse_send_apps(Some(" , ")), apps().send);
        assert_eq!(apps().send.len(), DEFAULT_SEND_APPS.len());
    }

    #[test]
    fn password_text_is_blocked() {
        for role in ["password text", "PasswordText", "Password Text"] {
            let typed = facts("type_text").with_element(Some(role), None, None);
            assert_eq!(classify(&typed, &apps()).tier, Tier::Blocked, "{role}");
            let set = facts("set_value").with_element(Some(role), Some("Password"), None);
            let cls = classify(&set, &apps());
            assert_eq!(cls.tier, Tier::Blocked, "{role}");
            assert_eq!(cls.reason, Some(Reason::PasswordField));
        }
        let plain = facts("type_text").with_element(Some("text"), Some("Password hint"), None);
        assert_eq!(classify(&plain, &apps()).tier, Tier::Low);
    }

    #[test]
    fn newline_in_type_text_follows_the_enter_rule() {
        let in_chat = in_window("type_text", "slack", "Slack").with_text("hello\nworld");
        assert_eq!(classify(&in_chat, &apps()).tier, Tier::High);
        let in_editor = in_window("type_text", "gedit", "notes").with_text("hello\nworld");
        assert_eq!(classify(&in_editor, &apps()).tier, Tier::Low);
        let unknown = facts("type_text").with_text("a\r\nb");
        assert_eq!(
            classify(&unknown, &apps()).reason,
            Some(Reason::NewlineInUnknownWindow)
        );
        let one_line = facts("type_text").with_text("hello world");
        assert_eq!(classify(&one_line, &apps()).tier, Tier::Low);
    }

    #[test]
    fn setup_tools_and_shell_are_high() {
        for tool in ["setup_accessibility", "setup_window_targeting", "run_shell"] {
            assert_eq!(classify(&facts(tool), &apps()).tier, Tier::High, "{tool}");
        }
    }

    #[test]
    fn plain_input_tools_are_low() {
        for tool in [
            "scroll",
            "drag",
            "activate_window",
            "move_window",
            "resize_window",
            "screenshot",
        ] {
            assert_eq!(classify(&facts(tool), &apps()).tier, Tier::Low, "{tool}");
        }
    }

    // ----- terminal apps -----

    #[test]
    fn terminal_app_list_has_a_default_and_an_override() {
        let defaults = parse_terminal_apps(None);
        assert_eq!(defaults.len(), DEFAULT_TERMINAL_APPS.len());
        for name in [
            "gnome-terminal",
            "kgx",
            "ptyxis",
            "console",
            "konsole",
            "kitty",
            "alacritty",
            "foot",
            "wezterm",
            "xterm",
            "tilix",
            "terminator",
            "ghostty",
            "terminal",
        ] {
            assert!(defaults.iter().any(|entry| entry == name), "{name}");
        }
        assert_eq!(
            parse_terminal_apps(Some(" Warp , ,ROXTERM ")),
            vec!["warp".to_string(), "roxterm".to_string()]
        );
        // An empty override must not turn the check off.
        assert_eq!(parse_terminal_apps(Some("")), defaults);
        assert_eq!(parse_terminal_apps(Some(" , ")), defaults);
    }

    #[test]
    fn terminal_override_comes_from_its_own_variable() {
        let config = PolicyConfig::from_lookup(|name| {
            (name == TERMINAL_APPS_ENV).then(|| "Warp".to_string())
        });
        assert_eq!(config.apps.terminal, vec!["warp".to_string()]);
        assert_eq!(config.apps.send, parse_send_apps(None));
        let status = Policy::new(config).status(false);
        assert_eq!(status.terminal_apps, vec!["warp".to_string()]);
    }

    #[test]
    fn enter_in_a_terminal_is_high_for_every_enter_spelling() {
        for (app, title) in [
            ("gnome-terminal-server", "Terminal"),
            ("org.gnome.Console", "Console"),
            ("org.gnome.Ptyxis", "Ptyxis"),
            ("org.kde.konsole", "Konsole"),
            ("kitty", "kitty"),
            ("Alacritty", "Alacritty"),
            ("foot", "foot"),
            ("org.wezfurlong.wezterm", "WezTerm"),
            ("XTerm", "xterm"),
            ("com.gexperts.Tilix", "Tilix"),
            ("terminator", "Terminator"),
            ("com.mitchellh.ghostty", "Ghostty"),
        ] {
            for key in ["Enter", "Return", "KP_Enter", "Ctrl+Enter", "Shift+Return"] {
                let cls = classify(&in_window("press_key", app, title).with_key(key), &apps());
                assert_eq!(cls.tier, Tier::High, "{app} {key}");
                assert_eq!(cls.reason, Some(Reason::EnterInTerminal), "{app} {key}");
            }
            let typed = in_window("type_text", app, title).with_text("ls\n");
            let cls = classify(&typed, &apps());
            assert_eq!(cls.reason, Some(Reason::NewlineInTerminal), "{app}");
        }
    }

    #[test]
    fn terminal_match_reads_wm_class_and_title_too() {
        let by_class = ActionFacts::new("press_key")
            .with_window(None, Some("Kitty"), Some("untitled"))
            .with_key("Enter");
        assert_eq!(classify(&by_class, &apps()).tier, Tier::High);
        let by_title = ActionFacts::new("press_key")
            .with_window(Some("org.example"), None, Some("user@host: ~ - Terminal"))
            .with_key("Enter");
        assert_eq!(
            classify(&by_title, &apps()).reason,
            Some(Reason::EnterInTerminal)
        );
    }

    #[test]
    fn a_terminal_override_replaces_the_default_list() {
        let custom = AppLists {
            terminal: parse_terminal_apps(Some("warp")),
            ..apps()
        };
        let in_warp = in_window("press_key", "dev.warp.Warp", "Warp").with_key("Enter");
        assert_eq!(
            classify(&in_warp, &custom).reason,
            Some(Reason::EnterInTerminal)
        );
        let in_kitty = in_window("press_key", "kitty", "kitty").with_key("Enter");
        assert_eq!(classify(&in_kitty, &custom).tier, Tier::Low);
        assert_eq!(classify(&in_kitty, &custom).reason, None);
    }

    #[test]
    fn typing_without_a_line_break_or_other_keys_in_a_terminal_is_low() {
        let plain = in_window("type_text", "kitty", "kitty").with_text("ls -la");
        assert_eq!(classify(&plain, &apps()).tier, Tier::Low);
        for key in ["Tab", "Escape", "Ctrl+L", "Up", "a"] {
            let cls = classify(
                &in_window("press_key", "kitty", "kitty").with_key(key),
                &apps(),
            );
            assert_eq!(cls.tier, Tier::Low, "{key}");
        }
    }

    #[test]
    fn enter_with_an_unknown_window_is_high_for_every_enter_spelling() {
        for key in ["Enter", "Return", "KP_Enter", "Ctrl+Return"] {
            let cls = classify(&facts("press_key").with_key(key), &apps());
            assert_eq!(cls.reason, Some(Reason::EnterInUnknownWindow), "{key}");
        }
        let typed = classify(&facts("type_text").with_text("a\nb"), &apps());
        assert_eq!(typed.reason, Some(Reason::NewlineInUnknownWindow));
    }

    #[test]
    fn enter_chords_follow_the_same_enter_rules() {
        for key in ["Ctrl+M", "ctrl+j", "Ctrl+Shift+M", "Alt+Control+J"] {
            // Terminal.
            let cls = classify(
                &in_window("press_key", "kitty", "kitty").with_key(key),
                &apps(),
            );
            assert_eq!(cls.tier, Tier::High, "{key}");
            assert_eq!(cls.reason, Some(Reason::EnterInTerminal), "{key}");
            // Send-on-Enter app.
            let cls = classify(
                &in_window("press_key", "org.telegram.desktop", "Telegram").with_key(key),
                &apps(),
            );
            assert_eq!(cls.reason, Some(Reason::EnterInSendApp), "{key}");
            // Unknown window.
            let cls = classify(&facts("press_key").with_key(key), &apps());
            assert_eq!(cls.reason, Some(Reason::EnterInUnknownWindow), "{key}");
            // Commit control.
            let facts = in_window("press_key", "gedit", "notes")
                .with_element(Some("push button"), Some("Send"), None)
                .with_key(key);
            assert_eq!(
                classify(&facts, &apps()).reason,
                Some(Reason::CommitWord("send")),
                "{key}"
            );
            // A known, harmless window stays low.
            let cls = classify(
                &in_window("press_key", "gedit", "notes").with_key(key),
                &apps(),
            );
            assert_eq!(cls.tier, Tier::Low, "{key}");
        }
    }

    #[test]
    fn other_control_chords_stay_low_in_a_terminal() {
        for key in [
            "Ctrl+C",
            "Ctrl+Shift+C",
            "Ctrl+L",
            "Ctrl+K",
            "Ctrl+R",
            "Alt+M",
        ] {
            let cls = classify(
                &in_window("press_key", "kitty", "kitty").with_key(key),
                &apps(),
            );
            assert_eq!(cls.tier, Tier::Low, "{key}");
        }
    }

    #[test]
    fn ctrl_o_is_enter_in_a_terminal_and_in_an_unknown_window_only() {
        for key in ["Ctrl+O", "ctrl+o", "Ctrl+Shift+O"] {
            let in_terminal = classify(
                &in_window("press_key", "kitty", "kitty").with_key(key),
                &apps(),
            );
            assert_eq!(in_terminal.tier, Tier::High, "{key}");
            assert_eq!(in_terminal.reason, Some(Reason::EnterInTerminal), "{key}");
            // A terminal that the window title shows is found too.
            let by_title = classify(
                &ActionFacts::new("press_key")
                    .with_window(None, Some("Foo"), Some("user@host: ~ - Terminal"))
                    .with_key(key),
                &apps(),
            );
            assert_eq!(by_title.reason, Some(Reason::EnterInTerminal), "{key}");
            // The app of an unknown window may be a terminal.
            let unknown = classify(&facts("press_key").with_key(key), &apps());
            assert_eq!(unknown.reason, Some(Reason::EnterInUnknownWindow), "{key}");
            // In a known app that is not a terminal, Ctrl+O is "open".
            for (app, title) in [
                ("gedit", "notes"),
                ("org.telegram.desktop", "Telegram"),
                ("firefox", "Mozilla Firefox"),
            ] {
                let cls = classify(&in_window("press_key", app, title).with_key(key), &apps());
                assert_eq!(cls.tier, Tier::Low, "{app} {key}");
            }
        }
        // The plain letter is never Enter.
        let plain = classify(
            &in_window("press_key", "kitty", "kitty").with_key("o"),
            &apps(),
        );
        assert_eq!(plain.tier, Tier::Low);
    }

    #[test]
    fn ctrl_o_on_a_commit_control_in_a_terminal_names_the_commit_word() {
        let facts = in_window("press_key", "kitty", "kitty")
            .with_element(Some("terminal"), Some("Send"), None)
            .with_key("Ctrl+O");
        assert_eq!(
            classify(&facts, &apps()).reason,
            Some(Reason::CommitWord("send"))
        );
    }

    // ----- keyboard activation of a control -----

    #[test]
    fn space_key_is_recognised_with_any_modifier() {
        for key in ["space", "Space", "SPACE", "Ctrl+space", "shift + Space"] {
            assert!(is_space_key(key), "{key}");
        }
        for key in ["Enter", "a", "Ctrl+L", "Spacebar2", "space+Tab", ""] {
            assert!(!is_space_key(key), "{key}");
        }
    }

    #[test]
    fn enter_and_space_on_a_commit_control_are_high() {
        for key in [
            "Return",
            "Enter",
            "KP_Enter",
            "space",
            "Ctrl+Return",
            "Shift+space",
        ] {
            for name in ["Send", "Delete account", "Kirim pesan"] {
                let facts = in_window("press_key", "gedit", "notes")
                    .with_element(Some("push button"), Some(name), None)
                    .with_key(key);
                let cls = classify(&facts, &apps());
                assert_eq!(cls.tier, Tier::High, "{key} {name}");
                assert!(
                    matches!(cls.reason, Some(Reason::CommitWord(_))),
                    "{key} {name}"
                );
            }
        }
        let by_description = in_window("press_key", "gedit", "notes")
            .with_element(Some("push button"), Some("Go"), Some("Pay now"))
            .with_key("space");
        assert_eq!(
            classify(&by_description, &apps()).reason,
            Some(Reason::CommitWord("pay"))
        );
    }

    #[test]
    fn enter_and_space_on_a_harmless_control_stay_low_outside_special_apps() {
        for key in ["Return", "space"] {
            let facts = in_window("press_key", "gedit", "notes")
                .with_element(Some("push button"), Some("Open"), None)
                .with_key(key);
            assert_eq!(classify(&facts, &apps()).tier, Tier::Low, "{key}");
        }
    }

    #[test]
    fn other_keys_on_a_commit_control_stay_low() {
        for key in ["Tab", "Escape", "a", "Ctrl+L", "Down"] {
            let facts = in_window("press_key", "gedit", "notes")
                .with_element(Some("push button"), Some("Send"), None)
                .with_key(key);
            assert_eq!(classify(&facts, &apps()).tier, Tier::Low, "{key}");
        }
    }

    #[test]
    fn a_line_break_on_a_commit_control_is_high_for_type_text() {
        let facts = in_window("type_text", "gedit", "notes")
            .with_element(Some("push button"), Some("Submit"), None)
            .with_text("ok\n");
        let cls = classify(&facts, &apps());
        assert_eq!(cls.reason, Some(Reason::CommitWord("submit")));
        // Text with no line break does not activate the control.
        let one_line = in_window("type_text", "gedit", "notes")
            .with_element(Some("text"), Some("Send to"), None)
            .with_text("ok");
        assert_eq!(classify(&one_line, &apps()).tier, Tier::Low);
    }

    #[test]
    fn key_focus_check_covers_enter_space_and_every_key_that_can_type() {
        for key in [
            "Enter",
            "Return",
            "space",
            "a",
            "7",
            "Ctrl+a",
            "Shift+Return",
            "F5",
            "Ctrl",
            "Ctrl+Shift+Tab",
            "Ctrl+Tab",
            "Ctrl+V",
            "Ctrl+O",
            "PageDown",
        ] {
            assert!(key_needs_focus_check(key), "{key}");
        }
        for key in [
            "Tab",
            "Escape",
            "Up",
            "Shift+Tab",
            "ISO_Left_Tab",
            "BackSpace",
            "Shift+Left",
            "",
            "+",
        ] {
            assert!(!key_needs_focus_check(key), "{key}");
        }
    }

    // ----- password field by single keys -----

    #[test]
    fn password_safe_keys_are_recognised() {
        for key in [
            "Tab",
            "tab",
            "ISO_Left_Tab",
            "iso-left-tab",
            "Escape",
            "Esc",
            "BackSpace",
            "Delete",
            "Left",
            "Right",
            "Up",
            "Down",
            "ArrowLeft",
            "Home",
            "End",
            "Return",
            "KP_Enter",
            "Enter",
            "Shift+Tab",
            "shift + Left",
            "Shift+Shift+Home",
            "Shift+End",
            "Shift+Return",
        ] {
            assert!(is_password_safe_key(key), "{key}");
        }
        for key in [
            "a",
            "Z",
            "7",
            "Shift+a",
            "Space",
            "Shift+Space",
            "F5",
            "PageUp",
            "Ctrl",
            "Shift",
            "Ctrl+Tab",
            "Ctrl+Left",
            "Alt+Tab",
            "Super+Up",
            "Ctrl+V",
            "Ctrl+M",
            "Ctrl+Shift+Left",
            "Tab+Tab",
            "Entertain",
            "",
            "+",
        ] {
            assert!(!is_password_safe_key(key), "{key}");
        }
    }

    #[test]
    fn typing_keys_in_a_password_field_are_blocked() {
        for key in [
            "a",
            "7",
            "Shift+a",
            "Z",
            "space",
            "Shift+Space",
            "F5",
            "PageDown",
            "Ctrl+V",
            "Ctrl+a",
            "Ctrl+Tab",
            "Alt+Left",
            "Ctrl+M",
            "Ctrl+J",
            "Ctrl+Shift+Enter",
            "Ctrl+Enter",
            "Super",
        ] {
            let facts = in_window("press_key", "firefox", "Login")
                .with_element(Some("password text"), Some("Password"), None)
                .with_key(key);
            let cls = classify(&facts, &apps());
            assert_eq!(cls.tier, Tier::Blocked, "{key}");
            assert_eq!(cls.reason, Some(Reason::PasswordField), "{key}");
        }
    }

    #[test]
    fn the_allowed_keys_in_a_password_field_are_not_blocked() {
        for key in [
            "Tab",
            "ISO_Left_Tab",
            "Shift+Tab",
            "Escape",
            "Backspace",
            "Delete",
            "Left",
            "Right",
            "Up",
            "Down",
            "Home",
            "End",
            "Shift+Left",
            "Shift+End",
        ] {
            let facts = in_window("press_key", "firefox", "Login")
                .with_element(Some("password text"), Some("Password"), None)
                .with_key(key);
            assert_eq!(classify(&facts, &apps()).tier, Tier::Low, "{key}");
        }
    }

    #[test]
    fn enter_in_a_password_field_follows_the_enter_rules() {
        for key in ["Return", "KP_Enter", "Enter", "Shift+Return"] {
            // A known window that is not special: Enter submits the form.
            let plain = in_window("press_key", "firefox", "Login")
                .with_element(Some("password text"), Some("Password"), None)
                .with_key(key);
            assert_eq!(classify(&plain, &apps()).tier, Tier::Low, "{key}");
            // A terminal still rates Enter high, never blocked.
            let terminal = in_window("press_key", "kitty", "kitty")
                .with_element(Some("password text"), Some("Password"), None)
                .with_key(key);
            let cls = classify(&terminal, &apps());
            assert_eq!(cls.reason, Some(Reason::EnterInTerminal), "{key}");
        }
    }

    #[test]
    fn single_keys_pass_when_the_role_is_unknown() {
        // Fail open: with no accessibility data, nothing shows a password field.
        let unknown = in_window("press_key", "firefox", "Login").with_key("a");
        assert_eq!(classify(&unknown, &apps()).tier, Tier::Low);
        let other_role = in_window("press_key", "firefox", "Login")
            .with_element(Some("text"), Some("Password hint"), None)
            .with_key("a");
        assert_eq!(classify(&other_role, &apps()).tier, Tier::Low);
        for key in ["space", "F5", "Ctrl+V", "Shift+a"] {
            let unknown = in_window("press_key", "firefox", "Login").with_key(key);
            assert_eq!(classify(&unknown, &apps()).tier, Tier::Low, "{key}");
        }
    }

    #[test]
    fn password_single_keys_are_refused_in_confirm_mode_only() {
        assert_eq!(
            verdict(Mode::Confirm, Tier::Blocked),
            Verdict::Refuse(RefuseCause::Blocked)
        );
        assert_eq!(verdict(Mode::Full, Tier::Blocked), Verdict::Allow);
    }

    // ----- coordinate clicks -----

    #[test]
    fn coordinate_click_on_a_commit_control_is_high() {
        // The server hit-tests the point and passes the element it found.
        let facts = in_window("click", "gedit", "notes").with_element(
            Some("push button"),
            Some("Send"),
            None,
        );
        assert_eq!(
            classify(&facts, &apps()).reason,
            Some(Reason::CommitWord("send"))
        );
    }

    #[test]
    fn coordinate_click_with_no_known_control_is_high_in_a_send_app_only() {
        let in_chat =
            in_window("click", "org.telegram.desktop", "Chat").with_unresolved_click_point();
        let cls = classify(&in_chat, &apps());
        assert_eq!(cls.tier, Tier::High);
        assert_eq!(cls.reason, Some(Reason::ClickInSendApp));
        assert_eq!(cls.reason.as_ref().unwrap().code(), "click_in_send_app");

        let in_editor = in_window("click", "gedit", "notes").with_unresolved_click_point();
        assert_eq!(classify(&in_editor, &apps()).tier, Tier::Low);
        // An unknown window keeps the current rating.
        let unknown = facts("click").with_unresolved_click_point();
        assert_eq!(classify(&unknown, &apps()).tier, Tier::Low);
    }

    #[test]
    fn coordinate_click_on_a_harmless_control_in_a_send_app_stays_low() {
        let facts = in_window("click", "org.telegram.desktop", "Chat").with_element(
            Some("push button"),
            Some("Attach"),
            None,
        );
        assert_eq!(classify(&facts, &apps()).tier, Tier::Low);
    }

    #[test]
    fn an_element_click_never_takes_the_unresolved_point_rule() {
        let facts = in_window("perform_action", "org.telegram.desktop", "Chat")
            .with_unresolved_click_point();
        assert_eq!(classify(&facts, &apps()).tier, Tier::Low);
    }

    // ----- run_shell -----

    #[test]
    fn long_shell_commands_are_blocked_in_confirm_mode() {
        let at_limit = "a".repeat(RUN_SHELL_PREVIEW_MAX_CHARS);
        let over = "a".repeat(RUN_SHELL_PREVIEW_MAX_CHARS + 1);
        let ok = classify(
            &ActionFacts::new("run_shell").with_command(&at_limit),
            &apps(),
        );
        assert_eq!(ok.reason, Some(Reason::ShellCommand));
        let blocked = classify(&ActionFacts::new("run_shell").with_command(&over), &apps());
        assert_eq!(blocked.tier, Tier::Blocked);
        assert_eq!(blocked.reason, Some(Reason::CommandTooLong));
        assert_eq!(blocked.reason.as_ref().unwrap().code(), "command_too_long");
        assert_eq!(verdict(Mode::Full, Tier::Blocked), Verdict::Allow);
    }

    #[test]
    fn the_prompt_shows_a_whole_command_of_2000_characters() {
        let command: String = (0..RUN_SHELL_PREVIEW_MAX_CHARS)
            .map(|i| char::from(b'a' + (i % 26) as u8))
            .collect();
        let facts = ActionFacts::new("run_shell").with_command(&command);
        let cls = classify(&facts, &apps());
        let text = prompt_text("agent", &facts, &cls);
        assert!(text.contains(&format!("> {command}")), "{text}");
        assert!(!text.contains('…'), "{text}");
        // A command over the limit has no preview. The gate refuses it first.
        let longer = ActionFacts::new("run_shell").with_command(&format!("{command}x"));
        assert!(longer.command_preview.is_none());
    }

    #[test]
    fn the_prompt_keeps_line_breaks_and_cannot_forge_dialog_lines() {
        let command = "echo hi\nWhy: harmless\nAllow this action? yes";
        let facts = ActionFacts::new("run_shell").with_command(command);
        let cls = classify(&facts, &apps());
        let text = prompt_text("agent", &facts, &cls);
        assert!(
            text.contains("> echo hi\n> Why: harmless\n> Allow this action? yes"),
            "{text}"
        );
        for line in text.lines() {
            assert!(
                !line.starts_with("Why: harmless") && !line.starts_with("Allow this action? yes"),
                "{text}"
            );
        }
        assert_eq!(text.lines().filter(|l| l.starts_with("Why:")).count(), 1);
    }

    #[test]
    fn the_prompt_shows_hidden_characters_as_escapes() {
        let command = "ls\r\nrm x\u{1b}[2K\u{202e}txt";
        let lines = command_display_lines(command);
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[1], "rm x\\u{001b}[2K\\u{202e}txt");
        let facts = ActionFacts::new("run_shell").with_command(command);
        let text = prompt_text("a", &facts, &classify(&facts, &apps()));
        assert!(
            !text.contains('\u{1b}') && !text.contains('\u{202e}'),
            "{text}"
        );
    }

    #[test]
    fn every_unicode_line_break_starts_a_new_line_in_the_command() {
        for (name, brk) in [
            ("LF", "\n"),
            ("CR", "\r"),
            ("CRLF", "\r\n"),
            ("NEL", "\u{0085}"),
            ("LS", "\u{2028}"),
            ("PS", "\u{2029}"),
            ("VT", "\u{000b}"),
            ("FF", "\u{000c}"),
        ] {
            let command = format!("echo one{brk}echo two");
            assert_eq!(
                command_display_lines(&command),
                vec!["echo one".to_string(), "echo two".to_string()],
                "{name}"
            );
            let facts = ActionFacts::new("run_shell").with_command(&format!(
                "echo hi{brk}Why: harmless{brk}Allow this action? yes"
            ));
            let text = prompt_text("agent", &facts, &classify(&facts, &apps()));
            for line in text.lines() {
                assert!(
                    !line.starts_with("Why: harmless")
                        && !line.starts_with("Allow this action? yes"),
                    "{name}: {text}"
                );
            }
            assert_eq!(
                text.lines().filter(|l| l.starts_with("Why:")).count(),
                1,
                "{name}"
            );
            assert!(
                text.contains("> echo hi\n> Why: harmless\n> Allow this action? yes"),
                "{name}: {text}"
            );
            for c in text.chars() {
                assert!(
                    c == '\n' || !(c.is_control() || is_line_break_char(c)),
                    "{name}: U+{:04X} in {text}",
                    c as u32
                );
            }
        }
    }

    #[test]
    fn a_line_break_pair_other_than_crlf_makes_two_breaks() {
        assert_eq!(command_display_lines("a\n\rb"), vec!["a", "", "b"]);
        assert_eq!(command_display_lines("a\r\r\nb"), vec!["a", "", "b"]);
        assert_eq!(
            command_display_lines("a\u{2028}\u{2029}b"),
            vec!["a", "", "b"]
        );
        assert_eq!(command_display_lines(""), vec![""]);
        assert_eq!(command_display_lines("a\n"), vec!["a", ""]);
    }

    #[test]
    fn control_and_format_characters_show_as_escapes_in_the_command() {
        let escaped = [
            ('\u{0000}', "\\u{0000}"),
            ('\u{0007}', "\\u{0007}"),
            ('\u{0008}', "\\u{0008}"),
            ('\t', "\\u{0009}"),
            ('\u{001b}', "\\u{001b}"),
            ('\u{007f}', "\\u{007f}"),
            ('\u{0080}', "\\u{0080}"),
            ('\u{009f}', "\\u{009f}"),
            ('\u{00ad}', "\\u{00ad}"),
            ('\u{061c}', "\\u{061c}"),
            ('\u{180e}', "\\u{180e}"),
            ('\u{200b}', "\\u{200b}"),
            ('\u{200e}', "\\u{200e}"),
            ('\u{202a}', "\\u{202a}"),
            ('\u{202e}', "\\u{202e}"),
            ('\u{2060}', "\\u{2060}"),
            ('\u{2066}', "\\u{2066}"),
            ('\u{2069}', "\\u{2069}"),
            ('\u{206f}', "\\u{206f}"),
            ('\u{feff}', "\\u{feff}"),
            ('\u{fff9}', "\\u{fff9}"),
            ('\u{e0001}', "\\u{e0001}"),
            ('\u{e0041}', "\\u{e0041}"),
            ('\u{e007f}', "\\u{e007f}"),
        ];
        for (c, shown) in escaped {
            let lines = command_display_lines(&format!("a{c}b"));
            assert_eq!(lines, vec![format!("a{shown}b")], "U+{:04X}", c as u32);
        }
    }

    #[test]
    fn normal_text_in_the_command_shows_as_it_is() {
        for command in [
            "echo 'Halo, apa kabar? Terima kasih banyak.'",
            "echo 'こんにちは 世界' > catatan.txt",
            "mv café.txt «naïve».txt",
            "echo 'Привет' | tee \"файл\"",
            "echo 😀 > emoji.txt",
            "grep -r 'ñ' ~/proyek",
            "echo \u{00a0}spasi\u{3000}lebar",
        ] {
            assert_eq!(
                command_display_lines(command),
                vec![command.to_string()],
                "{command}"
            );
        }
    }

    #[test]
    fn ordinary_characters_are_not_format_characters() {
        // A letter, a digit, a space, and non-ASCII text are not hidden.
        for c in [
            'a', 'Z', '0', ' ', 'é', 'あ', '日', '😀', '\u{00a0}', '\u{3000}',
        ] {
            assert!(!is_format_char(c), "U+{:04X}", c as u32);
            assert!(!c.is_control(), "U+{:04X}", c as u32);
        }
        for c in ['\u{200b}', '\u{061c}', '\u{180e}', '\u{e0020}'] {
            assert!(is_format_char(c), "U+{:04X}", c as u32);
        }
    }

    #[test]
    fn the_dialog_defaults_to_deny() {
        let args = zenity_args("text", Duration::from_secs(60));
        assert!(args.iter().any(|arg| arg == "--default-cancel"), "{args:?}");
        assert!(args.iter().any(|arg| arg == "--no-markup"), "{args:?}");
        assert!(args.iter().any(|arg| arg == "--question"), "{args:?}");
        assert!(args.iter().any(|arg| arg == "--timeout=60"), "{args:?}");
        assert!(args.iter().any(|arg| arg == "--text=text"), "{args:?}");
    }

    // ----- modes -----

    #[test]
    fn mode_decisions() {
        use RefuseCause::{Blocked, Observe};
        use Verdict::{Allow, Confirm, Refuse};
        let table = [
            (Mode::Confirm, Tier::Low, Allow),
            (Mode::Confirm, Tier::High, Confirm),
            (Mode::Confirm, Tier::Blocked, Refuse(Blocked)),
            (Mode::Full, Tier::Low, Allow),
            (Mode::Full, Tier::High, Allow),
            (Mode::Full, Tier::Blocked, Allow),
            (Mode::Observe, Tier::Low, Refuse(Observe)),
            (Mode::Observe, Tier::High, Refuse(Observe)),
            (Mode::Observe, Tier::Blocked, Refuse(Observe)),
        ];
        for (mode, tier, expected) in table {
            assert_eq!(verdict(mode, tier), expected, "{mode:?} {tier:?}");
        }
    }

    #[test]
    fn mode_parsing() {
        assert_eq!(Mode::parse(None), (Mode::Confirm, None));
        assert_eq!(Mode::parse(Some("")), (Mode::Confirm, None));
        assert_eq!(Mode::parse(Some("  ")), (Mode::Confirm, None));
        assert_eq!(Mode::parse(Some("confirm")), (Mode::Confirm, None));
        assert_eq!(Mode::parse(Some("FULL")), (Mode::Full, None));
        assert_eq!(Mode::parse(Some(" Observe ")), (Mode::Observe, None));
        let (mode, warning) = Mode::parse(Some("yolo"));
        assert_eq!(mode, Mode::Confirm);
        assert!(warning.unwrap().contains("yolo"));
    }

    #[test]
    fn config_reads_paths_from_the_environment() {
        let lookup = |vars: &'static [(&'static str, &'static str)]| {
            move |name: &str| {
                vars.iter()
                    .find(|(key, _)| *key == name)
                    .map(|(_, value)| (*value).to_string())
            }
        };
        let config = PolicyConfig::from_lookup(lookup(&[
            ("XDG_STATE_HOME", "/state"),
            ("XDG_RUNTIME_DIR", "/run/user/1000"),
            ("TAKE_MY_COMPUTER_POLICY", "observe"),
        ]));
        assert_eq!(config.mode, Mode::Observe);
        assert_eq!(config.state_dir, PathBuf::from("/state/take-my-computer"));
        assert_eq!(
            config.stop_file(),
            PathBuf::from("/state/take-my-computer/STOP")
        );
        assert_eq!(
            config.audit_path(),
            PathBuf::from("/state/take-my-computer/audit.jsonl")
        );
        assert_eq!(
            config.lock_path,
            PathBuf::from("/run/user/1000/take-my-computer-input.lock")
        );

        let home_only = PolicyConfig::from_lookup(lookup(&[("HOME", "/home/rama")]));
        assert_eq!(home_only.mode, Mode::Confirm);
        assert_eq!(
            home_only.state_dir,
            PathBuf::from("/home/rama/.local/state/take-my-computer")
        );
        assert_eq!(
            home_only.lock_path,
            PathBuf::from("/home/rama/.local/state/take-my-computer/input.lock")
        );

        // Without HOME, the private runtime dir comes before the shared temp dir.
        let runtime_only =
            PolicyConfig::from_lookup(lookup(&[("XDG_RUNTIME_DIR", "/run/user/1000")]));
        assert_eq!(
            runtime_only.state_dir,
            PathBuf::from("/run/user/1000/take-my-computer-state/take-my-computer")
        );

        // A relative XDG_STATE_HOME is invalid per the XDG specification.
        let relative = PolicyConfig::from_lookup(lookup(&[
            ("XDG_STATE_HOME", "relative/dir"),
            ("HOME", "/home/rama"),
        ]));
        assert_eq!(
            relative.state_dir,
            PathBuf::from("/home/rama/.local/state/take-my-computer")
        );
    }

    // ----- time -----

    #[test]
    fn rfc3339_formats_known_instants() {
        let at = |secs: u64, millis: u64| UNIX_EPOCH + Duration::from_millis(secs * 1000 + millis);
        assert_eq!(rfc3339_utc(at(0, 0)), "1970-01-01T00:00:00.000Z");
        assert_eq!(rfc3339_utc(at(951_782_400, 5)), "2000-02-29T00:00:00.005Z");
        assert_eq!(
            rfc3339_utc(at(1_700_000_000, 123)),
            "2023-11-14T22:13:20.123Z"
        );
        assert_eq!(
            rfc3339_utc(at(4_102_444_799, 999)),
            "2099-12-31T23:59:59.999Z"
        );
    }

    #[test]
    fn audit_key_hides_single_characters_only() {
        assert_eq!(audit_key("a"), "<char>");
        assert_eq!(audit_key("7"), "<char>");
        assert_eq!(audit_key("Ctrl+L"), "Ctrl+<char>");
        assert_eq!(audit_key("Enter"), "Enter");
        assert_eq!(audit_key("Ctrl+Shift+Enter"), "Ctrl+Shift+Enter");
        assert_eq!(audit_key("F5"), "F5");
    }

    // ----- prompt text -----

    #[test]
    fn prompt_shows_length_and_never_the_text() {
        let secret = "correct horse battery staple";
        let facts = in_window("type_text", "slack", "Slack").with_text(secret);
        let cls = classify(&facts, &apps());
        let text = prompt_text("claude-code", &facts, &cls);
        assert!(text.contains("28 characters"), "{text}");
        assert!(!text.contains("correct horse"), "{text}");
        assert!(text.contains("App: slack"), "{text}");
        assert!(text.contains("Window: Slack"), "{text}");
        assert!(
            text.contains("claude-code wants to run type_text."),
            "{text}"
        );
    }

    #[test]
    fn prompt_flattens_forged_lines_in_agent_controlled_names() {
        let facts = ActionFacts::new("click")
            .with_window(Some("app"), None, Some("a\nWhy: harmless\nAllow? yes"))
            .with_element(
                Some("push button"),
                Some("Send\u{7}\nElement: Cancel"),
                None,
            );
        let cls = classify(&facts, &apps());
        let text = prompt_text("agent\nWhy: fake", &facts, &cls);
        for line in text.lines() {
            assert!(
                !line.starts_with("Why: harmless") && !line.starts_with("Element: Cancel"),
                "{text}"
            );
        }
        assert_eq!(text.lines().filter(|l| l.starts_with("Why:")).count(), 1);
    }

    #[test]
    fn prompt_shows_a_command_preview_but_the_facts_keep_only_a_digest() {
        let facts = ActionFacts::new("run_shell").with_command("ls -la /tmp");
        assert_eq!(facts.text_chars, Some(11));
        assert_eq!(facts.text_sha256.as_deref().map(str::len), Some(64));
        let cls = classify(&facts, &apps());
        assert!(prompt_text("a", &facts, &cls).contains("Command:\n> ls -la /tmp"));
    }

    // ----- gate -----

    struct FakePrompter {
        elicit: Option<ElicitOutcome>,
        dialog: DialogOutcome,
        elicit_calls: AtomicUsize,
        dialog_calls: AtomicUsize,
        last_text: Mutex<String>,
        during_dialog: Option<Box<dyn Fn() + Send + Sync>>,
    }

    impl FakePrompter {
        fn new(elicit: Option<ElicitOutcome>, dialog: DialogOutcome) -> Self {
            Self {
                elicit,
                dialog,
                elicit_calls: AtomicUsize::new(0),
                dialog_calls: AtomicUsize::new(0),
                last_text: Mutex::new(String::new()),
                during_dialog: None,
            }
        }

        fn none() -> Self {
            Self::new(
                None,
                DialogOutcome::Unavailable("zenity is not installed".to_string()),
            )
        }

        fn calls(&self) -> (usize, usize) {
            (
                self.elicit_calls.load(Ordering::SeqCst),
                self.dialog_calls.load(Ordering::SeqCst),
            )
        }
    }

    impl Prompter for FakePrompter {
        fn can_elicit(&self) -> bool {
            self.elicit.is_some()
        }

        fn elicit<'a>(
            &'a self,
            message: &'a str,
            _timeout: Duration,
        ) -> BoxFuture<'a, ElicitOutcome> {
            self.elicit_calls.fetch_add(1, Ordering::SeqCst);
            *self.last_text.lock().unwrap() = message.to_string();
            if let Some(hook) = &self.during_dialog {
                hook();
            }
            let outcome = self.elicit.clone().expect("elicitation is not declared");
            Box::pin(async move { outcome })
        }

        fn dialog<'a>(&'a self, text: &'a str, _timeout: Duration) -> BoxFuture<'a, DialogOutcome> {
            self.dialog_calls.fetch_add(1, Ordering::SeqCst);
            *self.last_text.lock().unwrap() = text.to_string();
            if let Some(hook) = &self.during_dialog {
                hook();
            }
            let outcome = self.dialog.clone();
            Box::pin(async move { outcome })
        }
    }

    #[test]
    fn the_state_dir_must_not_be_a_symlink() {
        let dir = TestDir::new("symlink-state");
        let real = dir.0.join("real");
        std::fs::create_dir_all(&real).unwrap();
        let link = dir.0.join("link");
        std::os::unix::fs::symlink(&real, &link).unwrap();
        let error = ensure_dir(&link).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::PermissionDenied);
    }

    #[test]
    fn a_parent_that_others_can_write_without_sticky_bit_is_refused() {
        use std::os::unix::fs::PermissionsExt;
        let dir = TestDir::new("open-parent");
        let parent = dir.0.join("shared");
        std::fs::create_dir_all(&parent).unwrap();
        std::fs::set_permissions(&parent, std::fs::Permissions::from_mode(0o777)).unwrap();
        let error = ensure_dir(&parent.join("take-my-computer")).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::PermissionDenied);
        // The sticky bit makes a shared parent safe, as for /tmp.
        std::fs::set_permissions(&parent, std::fs::Permissions::from_mode(0o1777)).unwrap();
        ensure_dir(&parent.join("take-my-computer")).unwrap();
    }

    #[test]
    fn a_symlink_that_another_user_owns_is_refused_in_the_path() {
        // Only root can give a symlink to another user, so the test needs root.
        // SAFETY: getuid has no preconditions and cannot fail.
        if unsafe { libc::getuid() } != 0 {
            return;
        }
        let dir = TestDir::new("foreign-link");
        let real = dir.0.join("real");
        std::fs::create_dir_all(&real).unwrap();
        let link = dir.0.join("link");
        std::os::unix::fs::symlink(&real, &link).unwrap();
        std::os::unix::fs::lchown(&link, Some(65534), Some(65534)).unwrap();
        let error = ensure_dir(&link.join("take-my-computer")).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::PermissionDenied);
        // The same symlink is fine when the user owns it.
        std::os::unix::fs::lchown(&link, Some(0), Some(0)).unwrap();
        ensure_dir(&link.join("take-my-computer")).unwrap();
    }

    #[test]
    fn without_home_the_state_dir_comes_from_passwd_not_the_temp_dir() {
        let home = passwd_home().expect("the test user has a passwd entry");
        assert!(home.is_absolute());
        let config = PolicyConfig::from_lookup(|_| None);
        assert_eq!(
            config.state_dir,
            home.join(".local/state").join("take-my-computer")
        );
        assert!(!config.state_dir.starts_with(env::temp_dir()));
    }

    #[test]
    fn an_open_state_dir_gets_mode_0700() {
        use std::os::unix::fs::PermissionsExt;
        let dir = TestDir::new("open-state");
        std::fs::create_dir_all(&dir.0).unwrap();
        std::fs::set_permissions(&dir.0, std::fs::Permissions::from_mode(0o755)).unwrap();
        ensure_dir(&dir.0).unwrap();
        let mode = std::fs::metadata(&dir.0).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o700);
    }

    #[test]
    fn the_audit_log_is_not_opened_through_a_symlink() {
        let dir = TestDir::new("symlink-audit");
        let mut config = PolicyConfig::from_lookup(|_| None);
        config.state_dir = dir.0.join("state");
        ensure_dir(&config.state_dir).unwrap();
        let target = dir.0.join("elsewhere.jsonl");
        std::os::unix::fs::symlink(&target, config.audit_path()).unwrap();
        assert!(open_audit(&config).is_err());
        assert!(!target.exists());
    }

    struct TestDir(PathBuf);

    impl TestDir {
        fn new(label: &str) -> Self {
            static NEXT: AtomicUsize = AtomicUsize::new(0);
            let path = env::temp_dir().join(format!(
                "tmc-policy-test-{}-{}-{label}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::SeqCst)
            ));
            let _ = std::fs::remove_dir_all(&path);
            Self(path)
        }

        fn policy(&self, mode: Mode) -> Policy {
            let mut config = PolicyConfig::from_lookup(|_| None);
            config.mode = mode;
            config.state_dir = self.0.join("state");
            config.lock_path = self.0.join("run").join("input.lock");
            config.lock_timeout = Duration::from_millis(150);
            Policy::new(config)
        }
    }

    impl Drop for TestDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn audit_lines(policy: &Policy) -> Vec<serde_json::Value> {
        std::fs::read_to_string(policy.config().audit_path())
            .unwrap_or_default()
            .lines()
            .map(|line| serde_json::from_str(line).expect("each audit line is JSON"))
            .collect()
    }

    /// `phase:decision` of every audit line, in order.
    fn decisions_by_phase(policy: &Policy) -> Vec<String> {
        audit_lines(policy)
            .iter()
            .map(|line| {
                format!(
                    "{}:{}",
                    line["phase"].as_str().unwrap(),
                    line["decision"].as_str().unwrap()
                )
            })
            .collect()
    }

    fn click_send() -> ActionFacts {
        in_window("click", "org.telegram.desktop", "Chat").with_element(
            Some("push button"),
            Some("Send"),
            None,
        )
    }

    #[tokio::test]
    async fn full_mode_allows_everything_and_audits_without_plain_text() {
        let dir = TestDir::new("full");
        let policy = dir.policy(Mode::Full);
        let secret = "hunter2-do-not-log";
        let facts = in_window("type_text", "firefox", "Login")
            .with_element(Some("password text"), Some("Password"), None)
            .with_text(secret);
        let prompter = FakePrompter::none();
        let permit = policy.begin(facts, "claude-code", &prompter).await.unwrap();
        permit.finish(true);

        assert_eq!(prompter.calls(), (0, 0));
        let lines = audit_lines(&policy);
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0]["phase"], "start");
        assert!(lines[0].get("ok").is_none());
        assert_eq!(lines[1]["phase"], "end");
        assert_eq!(lines[0]["call_id"], lines[1]["call_id"]);
        let line = &lines[1];
        assert_eq!(line["tool"], "type_text");
        assert_eq!(line["mode"], "full");
        assert_eq!(line["tier"], "blocked");
        assert_eq!(line["decision"], "allowed");
        assert_eq!(line["client"], "claude-code");
        assert_eq!(line["app"], "firefox");
        assert_eq!(line["window_title"], "Login");
        assert_eq!(line["element_name"], "Password");
        assert_eq!(line["text_len"], secret.len());
        assert_eq!(line["text_sha256"], sha256_hex(secret));
        assert_eq!(line["ok"], true);
        assert_eq!(line["reason"], "password_field");
        assert!(line["ts"].as_str().unwrap().ends_with('Z'));
        let raw = std::fs::read_to_string(policy.config().audit_path()).unwrap();
        assert!(!raw.contains(secret), "{raw}");
        assert!(!raw.contains("hunter2"), "{raw}");
    }

    #[tokio::test]
    async fn audit_line_has_exactly_the_documented_fields() {
        let dir = TestDir::new("shape");
        let policy = dir.policy(Mode::Full);
        let facts = in_window("press_key", "gedit", "notes").with_key("Ctrl+S");
        policy
            .begin(facts, "c", &FakePrompter::none())
            .await
            .unwrap()
            .finish(true);
        let lines = audit_lines(&policy);
        assert_eq!(lines.len(), 2);
        let field_names = |line: &serde_json::Value| {
            let mut keys: Vec<String> = line.as_object().unwrap().keys().cloned().collect();
            keys.sort_unstable();
            keys
        };
        let common = [
            "app",
            "call_id",
            "client",
            "decision",
            "element_name",
            "key",
            "mode",
            "phase",
            "text_len",
            "text_sha256",
            "tier",
            "tool",
            "ts",
            "window_title",
        ];
        // The start line has no result yet. The end line adds `ok`.
        assert_eq!(field_names(&lines[0]), common);
        let mut with_ok = common.to_vec();
        with_ok.push("ok");
        with_ok.sort_unstable();
        assert_eq!(field_names(&lines[1]), with_ok);
        for line in &lines {
            assert_eq!(line["key"], "Ctrl+<char>");
            assert!(line["text_len"].is_null());
        }
    }

    #[tokio::test]
    async fn dropped_permit_audits_ok_false() {
        let dir = TestDir::new("drop");
        let policy = dir.policy(Mode::Full);
        let permit = policy
            .begin(facts("scroll"), "c", &FakePrompter::none())
            .await
            .unwrap();
        drop(permit);
        let lines = audit_lines(&policy);
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0]["phase"], "start");
        assert_eq!(lines[1]["phase"], "end");
        assert_eq!(lines[1]["ok"], false);
        assert_eq!(lines[1]["decision"], "allowed");
    }

    #[tokio::test]
    async fn observe_mode_refuses_before_any_prompt_and_audits_refusal() {
        let dir = TestDir::new("observe");
        let policy = dir.policy(Mode::Observe);
        let prompter = FakePrompter::none();
        let refusal = policy
            .begin(facts("click"), "claude-code", &prompter)
            .await
            .unwrap_err();
        assert!(refusal.message.contains("observe"), "{}", refusal.message);
        assert!(
            refusal.message.contains("TAKE_MY_COMPUTER_POLICY"),
            "{}",
            refusal.message
        );
        assert_eq!(prompter.calls(), (0, 0));
        let lines = audit_lines(&policy);
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0]["decision"], "refused");
        assert_eq!(lines[0]["phase"], "refused");
        assert_eq!(lines[0]["mode"], "observe");
        assert_eq!(lines[0]["ok"], false);
    }

    #[tokio::test]
    async fn confirm_mode_runs_low_impact_input_without_a_prompt() {
        let dir = TestDir::new("low");
        let policy = dir.policy(Mode::Confirm);
        let prompter = FakePrompter::none();
        let facts = in_window("click", "gedit", "notes").with_element(
            Some("push button"),
            Some("Open"),
            None,
        );
        policy
            .begin(facts, "c", &prompter)
            .await
            .unwrap()
            .finish(true);
        assert_eq!(prompter.calls(), (0, 0));
        assert_eq!(audit_lines(&policy)[0]["tier"], "low");
    }

    #[tokio::test]
    async fn confirm_mode_without_any_channel_refuses_and_names_full_mode() {
        let dir = TestDir::new("nochannel");
        let policy = dir.policy(Mode::Confirm);
        let prompter = FakePrompter::none();
        let refusal = policy
            .begin(click_send(), "c", &prompter)
            .await
            .unwrap_err();
        assert!(
            refusal.message.contains("TAKE_MY_COMPUTER_POLICY=full"),
            "{}",
            refusal.message
        );
        assert!(
            refusal.message.contains("no confirmation channel"),
            "{}",
            refusal.message
        );
        assert_eq!(prompter.calls(), (0, 1));
        let lines = audit_lines(&policy);
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0]["decision"], "refused");
        assert_eq!(lines[0]["phase"], "refused");
        assert_eq!(lines[0]["tier"], "high");
        assert_eq!(lines[0]["reason"], "commit_word:send");
    }

    #[tokio::test]
    async fn elicitation_answers_come_first_and_stop_the_dialog() {
        let dir = TestDir::new("elicit");
        let policy = dir.policy(Mode::Confirm);

        let approve = FakePrompter::new(Some(ElicitOutcome::Approved), DialogOutcome::Denied);
        let permit = policy.begin(click_send(), "c", &approve).await.unwrap();
        permit.finish(true);
        assert_eq!(approve.calls(), (1, 0));

        let decline = FakePrompter::new(Some(ElicitOutcome::Declined), DialogOutcome::Approved);
        let refusal = policy.begin(click_send(), "c", &decline).await.unwrap_err();
        assert!(refusal.message.contains("declined"), "{}", refusal.message);
        assert_eq!(decline.calls(), (1, 0));

        let silent = FakePrompter::new(Some(ElicitOutcome::NoAnswer), DialogOutcome::Approved);
        let refusal = policy.begin(click_send(), "c", &silent).await.unwrap_err();
        assert!(
            refusal.message.contains("did not answer"),
            "{}",
            refusal.message
        );
        assert_eq!(silent.calls(), (1, 0));

        assert_eq!(
            decisions_by_phase(&policy),
            [
                "start:confirmed",
                "end:confirmed",
                "refused:denied",
                "refused:denied"
            ]
        );
    }

    #[tokio::test]
    async fn failed_elicitation_falls_back_to_the_dialog() {
        let dir = TestDir::new("fallback");
        let policy = dir.policy(Mode::Confirm);
        let prompter = FakePrompter::new(
            Some(ElicitOutcome::Failed("method not found".to_string())),
            DialogOutcome::Approved,
        );
        let permit = policy.begin(click_send(), "c", &prompter).await.unwrap();
        permit.finish(true);
        assert_eq!(prompter.calls(), (1, 1));
        assert_eq!(audit_lines(&policy)[0]["decision"], "confirmed");
        assert_eq!(audit_lines(&policy)[0]["phase"], "start");
    }

    #[tokio::test]
    async fn dialog_outcomes_map_to_decisions() {
        let dir = TestDir::new("dialog");
        let policy = dir.policy(Mode::Confirm);

        let allow = FakePrompter::new(None, DialogOutcome::Approved);
        policy
            .begin(click_send(), "c", &allow)
            .await
            .unwrap()
            .finish(true);

        let deny = FakePrompter::new(None, DialogOutcome::Denied);
        let refusal = policy.begin(click_send(), "c", &deny).await.unwrap_err();
        assert!(refusal.message.contains("denied"), "{}", refusal.message);

        let late = FakePrompter::new(None, DialogOutcome::TimedOut);
        let refusal = policy.begin(click_send(), "c", &late).await.unwrap_err();
        assert!(
            refusal.message.contains("60 seconds"),
            "{}",
            refusal.message
        );

        assert_eq!(
            decisions_by_phase(&policy),
            [
                "start:confirmed",
                "end:confirmed",
                "refused:denied",
                "refused:denied"
            ]
        );
    }

    #[tokio::test]
    async fn dialog_text_never_holds_typed_text() {
        let dir = TestDir::new("dialogtext");
        let policy = dir.policy(Mode::Confirm);
        let prompter = FakePrompter::new(None, DialogOutcome::Approved);
        let facts = in_window("type_text", "slack", "general - Slack")
            .with_text("line one\nsecret-token-123");
        policy
            .begin(facts, "c", &prompter)
            .await
            .unwrap()
            .finish(true);
        let shown = prompter.last_text.lock().unwrap().clone();
        assert!(shown.contains("25 characters"), "{shown}");
        assert!(!shown.contains("secret-token"), "{shown}");
    }

    #[tokio::test]
    async fn password_input_is_refused_in_confirm_mode_without_a_prompt() {
        let dir = TestDir::new("password");
        let policy = dir.policy(Mode::Confirm);
        let prompter = FakePrompter::new(Some(ElicitOutcome::Approved), DialogOutcome::Approved);
        let facts = in_window("type_text", "firefox", "Login")
            .with_element(Some("password text"), None, None)
            .with_text("pw");
        let refusal = policy.begin(facts, "c", &prompter).await.unwrap_err();
        assert!(refusal.message.contains("password"), "{}", refusal.message);
        assert!(
            refusal.message.contains("TAKE_MY_COMPUTER_POLICY=full"),
            "{}",
            refusal.message
        );
        assert_eq!(prompter.calls(), (0, 0));
        assert_eq!(audit_lines(&policy)[0]["decision"], "refused");
    }

    #[tokio::test]
    async fn kill_switch_refuses_in_every_mode_until_resumed() {
        let dir = TestDir::new("kill");
        for mode in [Mode::Full, Mode::Confirm, Mode::Observe] {
            let policy = dir.policy(mode);
            assert!(!policy.config().kill_switch_active());
            let path = engage_kill_switch(policy.config()).unwrap();
            assert_eq!(path, policy.config().stop_file());
            assert!(policy.config().kill_switch_active());
            assert!(policy.status(false).kill_switch_active);

            let prompter = FakePrompter::none();
            let refusal = policy
                .begin(facts("scroll"), "c", &prompter)
                .await
                .unwrap_err();
            assert!(
                refusal.message.contains("kill switch"),
                "{}",
                refusal.message
            );
            assert_eq!(prompter.calls(), (0, 0));

            assert!(release_kill_switch(policy.config()).unwrap());
            assert!(!release_kill_switch(policy.config()).unwrap());
            assert!(!policy.config().kill_switch_active());
        }
        // Observe mode allows nothing, so only full and confirm run after resume.
        let policy = dir.policy(Mode::Full);
        policy
            .begin(facts("scroll"), "c", &FakePrompter::none())
            .await
            .unwrap()
            .finish(true);
        let refused = audit_lines(&policy)
            .iter()
            .filter(|line| line["decision"] == "refused")
            .count();
        assert_eq!(refused, 3);
    }

    #[tokio::test]
    async fn kill_switch_pressed_during_a_dialog_still_blocks_the_action() {
        let dir = TestDir::new("killdialog");
        let policy = dir.policy(Mode::Confirm);
        let config = policy.config().clone();
        let mut prompter = FakePrompter::new(None, DialogOutcome::Approved);
        prompter.during_dialog = Some(Box::new(move || {
            engage_kill_switch(&config).unwrap();
        }));
        let refusal = policy
            .begin(click_send(), "c", &prompter)
            .await
            .unwrap_err();
        assert!(
            refusal.message.contains("kill switch"),
            "{}",
            refusal.message
        );
    }

    #[tokio::test]
    async fn a_second_session_cannot_send_input_while_the_lock_is_held() {
        let dir = TestDir::new("lock");
        let policy = dir.policy(Mode::Full);
        let prompter = FakePrompter::none();
        let first = policy.begin(facts("scroll"), "a", &prompter).await.unwrap();
        let refusal = policy
            .begin(facts("scroll"), "b", &prompter)
            .await
            .unwrap_err();
        assert!(
            refusal.message.contains("input lock"),
            "{}",
            refusal.message
        );
        first.finish(true);
        policy
            .begin(facts("scroll"), "b", &prompter)
            .await
            .unwrap()
            .finish(true);
    }

    #[tokio::test]
    async fn state_files_are_private() {
        use std::os::unix::fs::PermissionsExt;
        let dir = TestDir::new("modes");
        let policy = dir.policy(Mode::Full);
        policy
            .begin(facts("scroll"), "c", &FakePrompter::none())
            .await
            .unwrap()
            .finish(true);
        engage_kill_switch(policy.config()).unwrap();
        let mode = |path: &Path| std::fs::metadata(path).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode(&policy.config().state_dir), 0o700);
        assert_eq!(mode(&policy.config().audit_path()), 0o600);
        assert_eq!(mode(&policy.config().stop_file()), 0o600);
        assert_eq!(mode(policy.config().lock_path.parent().unwrap()), 0o700);
    }

    // ----- lock order and audit phases -----

    /// True when no other holder has the lock. The check takes the lock and
    /// drops it again at once.
    fn lock_is_free(path: &Path) -> bool {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .mode(0o600)
            .open(path)
            .unwrap();
        // SAFETY: flock only reads the descriptor, which `file` keeps open.
        unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) == 0 }
    }

    #[tokio::test]
    async fn the_input_lock_is_held_while_the_dialog_is_open() {
        let dir = TestDir::new("lockdialog");
        let policy = dir.policy(Mode::Confirm);
        let lock_path = policy.config().lock_path.clone();
        let free_during_dialog = Arc::new(AtomicBool::new(true));
        let seen = Arc::clone(&free_during_dialog);
        let mut prompter = FakePrompter::new(None, DialogOutcome::Approved);
        prompter.during_dialog = Some(Box::new(move || {
            if !lock_is_free(&lock_path) {
                seen.store(false, Ordering::SeqCst);
            }
        }));
        policy
            .begin(click_send(), "c", &prompter)
            .await
            .unwrap()
            .finish(true);
        assert_eq!(prompter.calls(), (0, 1));
        assert!(
            !free_during_dialog.load(Ordering::SeqCst),
            "the lock was free while the dialog was open"
        );
    }

    #[tokio::test]
    async fn the_input_lock_is_held_while_an_elicitation_is_open() {
        let dir = TestDir::new("lockelicit");
        let policy = dir.policy(Mode::Confirm);
        let lock_path = policy.config().lock_path.clone();
        let free = Arc::new(AtomicBool::new(true));
        let seen = Arc::clone(&free);
        let mut prompter = FakePrompter::new(Some(ElicitOutcome::Approved), DialogOutcome::Denied);
        prompter.during_dialog = Some(Box::new(move || {
            if !lock_is_free(&lock_path) {
                seen.store(false, Ordering::SeqCst);
            }
        }));
        policy
            .begin(click_send(), "c", &prompter)
            .await
            .unwrap()
            .finish(true);
        assert!(!free.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn another_server_cannot_send_input_while_a_dialog_is_open() {
        let dir = TestDir::new("lockother");
        let policy = dir.policy(Mode::Confirm);
        // A second server process has its own policy value and the same lock file.
        let other = Policy::new(policy.config().clone());
        let outcome = Arc::new(Mutex::new(None::<String>));
        let slot = Arc::clone(&outcome);
        let mut prompter = FakePrompter::new(None, DialogOutcome::Approved);
        prompter.during_dialog = Some(Box::new(move || {
            let other = other.clone();
            let slot = Arc::clone(&slot);
            // The hook is not async, so it runs the second call on a thread.
            std::thread::spawn(move || {
                let runtime = tokio::runtime::Builder::new_current_thread()
                    .enable_time()
                    .build()
                    .unwrap();
                let result =
                    runtime.block_on(other.begin(facts("scroll"), "other", &FakePrompter::none()));
                *slot.lock().unwrap() = Some(match result {
                    Ok(permit) => {
                        permit.finish(true);
                        "allowed".to_string()
                    }
                    Err(refusal) => refusal.message,
                });
            })
            .join()
            .unwrap();
        }));
        policy
            .begin(click_send(), "c", &prompter)
            .await
            .unwrap()
            .finish(true);
        let message = outcome.lock().unwrap().clone().unwrap();
        assert!(message.contains("input lock"), "{message}");
    }

    #[tokio::test]
    async fn the_lock_is_taken_before_the_caller_reads_facts() {
        let dir = TestDir::new("lockfirst");
        let policy = dir.policy(Mode::Confirm);
        let base = ActionFacts::new("click");
        let entry = policy.enter(&base, "c").await.unwrap();
        // The caller reads window and element facts here, with the lock held.
        assert!(!lock_is_free(&policy.config().lock_path));
        let refusal = policy
            .begin(facts("scroll"), "b", &FakePrompter::none())
            .await
            .unwrap_err();
        assert!(
            refusal.message.contains("input lock"),
            "{}",
            refusal.message
        );
        let permit = entry
            .decide(facts("click"), &FakePrompter::none())
            .await
            .unwrap();
        permit.finish(true);
        assert!(lock_is_free(&policy.config().lock_path));
    }

    #[tokio::test]
    async fn the_lock_stays_until_the_end_line_is_written() {
        let dir = TestDir::new("lockend");
        let policy = dir.policy(Mode::Full);
        let permit = policy
            .begin(facts("scroll"), "c", &FakePrompter::none())
            .await
            .unwrap();
        assert!(!lock_is_free(&policy.config().lock_path));
        permit.finish(true);
        assert!(lock_is_free(&policy.config().lock_path));
    }

    #[tokio::test]
    async fn a_refusal_releases_the_lock() {
        let dir = TestDir::new("lockrelease");
        let policy = dir.policy(Mode::Confirm);
        let prompter = FakePrompter::new(None, DialogOutcome::Denied);
        policy
            .begin(click_send(), "c", &prompter)
            .await
            .unwrap_err();
        assert!(lock_is_free(&policy.config().lock_path));
    }

    #[tokio::test]
    async fn the_start_line_is_written_before_the_action_and_the_end_line_after() {
        let dir = TestDir::new("phases");
        let policy = dir.policy(Mode::Confirm);
        let prompter = FakePrompter::new(None, DialogOutcome::Approved);
        let permit = policy.begin(click_send(), "c", &prompter).await.unwrap();
        // The action runs here. The start line is on disk already.
        let before = audit_lines(&policy);
        assert_eq!(before.len(), 1);
        assert_eq!(before[0]["phase"], "start");
        assert_eq!(before[0]["decision"], "confirmed");
        assert_eq!(before[0]["tier"], "high");
        assert!(before[0].get("ok").is_none(), "{}", before[0]);
        permit.finish(true);
        let after = audit_lines(&policy);
        assert_eq!(after.len(), 2);
        assert_eq!(after[1]["phase"], "end");
        assert_eq!(after[1]["ok"], true);
        assert_eq!(after[1]["decision"], "confirmed");
        assert_eq!(after[0]["call_id"], after[1]["call_id"]);
        assert!(!after[0]["call_id"].as_str().unwrap().is_empty());
    }

    #[tokio::test]
    async fn each_call_has_its_own_call_id() {
        let dir = TestDir::new("callid");
        let policy = dir.policy(Mode::Full);
        for _ in 0..3 {
            policy
                .begin(facts("scroll"), "c", &FakePrompter::none())
                .await
                .unwrap()
                .finish(true);
        }
        let ids: std::collections::BTreeSet<String> = audit_lines(&policy)
            .iter()
            .map(|line| line["call_id"].as_str().unwrap().to_string())
            .collect();
        assert_eq!(ids.len(), 3);
    }

    #[tokio::test]
    async fn a_failed_action_writes_ok_false_on_the_end_line() {
        let dir = TestDir::new("endfalse");
        let policy = dir.policy(Mode::Full);
        policy
            .begin(facts("scroll"), "c", &FakePrompter::none())
            .await
            .unwrap()
            .finish(false);
        let lines = audit_lines(&policy);
        assert_eq!(lines[1]["phase"], "end");
        assert_eq!(lines[1]["ok"], false);
    }

    #[tokio::test]
    async fn every_refusal_writes_exactly_one_refused_line() {
        let dir = TestDir::new("refusedone");
        let cases: Vec<(&str, Mode, ActionFacts, FakePrompter)> = vec![
            (
                "observe",
                Mode::Observe,
                facts("click"),
                FakePrompter::none(),
            ),
            (
                "blocked",
                Mode::Confirm,
                in_window("type_text", "firefox", "Login")
                    .with_element(Some("password text"), None, None)
                    .with_text("pw"),
                FakePrompter::none(),
            ),
            (
                "denied",
                Mode::Confirm,
                click_send(),
                FakePrompter::new(None, DialogOutcome::Denied),
            ),
            (
                "nochannel",
                Mode::Confirm,
                click_send(),
                FakePrompter::none(),
            ),
        ];
        for (label, mode, facts, prompter) in cases {
            let policy = dir.policy(mode);
            let before = audit_lines(&policy).len();
            policy.begin(facts, "c", &prompter).await.unwrap_err();
            let lines = audit_lines(&policy);
            assert_eq!(lines.len(), before + 1, "{label}");
            let line = lines.last().unwrap();
            assert_eq!(line["phase"], "refused", "{label}");
            assert_eq!(line["ok"], false, "{label}");
            assert!(!line["call_id"].as_str().unwrap().is_empty(), "{label}");
        }
        assert!(lock_is_free(&dir.policy(Mode::Full).config().lock_path));
    }

    #[tokio::test]
    async fn a_call_that_cannot_get_the_lock_writes_a_refused_line() {
        let dir = TestDir::new("locktimeout");
        let policy = dir.policy(Mode::Full);
        let holder = policy
            .begin(facts("scroll"), "a", &FakePrompter::none())
            .await
            .unwrap();
        let refusal = policy
            .begin(facts("scroll"), "b", &FakePrompter::none())
            .await
            .unwrap_err();
        assert!(
            refusal.message.contains("input lock"),
            "{}",
            refusal.message
        );
        holder.finish(true);
        let phases: Vec<String> = decisions_by_phase(&policy);
        assert_eq!(phases, ["start:allowed", "refused:refused", "end:allowed"]);
    }

    #[tokio::test]
    async fn a_dropped_entry_writes_a_refused_line_and_releases_the_lock() {
        let dir = TestDir::new("cancelled");
        let policy = dir.policy(Mode::Confirm);
        let entry = policy.enter(&ActionFacts::new("click"), "c").await.unwrap();
        drop(entry);
        assert_eq!(decisions_by_phase(&policy), ["refused:cancelled"]);
        assert!(lock_is_free(&policy.config().lock_path));
    }

    #[tokio::test]
    async fn the_gate_does_not_send_input_when_the_audit_file_cannot_open() {
        let dir = TestDir::new("auditopen");
        let policy = dir.policy(Mode::Full);
        // A directory in place of the log makes every open fail.
        std::fs::create_dir_all(policy.config().audit_path()).unwrap();
        let refusal = policy
            .begin(facts("scroll"), "c", &FakePrompter::none())
            .await
            .unwrap_err();
        assert!(
            refusal.message.contains("audit log cannot be written"),
            "{}",
            refusal.message
        );
        assert!(
            refusal.message.contains("No input was sent"),
            "{}",
            refusal.message
        );
        assert!(lock_is_free(&policy.config().lock_path));
    }

    #[tokio::test]
    async fn the_gate_does_not_send_input_when_the_start_line_cannot_be_written() {
        let dir = TestDir::new("auditfull");
        if !Path::new("/dev/full").exists() {
            return;
        }
        let policy = dir.policy(Mode::Full);
        std::fs::create_dir_all(&policy.config().state_dir).unwrap();
        // Every write to /dev/full fails with "no space left".
        std::os::unix::fs::symlink("/dev/full", policy.config().audit_path()).unwrap();
        let prompter = FakePrompter::none();
        let refusal = policy
            .begin(facts("scroll"), "c", &prompter)
            .await
            .unwrap_err();
        assert!(
            refusal.message.contains("audit log cannot be written"),
            "{}",
            refusal.message
        );
        assert!(
            refusal.message.contains("No input was sent"),
            "{}",
            refusal.message
        );
        assert!(lock_is_free(&policy.config().lock_path));
    }

    #[tokio::test]
    async fn a_confirmed_call_with_a_broken_log_never_runs() {
        let dir = TestDir::new("auditconfirm");
        if !Path::new("/dev/full").exists() {
            return;
        }
        let policy = dir.policy(Mode::Confirm);
        std::fs::create_dir_all(&policy.config().state_dir).unwrap();
        std::os::unix::fs::symlink("/dev/full", policy.config().audit_path()).unwrap();
        let prompter = FakePrompter::new(None, DialogOutcome::Approved);
        let result = policy.begin(click_send(), "c", &prompter).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn long_shell_commands_are_refused_in_confirm_mode_and_allowed_in_full_mode() {
        let dir = TestDir::new("longshell");
        let long = "x".repeat(RUN_SHELL_PREVIEW_MAX_CHARS + 1);
        let confirm = dir.policy(Mode::Confirm);
        let prompter = FakePrompter::new(Some(ElicitOutcome::Approved), DialogOutcome::Approved);
        let refusal = confirm
            .begin(
                ActionFacts::new("run_shell").with_command(&long),
                "c",
                &prompter,
            )
            .await
            .unwrap_err();
        assert!(
            refusal.message.contains("TAKE_MY_COMPUTER_POLICY=full"),
            "{}",
            refusal.message
        );
        assert!(refusal.message.contains("2000"), "{}", refusal.message);
        assert_eq!(prompter.calls(), (0, 0));
        let lines = audit_lines(&confirm);
        assert_eq!(lines[0]["reason"], "command_too_long");
        let raw = std::fs::read_to_string(confirm.config().audit_path()).unwrap();
        assert!(!raw.contains(&long), "the audit log holds the command");

        let full = dir.policy(Mode::Full);
        full.begin(
            ActionFacts::new("run_shell").with_command(&long),
            "c",
            &FakePrompter::none(),
        )
        .await
        .unwrap()
        .finish(true);
    }

    #[tokio::test]
    async fn a_short_shell_command_shows_in_full_in_the_confirmation_message() {
        let dir = TestDir::new("shellprompt");
        let policy = dir.policy(Mode::Confirm);
        let command = format!("echo start && {} && echo end", "true && ".repeat(100));
        let prompter = FakePrompter::new(Some(ElicitOutcome::Approved), DialogOutcome::Denied);
        policy
            .begin(
                ActionFacts::new("run_shell").with_command(&command),
                "c",
                &prompter,
            )
            .await
            .unwrap()
            .finish(true);
        let shown = prompter.last_text.lock().unwrap().clone();
        assert!(shown.contains(&command), "{shown}");
        let raw = std::fs::read_to_string(policy.config().audit_path()).unwrap();
        assert!(
            !raw.contains("echo start"),
            "the audit log holds the command"
        );
    }

    #[tokio::test]
    async fn a_kill_switch_after_the_lock_wait_still_blocks_the_action() {
        let dir = TestDir::new("killafterlock");
        let policy = dir.policy(Mode::Full);
        let entry = policy
            .enter(&ActionFacts::new("scroll"), "c")
            .await
            .unwrap();
        // The operator presses the kill switch while the caller reads facts.
        engage_kill_switch(policy.config()).unwrap();
        let refusal = entry
            .decide(facts("scroll"), &FakePrompter::none())
            .await
            .unwrap_err();
        assert!(
            refusal.message.contains("kill switch"),
            "{}",
            refusal.message
        );
        assert_eq!(decisions_by_phase(&policy), ["refused:refused"]);
    }

    #[test]
    fn status_reports_mode_switch_apps_and_paths() {
        let dir = TestDir::new("status");
        let mut config = dir.policy(Mode::Confirm).config().clone();
        config.warning = Some("careful".to_string());
        let policy = Policy::new(config);
        let status = policy.status(true);
        assert_eq!(status.mode, "confirm");
        assert!(!status.kill_switch_active);
        assert!(
            status.kill_switch_file.ends_with("take-my-computer/STOP")
                || status.kill_switch_file.ends_with("state/STOP")
        );
        assert!(status.audit_log.ends_with("audit.jsonl"));
        assert_eq!(status.send_on_enter_apps.len(), 10);
        assert_eq!(status.terminal_apps.len(), 14);
        assert_eq!(status.confirmation_channels.first().unwrap(), "elicitation");
        assert_eq!(status.warning.as_deref(), Some("careful"));
        let json = serde_json::to_value(&status).unwrap();
        assert_eq!(json["mode"], "confirm");
        assert_eq!(json["kill_switch_active"], false);
    }
}
