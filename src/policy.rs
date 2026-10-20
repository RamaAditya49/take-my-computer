//! Safety gate for mutating tools (take-my-computer).
//!
//! MCP annotations are hints only, and many hosts run with approvals off, so
//! the server enforces the gate itself. Each mutating tool calls
//! [`Policy::begin`] before it changes focus, shows an indicator event, or
//! sends input. The gate does five things:
//!
//! 1. It refuses every call when the kill switch file exists.
//! 2. It applies the policy mode (`confirm`, `full`, or `observe`).
//! 3. It asks the user to confirm high-impact actions.
//! 4. It holds an exclusive `flock` while the call runs.
//! 5. It appends one audit line per call. The line never holds typed text.

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
    sync::Arc,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

pub(crate) const POLICY_ENV: &str = "TAKE_MY_COMPUTER_POLICY";
pub(crate) const SEND_APPS_ENV: &str = "TAKE_MY_COMPUTER_SEND_APPS";

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
const PROMPT_COMMAND_MAX_CHARS: usize = 120;

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

#[derive(Debug, Clone)]
pub(crate) struct PolicyConfig {
    pub mode: Mode,
    pub warning: Option<String>,
    pub state_dir: PathBuf,
    pub lock_path: PathBuf,
    pub send_apps: Vec<String>,
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
            send_apps: parse_send_apps(get(SEND_APPS_ENV).as_deref()),
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
        .unwrap_or_else(|| {
            // SAFETY: getuid has no preconditions and cannot fail.
            let uid = unsafe { libc::getuid() };
            env::temp_dir().join(format!("take-my-computer-state-{uid}"))
        });
    base.join("take-my-computer")
}

/// An empty or blank list means "use the default list". That is the safe
/// reading: an empty override must not silently disable the check.
fn parse_send_apps(value: Option<&str>) -> Vec<String> {
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
        DEFAULT_SEND_APPS
            .iter()
            .map(|app| (*app).to_string())
            .collect()
    } else {
        parsed
    }
}

fn ensure_dir(path: &Path) -> std::io::Result<()> {
    DirBuilder::new().recursive(true).mode(0o700).create(path)
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
    EnterInUnknownWindow,
    NewlineInSendApp,
    NewlineInUnknownWindow,
    PasswordField,
    UnresolvedElement,
    SetupTool,
    ShellCommand,
}

impl Reason {
    /// Stable code for the audit log.
    pub(crate) fn code(&self) -> String {
        match self {
            Reason::CommitWord(word) => format!("commit_word:{word}"),
            Reason::EnterInSendApp => "enter_in_send_app".to_string(),
            Reason::EnterInUnknownWindow => "enter_in_unknown_window".to_string(),
            Reason::NewlineInSendApp => "newline_in_send_app".to_string(),
            Reason::NewlineInUnknownWindow => "newline_in_unknown_window".to_string(),
            Reason::PasswordField => "password_field".to_string(),
            Reason::UnresolvedElement => "unresolved_element".to_string(),
            Reason::SetupTool => "setup_tool".to_string(),
            Reason::ShellCommand => "shell_command".to_string(),
        }
    }

    /// Plain sentence for the confirmation dialog and for refusal messages.
    pub(crate) fn describe(&self) -> String {
        match self {
            Reason::CommitWord(word) => {
                format!("The control name or description contains the word \"{word}\".")
            }
            Reason::EnterInSendApp => "Enter can send a message in this app.".to_string(),
            Reason::EnterInUnknownWindow => {
                "The app that receives Enter is unknown. Enter can send a message.".to_string()
            }
            Reason::NewlineInSendApp => {
                "The text has a line break. A line break can send a message in this app."
                    .to_string()
            }
            Reason::NewlineInUnknownWindow => {
                "The text has a line break and the receiving app is unknown. A line break can send a message."
                    .to_string()
            }
            Reason::PasswordField => "The target is a password field.".to_string(),
            Reason::UnresolvedElement => {
                "The target element is not in the last accessibility snapshot.".to_string()
            }
            Reason::SetupTool => "This tool changes the desktop configuration.".to_string(),
            Reason::ShellCommand => "This tool runs a shell command.".to_string(),
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
    /// A short preview is for the confirmation dialog only.
    pub(crate) fn with_command(mut self, command: &str) -> Self {
        self.text_chars = Some(command.chars().count());
        self.text_sha256 = Some(sha256_hex(command));
        self.command_preview = Some(display_text(command, PROMPT_COMMAND_MAX_CHARS));
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

/// True for Return, KP_Enter, or Enter with any modifiers. This follows the
/// key grammar of `press_key` (parts joined with `+`, case-insensitive,
/// hyphens and spaces ignored) and also accepts `KP_Enter` spellings.
pub(crate) fn is_enter_key(key: &str) -> bool {
    key.split('+')
        .map(str::trim)
        .rfind(|part| !part.is_empty())
        .is_some_and(|last| {
            let normalized = last.to_ascii_lowercase().replace(['-', '_', ' '], "");
            matches!(
                normalized.as_str(),
                "enter" | "return" | "kpenter" | "numpadenter"
            )
        })
}

/// The send-on-Enter entry that matches the window, if any. Entries are
/// lowercase substrings of the app id, the wm class, or the title.
pub(crate) fn send_app_match<'a>(facts: &ActionFacts, send_apps: &'a [String]) -> Option<&'a str> {
    let fields = [
        facts.app_id.as_deref(),
        facts.wm_class.as_deref(),
        facts.window_title.as_deref(),
    ];
    send_apps
        .iter()
        .find(|entry| {
            fields
                .iter()
                .flatten()
                .any(|field| field.to_lowercase().contains(entry.as_str()))
        })
        .map(String::as_str)
}

/// Rate one call. Pure: it reads only `facts` and `send_apps`.
pub(crate) fn classify(facts: &ActionFacts, send_apps: &[String]) -> Classification {
    match facts.tool {
        "setup_accessibility" | "setup_window_targeting" => Classification::high(Reason::SetupTool),
        "run_shell" => Classification::high(Reason::ShellCommand),
        "click" | "perform_action" => {
            let word = [
                facts.element_name.as_deref(),
                facts.element_description.as_deref(),
            ]
            .into_iter()
            .flatten()
            .find_map(commit_word);
            if let Some(word) = word {
                return Classification::high(Reason::CommitWord(word));
            }
            // A click that names a missing element fails inside the tool
            // without input, so only perform_action needs the unresolved rule.
            if facts.tool == "perform_action" && facts.element_unresolved {
                return Classification::high(Reason::UnresolvedElement);
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
                if send_app_match(facts, send_apps).is_some() {
                    return Classification::high(Reason::NewlineInSendApp);
                }
                if !facts.has_window() {
                    return Classification::high(Reason::NewlineInUnknownWindow);
                }
            }
            Classification::low()
        }
        "press_key" => {
            if facts.key.as_deref().is_some_and(is_enter_key) {
                if send_app_match(facts, send_apps).is_some() {
                    return Classification::high(Reason::EnterInSendApp);
                }
                if !facts.has_window() {
                    return Classification::high(Reason::EnterInUnknownWindow);
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
#[derive(Debug, Clone, Serialize)]
pub(crate) struct AuditRecord {
    pub ts: String,
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
    pub ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

impl AuditRecord {
    fn new(client: &str, facts: &ActionFacts, mode: Mode, classification: &Classification) -> Self {
        Self {
            ts: rfc3339_utc(SystemTime::now()),
            client: truncate_chars(client, AUDIT_FIELD_MAX_CHARS),
            tool: facts.tool.to_string(),
            mode: mode.as_str(),
            tier: classification.tier.as_str(),
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
            ok: false,
            reason: classification.reason.as_ref().map(Reason::code),
        }
    }

    fn line(&self) -> String {
        let mut line = serde_json::to_string(self).unwrap_or_else(|_| "{}".to_string());
        line.push('\n');
        line
    }
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
        .arg("--question")
        .arg("--no-markup")
        .arg("--title=take-my-computer")
        .arg("--ok-label=Allow")
        .arg("--cancel-label=Deny")
        .arg("--width=440")
        .arg(format!("--timeout={}", timeout.as_secs().max(1)))
        .arg(format!("--text={text}"))
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
    if let Some(preview) = facts.command_preview.as_deref() {
        lines.push(format!("Command: {preview}"));
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
/// [`Permit::finish`] with the result to write the audit line. A permit that
/// drops without `finish` (for example a cancelled call) writes `ok: false`.
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
        if let Some((mut file, mut record)) = self.audit.take() {
            record.ok = ok;
            if let Err(error) = append_audit(&mut file, &record) {
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
            send_on_enter_apps: self.config.send_apps.clone(),
            audit_log: self.config.audit_path().display().to_string(),
            confirmation_channels: channels,
            warning: self.config.warning.clone(),
        }
    }

    /// Decide one call. On success the caller holds the returned [`Permit`]
    /// for the whole call. On refusal no input may be sent.
    pub(crate) async fn begin(
        &self,
        facts: ActionFacts,
        client: &str,
        prompter: &dyn Prompter,
    ) -> Result<Permit, Refusal> {
        let config = &self.config;
        let classification = classify(&facts, &config.send_apps);
        let mut record = AuditRecord::new(client, &facts, config.mode, &classification);

        if config.kill_switch_active() {
            return Err(self.refuse(record, "refused", self.kill_switch_message()));
        }

        let mut decision = "allowed";
        match verdict(config.mode, classification.tier) {
            Verdict::Allow => {}
            Verdict::Refuse(RefuseCause::Observe) => {
                let message = format!(
                    "Refused by the safety policy: the mode is observe, so {} does not run. Read-only tools still work. Mode confirm or full (TAKE_MY_COMPUTER_POLICY) allows this tool.",
                    facts.tool
                );
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
                match self
                    .confirm(&facts, &classification, client, prompter)
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

        let lock = match acquire_input_lock(&config.lock_path, config.lock_timeout).await {
            Ok(lock) => lock,
            Err(message) => return Err(self.refuse(record, "refused", message)),
        };
        // The user may press the kill switch while a dialog or the lock waits.
        if config.kill_switch_active() {
            return Err(self.refuse(record, "refused", self.kill_switch_message()));
        }
        let file = match open_audit(config) {
            Ok(file) => file,
            Err(error) => {
                let message = format!(
                    "Refused: the audit log cannot be written ({}): {error}. No input was sent.",
                    config.audit_path().display()
                );
                return Err(self.refuse(record, "refused", message));
            }
        };
        record.decision = decision;
        Ok(Permit {
            _lock: lock,
            audit: Some((file, record)),
        })
    }

    fn kill_switch_message(&self) -> String {
        format!(
            "Refused: the kill switch is on, so no input is sent. The operator turns it off with 'computer-use-linux resume' (this removes {}).",
            self.config.stop_file().display()
        )
    }

    /// Write the audit line for a refused call and build the refusal.
    fn refuse(&self, mut record: AuditRecord, decision: &'static str, message: String) -> Refusal {
        record.decision = decision;
        record.ok = false;
        match open_audit(&self.config) {
            Ok(mut file) => {
                if let Err(error) = append_audit(&mut file, &record) {
                    eprintln!("[take-my-computer] audit log write failed: {error}");
                }
            }
            Err(error) => eprintln!("[take-my-computer] audit log open failed: {error}"),
        }
        Refusal { message }
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
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;

    fn apps() -> Vec<String> {
        parse_send_apps(None)
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
        let custom = parse_send_apps(Some(" Mattermost , ,ZULIP "));
        assert_eq!(custom, vec!["mattermost".to_string(), "zulip".to_string()]);
        let in_custom = in_window("press_key", "Mattermost", "Town Square").with_key("Enter");
        assert_eq!(classify(&in_custom, &custom).tier, Tier::High);
        let in_old_default = in_window("press_key", "slack", "Slack").with_key("Enter");
        assert_eq!(classify(&in_old_default, &custom).tier, Tier::Low);
    }

    #[test]
    fn empty_send_app_override_keeps_the_default_list() {
        assert_eq!(parse_send_apps(Some("")), apps());
        assert_eq!(parse_send_apps(Some(" , ")), apps());
        assert_eq!(apps().len(), DEFAULT_SEND_APPS.len());
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
        assert!(prompt_text("a", &facts, &cls).contains("Command: ls -la /tmp"));
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
        assert_eq!(lines.len(), 1);
        let line = &lines[0];
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
        let line = &audit_lines(&policy)[0];
        let mut keys: Vec<&str> = line
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect();
        keys.sort_unstable();
        assert_eq!(
            keys,
            [
                "app",
                "client",
                "decision",
                "element_name",
                "key",
                "mode",
                "ok",
                "text_len",
                "text_sha256",
                "tier",
                "tool",
                "ts",
                "window_title",
            ]
        );
        assert_eq!(line["key"], "Ctrl+<char>");
        assert!(line["text_len"].is_null());
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
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0]["ok"], false);
        assert_eq!(lines[0]["decision"], "allowed");
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
        assert_eq!(lines[0]["decision"], "refused");
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

        let decisions: Vec<String> = audit_lines(&policy)
            .iter()
            .map(|line| line["decision"].as_str().unwrap().to_string())
            .collect();
        assert_eq!(decisions, ["confirmed", "denied", "denied"]);
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

        let decisions: Vec<String> = audit_lines(&policy)
            .iter()
            .map(|line| line["decision"].as_str().unwrap().to_string())
            .collect();
        assert_eq!(decisions, ["confirmed", "denied", "denied"]);
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
        assert_eq!(status.confirmation_channels.first().unwrap(), "elicitation");
        assert_eq!(status.warning.as_deref(), Some("careful"));
        let json = serde_json::to_value(&status).unwrap();
        assert_eq!(json["mode"], "confirm");
        assert_eq!(json["kill_switch_active"], false);
    }
}
