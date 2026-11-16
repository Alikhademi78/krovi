//! Gemini Live API (BidiGenerateContent over WebSocket) protocol and client.
//!
//! Pure message builders and parsers are always compiled and unit-tested.
//! The WebSocket transport is gated behind the `net` feature (`tungstenite` + pure-Rust `rustls`)
//! and connects directly from this PC to `generativelanguage.googleapis.com` with the user's
//! stored Gemini API key (never forwarded to the webview or logged).

use base64::engine::general_purpose::STANDARD;
use base64::Engine as _;
use serde::Serialize;
use serde_json::{json, Value};

pub const DEFAULT_LIVE_MODEL: &str = "gemini-3.8-live";
pub const DEFAULT_LIVE_VOICE: &str = "Puck";
pub const INPUT_SAMPLE_RATE: u32 = 16_000;
pub const DEFAULT_OUTPUT_SAMPLE_RATE: u32 = 24_000;

pub const LIVE_WS_ENDPOINT: &str =
    "wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent";

pub const LIVE_VOICES: &[&str] = &["Puck", "Charon", "Kore", "Fenrir", "Aoede"];

#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct LiveModelOption {
    pub id: &'static str,
    pub label: &'static str,
}

pub const LIVE_MODELS: &[LiveModelOption] = &[
    LiveModelOption {
        id: "gemini-3.8-live",
        label: "Gemini 3.8 Live (recommended)",
    },
    LiveModelOption {
        id: "gemini-3.8-live-extended-thinking",
        label: "Gemini 3.8 Live Extended Thinking",
    },
    LiveModelOption {
        id: "gemini-3.1-flash-live-preview",
        label: "Gemini 3.1 Flash Live (Preview)",
    },
    LiveModelOption {
        id: "gemini-2.5-flash-native-audio-latest",
        label: "Gemini 2.5 Flash Native Audio",
    },
];

pub const DEFAULT_SYSTEM_INSTRUCTION: &str = "You are Krovi, a helpful, concise live voice assistant on the user's Windows PC. Reply naturally, clearly, and concisely in the same language the user speaks (for example Persian if they speak Persian, English if they speak English).";

/// Strip any `models/` prefix and fall back to [`DEFAULT_LIVE_MODEL`] when blank.
pub fn normalize_live_model(raw: &str) -> String {
    let trimmed = raw.trim();
    let stripped = trimmed.strip_prefix("models/").unwrap_or(trimmed).trim();
    if stripped.is_empty()
        || stripped
            .chars()
            .any(|c| c.is_control() || c.is_whitespace())
    {
        DEFAULT_LIVE_MODEL.to_string()
    } else {
        stripped.to_string()
    }
}

/// Format a model id as `models/<id>` for `BidiGenerateContentSetup`.
pub fn live_model_resource(raw: &str) -> String {
    format!("models/{}", normalize_live_model(raw))
}

/// Match one of the supported prebuilt Gemini Live voices, falling back to `Puck`.
pub fn normalize_live_voice(raw: &str) -> &'static str {
    let trimmed = raw.trim();
    LIVE_VOICES
        .iter()
        .copied()
        .find(|v| v.eq_ignore_ascii_case(trimmed))
        .unwrap_or(DEFAULT_LIVE_VOICE)
}

/// Build the `v1beta` BidiGenerateContent WebSocket URL. Never log the returned string.
pub fn live_ws_url(api_key: &str) -> Result<String, String> {
    let key = api_key.trim();
    if key.is_empty() {
        return Err("Missing Gemini API key.".into());
    }
    if key.chars().any(|c| c.is_whitespace() || c.is_control()) {
        return Err("Invalid Gemini API key format.".into());
    }
    Ok(format!("{LIVE_WS_ENDPOINT}?key={key}"))
}

/// Scrub `?key=...` (or `key=...`) from any transport error message before showing or logging it.
pub fn sanitize_ws_error(raw: &str) -> String {
    if let Some(idx) = raw.find("key=") {
        let head = &raw[..idx];
        let rest = &raw[idx + 4..];
        let tail_start = rest
            .find(|c: char| c.is_whitespace() || matches!(c, '&' | '"' | '\'' | ')'))
            .unwrap_or(rest.len());
        format!("{head}key=[redacted]{}", &rest[tail_start..])
    } else {
        raw.to_string()
    }
}

/// Build the initial `BidiGenerateContentSetup` JSON message sent right after WS connect.
pub fn build_setup_json(model: &str, voice: &str, system_instruction: Option<&str>) -> Value {
    let model_res = live_model_resource(model);
    let voice_name = normalize_live_voice(voice);
    let instruction = system_instruction
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .unwrap_or(DEFAULT_SYSTEM_INSTRUCTION);

    json!({
        "setup": {
            "model": model_res,
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "speechConfig": {
                    "voiceConfig": {
                        "prebuiltVoiceConfig": {
                            "voiceName": voice_name
                        }
                    }
                }
            },
            "systemInstruction": {
                "parts": [{ "text": instruction }]
            },
            "inputAudioTranscription": {},
            "outputAudioTranscription": {}
        }
    })
}

/// Convert 16-bit PCM mono samples into little-endian bytes ready for `audio/pcm;rate=16000`.
pub fn pcm16_samples_to_le_bytes(samples: &[i16]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(samples.len() * 2);
    for &s in samples {
        bytes.extend_from_slice(&s.to_le_bytes());
    }
    bytes
}

/// Build a `realtimeInput` audio chunk payload for 16 kHz mono 16-bit little-endian PCM.
pub fn build_audio_chunk_json(pcm16_le_bytes: &[u8]) -> String {
    let b64 = STANDARD.encode(pcm16_le_bytes);
    json!({
        "realtimeInput": {
            "audio": {
                "mimeType": "audio/pcm;rate=16000",
                "data": b64
            }
        }
    })
    .to_string()
}

/// Signal end-of-audio-stream (e.g. when releasing a push-to-talk key).
pub fn build_audio_end_json() -> &'static str {
    r#"{"realtimeInput":{"audioStreamEnd":true}}"#
}

/// Send a typed user turn into an active Gemini Live session.
pub fn build_client_text_json(text: &str) -> String {
    json!({
        "clientContent": {
            "turns": [{
                "role": "user",
                "parts": [{ "text": text }]
            }],
            "turnComplete": true
        }
    })
    .to_string()
}

/// Parse sample rate from MIME types like `audio/pcm;rate=24000`. Defaults to 24000 Hz.
pub fn parse_pcm_sample_rate(mime: &str) -> u32 {
    for part in mime.split(';').skip(1) {
        let part = part.trim();
        if let Some(rate_str) = part.strip_prefix("rate=") {
            if let Ok(rate) = rate_str.trim().parse::<u32>() {
                if (8_000..=96_000).contains(&rate) {
                    return rate;
                }
            }
        }
    }
    DEFAULT_OUTPUT_SAMPLE_RATE
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LiveServerEvent {
    SetupComplete,
    InputTranscript { text: String },
    OutputTranscript { text: String },
    OutputText { text: String },
    AudioChunk { base64: String, sample_rate: u32 },
    Interrupted,
    TurnComplete,
    GoAway { time_left: Option<String> },
    Error { message: String },
}

/// Parse one JSON frame from Gemini Live into zero or more semantic events.
pub fn parse_server_message(raw: &str) -> Result<Vec<LiveServerEvent>, String> {
    let v: Value = serde_json::from_str(raw).map_err(|e| format!("invalid live JSON: {e}"))?;
    let mut events = Vec::new();

    if let Some(err) = v.get("error") {
        let msg = err
            .get("message")
            .and_then(Value::as_str)
            .unwrap_or("Gemini Live returned an error.")
            .to_string();
        events.push(LiveServerEvent::Error { message: msg });
        return Ok(events);
    }

    if v.get("setupComplete").is_some() {
        events.push(LiveServerEvent::SetupComplete);
    }

    if let Some(go_away) = v.get("goAway") {
        let time_left = go_away
            .get("timeLeft")
            .and_then(Value::as_str)
            .map(String::from);
        events.push(LiveServerEvent::GoAway { time_left });
    }

    if let Some(sc) = v.get("serverContent") {
        if let Some(t) = sc
            .pointer("/inputTranscription/text")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
        {
            events.push(LiveServerEvent::InputTranscript {
                text: t.to_string(),
            });
        }

        if let Some(parts) = sc.pointer("/modelTurn/parts").and_then(Value::as_array) {
            for part in parts {
                if let Some(inline) = part.get("inlineData") {
                    let mime = inline.get("mimeType").and_then(Value::as_str).unwrap_or("");
                    if mime.starts_with("audio/pcm") {
                        if let Some(data) = inline
                            .get("data")
                            .and_then(Value::as_str)
                            .filter(|s| !s.is_empty())
                        {
                            events.push(LiveServerEvent::AudioChunk {
                                base64: data.to_string(),
                                sample_rate: parse_pcm_sample_rate(mime),
                            });
                        }
                    }
                }
                let is_thought = part
                    .get("thought")
                    .and_then(Value::as_bool)
                    .unwrap_or(false);
                if !is_thought {
                    if let Some(txt) = part
                        .get("text")
                        .and_then(Value::as_str)
                        .filter(|s| !s.is_empty())
                    {
                        events.push(LiveServerEvent::OutputText {
                            text: txt.to_string(),
                        });
                    }
                }
            }
        }

        if let Some(t) = sc
            .pointer("/outputTranscription/text")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
        {
            events.push(LiveServerEvent::OutputTranscript {
                text: t.to_string(),
            });
        }

        if sc
            .get("interrupted")
            .and_then(Value::as_bool)
            .unwrap_or(false)
        {
            events.push(LiveServerEvent::Interrupted);
        }

        if sc
            .get("turnComplete")
            .and_then(Value::as_bool)
            .unwrap_or(false)
        {
            events.push(LiveServerEvent::TurnComplete);
        }
    }

    Ok(events)
}

#[cfg(feature = "net")]
pub mod ws {
    use std::io::ErrorKind;
    use std::net::TcpStream;
    use std::time::Duration;

    use serde_json::Value;
    use tungstenite::stream::MaybeTlsStream;
    use tungstenite::{connect, Error as WsError, Message, WebSocket};

    use super::{live_ws_url, parse_server_message, sanitize_ws_error, LiveServerEvent};

    #[derive(Debug)]
    pub enum LivePollError {
        Closed,
        Fatal(String),
    }

    pub struct LiveSocket {
        ws: WebSocket<MaybeTlsStream<TcpStream>>,
    }

    impl LiveSocket {
        pub fn connect(api_key: &str, setup_json: &Value) -> Result<Self, String> {
            let url = live_ws_url(api_key)?;
            let (mut ws, _resp) = connect(url.as_str()).map_err(|e| {
                let clean = sanitize_ws_error(&e.to_string());
                format!("Could not connect to Gemini Live: {clean}")
            })?;

            match ws.get_mut() {
                MaybeTlsStream::Plain(s) => {
                    let _ = s.set_read_timeout(Some(Duration::from_millis(20)));
                }
                MaybeTlsStream::Rustls(s) => {
                    let _ = s
                        .get_ref()
                        .set_read_timeout(Some(Duration::from_millis(20)));
                }
                _ => {}
            }

            let setup_text = setup_json.to_string();
            ws.send(Message::Text(setup_text.into())).map_err(|e| {
                format!(
                    "Failed to initialize Gemini Live session: {}",
                    sanitize_ws_error(&e.to_string())
                )
            })?;

            Ok(Self { ws })
        }

        pub fn send_text(&mut self, text: &str) -> Result<(), String> {
            self.ws.send(Message::Text(text.into())).map_err(|e| {
                format!(
                    "Gemini Live send failed: {}",
                    sanitize_ws_error(&e.to_string())
                )
            })
        }

        pub fn poll_events(&mut self) -> Result<Vec<LiveServerEvent>, LivePollError> {
            let mut out = Vec::new();
            for _ in 0..24 {
                match self.ws.read() {
                    Ok(Message::Text(txt)) => {
                        if let Ok(mut evs) = parse_server_message(txt.as_str()) {
                            out.append(&mut evs);
                        }
                    }
                    Ok(Message::Binary(bin)) => {
                        if let Ok(txt) = std::str::from_utf8(&bin) {
                            if let Ok(mut evs) = parse_server_message(txt) {
                                out.append(&mut evs);
                            }
                        }
                    }
                    Ok(Message::Ping(payload)) => {
                        let _ = self.ws.send(Message::Pong(payload));
                    }
                    Ok(Message::Pong(_)) | Ok(Message::Frame(_)) => {}
                    Ok(Message::Close(_)) => return Err(LivePollError::Closed),
                    Err(WsError::Io(e))
                        if matches!(e.kind(), ErrorKind::WouldBlock | ErrorKind::TimedOut) =>
                    {
                        break;
                    }
                    Err(WsError::ConnectionClosed) | Err(WsError::AlreadyClosed) => {
                        return Err(LivePollError::Closed);
                    }
                    Err(e) => {
                        return Err(LivePollError::Fatal(sanitize_ws_error(&e.to_string())));
                    }
                }
            }
            Ok(out)
        }

        pub fn close(&mut self) {
            let _ = self.ws.close(None);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_and_voice_normalisation_defaults_cleanly() {
        assert_eq!(normalize_live_model(""), "gemini-3.8-live");
        assert_eq!(
            normalize_live_model("models/gemini-3.8-live"),
            "gemini-3.8-live"
        );
        assert_eq!(
            live_model_resource("gemini-3.8-live"),
            "models/gemini-3.8-live"
        );
        assert_eq!(normalize_live_voice("aoede"), "Aoede");
        assert_eq!(normalize_live_voice("unknown"), "Puck");
    }

    #[test]
    fn ws_url_validation_and_error_redaction() {
        assert!(live_ws_url("").is_err());
        assert!(live_ws_url("bad key").is_err());
        let url = live_ws_url("AIzaTest1234").unwrap();
        assert!(url.starts_with(LIVE_WS_ENDPOINT));
        assert!(url.ends_with("?key=AIzaTest1234"));

        let err = format!("failed connecting to {url}: TLS handshake error");
        let clean = sanitize_ws_error(&err);
        assert!(!clean.contains("AIzaTest1234"));
        assert!(clean.contains("key=[redacted]"));
    }

    #[test]
    fn setup_and_audio_payloads_match_gemini_live_schema() {
        let setup = build_setup_json("gemini-3.8-live", "Kore", Some("Speak Persian"));
        assert_eq!(setup["setup"]["model"], "models/gemini-3.8-live");
        assert_eq!(
            setup["setup"]["generationConfig"]["responseModalities"],
            json!(["AUDIO"])
        );
        assert_eq!(
            setup["setup"]["generationConfig"]["speechConfig"]["voiceConfig"]
                ["prebuiltVoiceConfig"]["voiceName"],
            "Kore"
        );
        assert_eq!(
            setup["setup"]["systemInstruction"]["parts"][0]["text"],
            "Speak Persian"
        );
        assert!(setup["setup"]["inputAudioTranscription"].is_object());
        assert!(setup["setup"]["outputAudioTranscription"].is_object());

        let le = pcm16_samples_to_le_bytes(&[0x0102, -1]);
        assert_eq!(le, vec![0x02, 0x01, 0xFF, 0xFF]);
        let chunk: Value = serde_json::from_str(&build_audio_chunk_json(&le)).unwrap();
        assert_eq!(
            chunk["realtimeInput"]["audio"]["mimeType"],
            "audio/pcm;rate=16000"
        );
        assert_eq!(chunk["realtimeInput"]["audio"]["data"], "AgH//w==");
        assert_eq!(
            build_audio_end_json(),
            r#"{"realtimeInput":{"audioStreamEnd":true}}"#
        );
    }

    #[test]
    fn parses_server_messages_across_events() {
        assert_eq!(
            parse_server_message(r#"{"setupComplete":{}}"#).unwrap(),
            vec![LiveServerEvent::SetupComplete]
        );

        let audio_msg = r#"{
            "serverContent": {
                "inputTranscription": { "text": "Salam " },
                "modelTurn": {
                    "parts": [
                        { "inlineData": { "mimeType": "audio/pcm;rate=24000", "data": "AAAA" } },
                        { "text": "internal", "thought": true }
                    ]
                },
                "outputTranscription": { "text": "Salam! Chetor mitoonam komak konam?" },
                "turnComplete": true
            }
        }"#;
        let evs = parse_server_message(audio_msg).unwrap();
        assert_eq!(
            evs,
            vec![
                LiveServerEvent::InputTranscript {
                    text: "Salam ".into()
                },
                LiveServerEvent::AudioChunk {
                    base64: "AAAA".into(),
                    sample_rate: 24_000
                },
                LiveServerEvent::OutputTranscript {
                    text: "Salam! Chetor mitoonam komak konam?".into()
                },
                LiveServerEvent::TurnComplete,
            ]
        );

        assert_eq!(
            parse_server_message(r#"{"serverContent":{"interrupted":true}}"#).unwrap(),
            vec![LiveServerEvent::Interrupted]
        );
        assert_eq!(
            parse_server_message(r#"{"error":{"message":"Quota exceeded"}}"#).unwrap(),
            vec![LiveServerEvent::Error {
                message: "Quota exceeded".into()
            }]
        );
    }
}
