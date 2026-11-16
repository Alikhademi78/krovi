//! Gemini Live voice conversation controller for Agent Mode (v0.3+).
//!
//! Connects directly from Rust to Google's `BidiGenerateContent` WebSocket using the Gemini API
//! key stored in Windows Credential Manager (`Krovi/ai/gemini`). The key never reaches the
//! webview. Microphone audio is captured in 16 kHz mono 16-bit PCM via `krovi_voice::capture`
//! and streamed in real time; 24 kHz PCM16 responses and live transcripts are emitted to the UI.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{self, Sender};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::thread;
use std::time::{Duration, Instant};

use serde::Serialize;
use serde_json::json;
use tauri::{AppHandle, Emitter, Manager};
use tauri_plugin_global_shortcut::ShortcutState;

use krovi_net::ai::Provider;
use krovi_net::gemini_live::{
    normalize_live_model, normalize_live_voice, pcm16_samples_to_le_bytes, LiveModelOption,
    LIVE_MODELS, LIVE_VOICES,
};
use krovi_voice::capture::Recorder;

use crate::ai::keys;
use crate::commands::CmdResult;
use crate::dictation::{agent_live_config, Dictation, TalkMode};

const MIC_POLL: Duration = Duration::from_millis(40);
const LEVEL_EMIT_EVERY: Duration = Duration::from_millis(100);
const IDLE_TIMEOUT: Duration = Duration::from_secs(180);
const THINKING_TIMEOUT: Duration = Duration::from_secs(10);
const GEN_COMPLETE_GRACE: Duration = Duration::from_millis(600);
const HOLD_RELEASE_TAIL: Duration = Duration::from_millis(320);
const MIN_HOLD_SAMPLES: usize = 3_200; // 200 ms at 16 kHz
const SPEAKING_SAFETY_GRACE: Duration = Duration::from_millis(1_800);
const SPEAKING_MIN_CHUNK_QUIET: Duration = Duration::from_millis(250);
const MAX_PENDING_AUDIO_CHUNKS: usize = 120;

#[derive(Debug)]
enum LiveCmd {
    AudioPcm(Vec<u8>),
    AudioEnd,
    CancelTurn,
    OutputPlaybackChanged(bool),
    ClientText(String),
    Stop,
}

#[derive(Clone)]
struct SessionHandle {
    gen_id: u64,
    tx: Sender<LiveCmd>,
    mic_open: Arc<AtomicBool>,
    output_playing: Arc<AtomicBool>,
    mic_gen: Arc<AtomicU64>,
    stop_mic: Arc<Mutex<Option<Arc<AtomicBool>>>>,
    tail_abort: Arc<Mutex<Option<Arc<AtomicBool>>>>,
    model: String,
    voice: String,
    phase: Arc<Mutex<String>>,
}

pub struct GeminiLive {
    active: Mutex<Option<SessionHandle>>,
    next_gen: AtomicU64,
}

impl Default for GeminiLive {
    fn default() -> Self {
        Self::new()
    }
}

impl GeminiLive {
    pub fn new() -> Self {
        Self {
            active: Mutex::new(None),
            next_gen: AtomicU64::new(1),
        }
    }
}

#[derive(Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct GeminiLiveStatus {
    pub state: String,
    pub session_active: bool,
    pub model: String,
    pub voice: String,
    pub ready: bool,
    pub readiness_error: Option<String>,
}

#[derive(Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct GeminiLiveCatalog {
    pub models: &'static [LiveModelOption],
    pub voices: &'static [&'static str],
}

fn lk<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

fn emit_state(
    app: &AppHandle,
    state: &str,
    level: Option<f32>,
    session_active: bool,
    model: &str,
    voice: &str,
) {
    if let Some(h) = lk(&app.state::<GeminiLive>().active).as_ref() {
        *lk(&h.phase) = state.to_string();
    }
    let _ = app.emit(
        "gemini-live-state",
        json!({
            "state": state,
            "level": level,
            "sessionActive": session_active,
            "model": model,
            "voice": voice,
        }),
    );
}

fn emit_error(app: &AppHandle, kind: &str, message: impl Into<String>) {
    let _ = app.emit(
        "gemini-live-error",
        json!({
            "kind": kind,
            "message": message.into(),
        }),
    );
}

fn stop_active_mic(handle: &SessionHandle) {
    if let Some(flag) = lk(&handle.stop_mic).take() {
        flag.store(true, Ordering::SeqCst);
    }
}

fn abort_hold_tail(handle: &SessionHandle) {
    if let Some(flag) = lk(&handle.tail_abort).take() {
        flag.store(true, Ordering::SeqCst);
    }
}

fn clear_if_current(app: &AppHandle, gen_id: u64, model: &str, voice: &str) {
    let live = app.state::<GeminiLive>();
    let mut slot = lk(&live.active);
    let matches = slot.as_ref().is_some_and(|h| h.gen_id == gen_id);
    if matches {
        if let Some(h) = slot.take() {
            h.output_playing.store(false, Ordering::SeqCst);
            abort_hold_tail(&h);
            stop_active_mic(&h);
        }
        drop(slot);
        emit_state(app, "idle", None, false, model, voice);
    }
}

/// Ensure Gemini key + notice consent are valid and return `(api_key, device, model, voice, prompt)`.
fn prepare_config(
    app: &AppHandle,
) -> Result<(String, Option<String>, String, String, String), (String, String)> {
    if let Err(e) = crate::ai::check_usable(Provider::Gemini) {
        return Err((e.kind, e.message));
    }
    let key = keys::read(Provider::Gemini)
        .map_err(|m| ("other".to_string(), m))?
        .ok_or_else(|| {
            (
                "no-provider".to_string(),
                "Set up a Gemini API key in Settings \u{2192} AI first.".to_string(),
            )
        })?;
    let (device, raw_model, raw_voice, system_prompt) =
        agent_live_config(&app.state::<Dictation>());
    let model = normalize_live_model(&raw_model);
    let voice = normalize_live_voice(&raw_voice).to_string();
    Ok((key, device, model, voice, system_prompt))
}

/// Start a brand-new Gemini Live WebSocket session (if none is active) and return its handle.
fn ensure_session(app: &AppHandle) -> Result<(SessionHandle, Option<String>), (String, String)> {
    let live = app.state::<GeminiLive>();
    if let Some(existing) = lk(&live.active).clone() {
        let (device, _, _, _) = agent_live_config(&app.state::<Dictation>());
        return Ok((existing, device));
    }

    let (api_key, device, model, voice, system_prompt) = prepare_config(app)?;
    let gen_id = live.next_gen.fetch_add(1, Ordering::SeqCst);
    let (tx, rx) = mpsc::channel::<LiveCmd>();
    let mic_open = Arc::new(AtomicBool::new(false));
    let output_playing = Arc::new(AtomicBool::new(false));
    let handle = SessionHandle {
        gen_id,
        tx,
        mic_open: mic_open.clone(),
        output_playing: output_playing.clone(),
        mic_gen: Arc::new(AtomicU64::new(0)),
        stop_mic: Arc::new(Mutex::new(None)),
        tail_abort: Arc::new(Mutex::new(None)),
        model: model.clone(),
        voice: voice.clone(),
        phase: Arc::new(Mutex::new("connecting".into())),
    };
    *lk(&live.active) = Some(handle.clone());
    emit_state(app, "connecting", None, true, &model, &voice);

    let app2 = app.clone();
    let m2 = model.clone();
    let v2 = voice.clone();
    thread::Builder::new()
        .name("krovi-gemini-live-ws".into())
        .spawn(move || {
            run_ws_worker(
                app2,
                gen_id,
                rx,
                mic_open,
                output_playing,
                api_key,
                m2,
                v2,
                system_prompt,
            );
        })
        .map_err(|e| {
            clear_if_current(app, gen_id, &model, &voice);
            (
                "other".to_string(),
                format!("Could not start live worker: {e}"),
            )
        })?;

    Ok((handle, device))
}

/// Open WASAPI microphone and stream 16 kHz PCM16 chunks into `handle.tx` until `stop_flag` fires.
fn spawn_mic_pump(
    app: &AppHandle,
    handle: &SessionHandle,
    device: Option<String>,
    send_end_on_release: bool,
) {
    abort_hold_tail(handle);
    stop_active_mic(handle);
    let my_mic_gen = handle.mic_gen.fetch_add(1, Ordering::SeqCst) + 1;
    let stop_flag = Arc::new(AtomicBool::new(false));
    let tail_abort = Arc::new(AtomicBool::new(false));
    *lk(&handle.stop_mic) = Some(stop_flag.clone());
    *lk(&handle.tail_abort) = Some(tail_abort.clone());

    let app2 = app.clone();
    let tx = handle.tx.clone();
    let mic_open = handle.mic_open.clone();
    let output_playing = handle.output_playing.clone();
    let mic_gen_ref = handle.mic_gen.clone();
    let stop_mic_slot = handle.stop_mic.clone();
    let tail_abort_slot = handle.tail_abort.clone();
    let gen_id = handle.gen_id;
    let model = handle.model.clone();
    let voice = handle.voice.clone();

    let spawned = thread::Builder::new()
        .name("krovi-gemini-live-mic".into())
        .spawn(move || {
            let rec = match Recorder::start(device) {
                Ok(r) => r,
                Err(e) => {
                    if mic_gen_ref.load(Ordering::SeqCst) == my_mic_gen {
                        *lk(&stop_mic_slot) = None;
                        *lk(&tail_abort_slot) = None;
                    }
                    emit_error(&app2, "mic", e);
                    return;
                }
            };

            if stop_flag.load(Ordering::SeqCst) || mic_gen_ref.load(Ordering::SeqCst) != my_mic_gen
            {
                if mic_gen_ref.load(Ordering::SeqCst) == my_mic_gen {
                    *lk(&stop_mic_slot) = None;
                    *lk(&tail_abort_slot) = None;
                }
                rec.cancel();
                let still_same = lk(&app2.state::<GeminiLive>().active)
                    .as_ref()
                    .is_some_and(|h| h.gen_id == gen_id);
                if still_same
                    && send_end_on_release
                    && !tail_abort.load(Ordering::SeqCst)
                    && mic_gen_ref.load(Ordering::SeqCst) == my_mic_gen
                {
                    let _ = tx.send(LiveCmd::CancelTurn);
                    emit_error(
                        &app2,
                        "nothing-heard",
                        "Hold the shortcut while speaking (or switch to Toggle in Settings).",
                    );
                }
                return;
            }

            mic_open.store(true, Ordering::SeqCst);
            if !output_playing.load(Ordering::SeqCst) {
                emit_state(&app2, "listening", Some(0.0), true, &model, &voice);
            }
            let mut last_level_emit = Instant::now();
            let mut was_muted = output_playing.load(Ordering::SeqCst);
            let mut total_samples: usize = 0;

            while !stop_flag.load(Ordering::SeqCst) {
                if rec.finished() || mic_gen_ref.load(Ordering::SeqCst) != my_mic_gen {
                    break;
                }
                let samples = rec.drain_pcm16();
                if output_playing.load(Ordering::SeqCst) {
                    // Discard speaker echo while Gemini is speaking so it cannot self-interrupt.
                    was_muted = true;
                } else {
                    if was_muted {
                        was_muted = false;
                        last_level_emit = Instant::now();
                        emit_state(&app2, "listening", Some(rec.level()), true, &model, &voice);
                    }
                    if !samples.is_empty() {
                        total_samples = total_samples.saturating_add(samples.len());
                        let bytes = pcm16_samples_to_le_bytes(&samples);
                        if tx.send(LiveCmd::AudioPcm(bytes)).is_err() {
                            break;
                        }
                    }
                    if last_level_emit.elapsed() >= LEVEL_EMIT_EVERY {
                        last_level_emit = Instant::now();
                        emit_state(&app2, "listening", Some(rec.level()), true, &model, &voice);
                    }
                }
                thread::sleep(MIC_POLL);
            }

            if mic_gen_ref.load(Ordering::SeqCst) == my_mic_gen {
                mic_open.store(false, Ordering::SeqCst);
                *lk(&stop_mic_slot) = None;
            }

            let still_same = || {
                lk(&app2.state::<GeminiLive>().active)
                    .as_ref()
                    .is_some_and(|h| h.gen_id == gen_id)
                    && mic_gen_ref.load(Ordering::SeqCst) == my_mic_gen
            };

            if send_end_on_release && !tail_abort.load(Ordering::SeqCst) && still_same() {
                if total_samples < MIN_HOLD_SAMPLES {
                    drop(rec);
                    if mic_gen_ref.load(Ordering::SeqCst) == my_mic_gen {
                        *lk(&tail_abort_slot) = None;
                    }
                    let _ = tx.send(LiveCmd::CancelTurn);
                    emit_error(
                        &app2,
                        "nothing-heard",
                        "Hold the shortcut while speaking (or switch to Toggle in Settings).",
                    );
                    return;
                }

                // Update UI immediately while letting WASAPI capture a brief natural tail so the
                // final syllable and server-side automatic VAD endpointing are not clipped.
                emit_state(&app2, "thinking", None, true, &model, &voice);
                let tail_until = Instant::now() + HOLD_RELEASE_TAIL;
                while Instant::now() < tail_until
                    && !tail_abort.load(Ordering::SeqCst)
                    && !rec.finished()
                    && mic_gen_ref.load(Ordering::SeqCst) == my_mic_gen
                {
                    let tail = rec.drain_pcm16();
                    if !tail.is_empty() && !output_playing.load(Ordering::SeqCst) {
                        let bytes = pcm16_samples_to_le_bytes(&tail);
                        if tx.send(LiveCmd::AudioPcm(bytes)).is_err() {
                            break;
                        }
                    }
                    thread::sleep(MIC_POLL);
                }

                let final_tail = rec.drain_pcm16();
                if !final_tail.is_empty()
                    && !tail_abort.load(Ordering::SeqCst)
                    && !output_playing.load(Ordering::SeqCst)
                {
                    let bytes = pcm16_samples_to_le_bytes(&final_tail);
                    let _ = tx.send(LiveCmd::AudioPcm(bytes));
                }
                drop(rec);
                if mic_gen_ref.load(Ordering::SeqCst) == my_mic_gen {
                    *lk(&tail_abort_slot) = None;
                }
                if !tail_abort.load(Ordering::SeqCst) && still_same() {
                    let _ = tx.send(LiveCmd::AudioEnd);
                }
            } else {
                drop(rec);
                if mic_gen_ref.load(Ordering::SeqCst) == my_mic_gen {
                    *lk(&tail_abort_slot) = None;
                }
            }
        });

    if let Err(e) = spawned {
        if handle.mic_gen.load(Ordering::SeqCst) == my_mic_gen {
            *lk(&handle.stop_mic) = None;
            *lk(&handle.tail_abort) = None;
        }
        emit_error(
            app,
            "mic",
            format!("Could not start microphone thread: {e}"),
        );
    }
}

#[cfg(feature = "net")]
#[allow(clippy::too_many_arguments)]
fn run_ws_worker(
    app: AppHandle,
    gen_id: u64,
    rx: mpsc::Receiver<LiveCmd>,
    mic_open: Arc<AtomicBool>,
    output_playing: Arc<AtomicBool>,
    api_key: String,
    model: String,
    voice: String,
    system_prompt: String,
) {
    use krovi_net::gemini_live::ws::{LivePollError, LiveSocket};
    use krovi_net::gemini_live::{
        build_audio_chunk_json, build_audio_end_json, build_client_text_json, build_setup_json,
        estimate_pcm_duration_ms, LiveServerEvent,
    };

    let setup_json = build_setup_json(&model, &voice, Some(&system_prompt));
    let mut ws = match LiveSocket::connect(&api_key, &setup_json) {
        Ok(s) => s,
        Err(msg) => {
            emit_error(&app, "network", msg);
            clear_if_current(&app, gen_id, &model, &voice);
            return;
        }
    };

    let mut setup_done = false;
    let mut pending_audio: Vec<Vec<u8>> = Vec::new();
    let mut pending_end = false;
    let mut pending_texts: Vec<String> = Vec::new();
    let mut turn_has_output_transcript = false;
    let mut turn_has_audio = false;
    let mut last_activity = Instant::now();
    let mut thinking_since: Option<Instant> = None;
    let mut gen_complete_without_audio_since: Option<Instant> = None;
    let mut output_playing_deadline: Option<Instant> = None;
    let mut last_audio_chunk_at: Option<Instant> = None;

    let idle_phase = |mic: &AtomicBool| {
        if mic.load(Ordering::SeqCst) {
            "listening"
        } else {
            "ready"
        }
    };

    loop {
        let mut did_work = false;

        while let Ok(cmd) = rx.try_recv() {
            did_work = true;
            last_activity = Instant::now();
            match cmd {
                LiveCmd::AudioPcm(bytes) => {
                    thinking_since = None;
                    gen_complete_without_audio_since = None;
                    if setup_done {
                        let payload = build_audio_chunk_json(&bytes);
                        if let Err(e) = ws.send_text(&payload) {
                            emit_error(&app, "network", e);
                            ws.close();
                            clear_if_current(&app, gen_id, &model, &voice);
                            return;
                        }
                    } else if pending_audio.len() < MAX_PENDING_AUDIO_CHUNKS {
                        pending_audio.push(bytes);
                    }
                }
                LiveCmd::AudioEnd => {
                    if setup_done {
                        if let Err(e) = ws.send_text(build_audio_end_json()) {
                            emit_error(&app, "network", e);
                            ws.close();
                            clear_if_current(&app, gen_id, &model, &voice);
                            return;
                        }
                        thinking_since = Some(Instant::now());
                        turn_has_audio = false;
                        gen_complete_without_audio_since = None;
                        emit_state(&app, "thinking", None, true, &model, &voice);
                    } else {
                        pending_end = true;
                        thinking_since = Some(Instant::now());
                    }
                }
                LiveCmd::CancelTurn => {
                    pending_end = false;
                    thinking_since = None;
                    gen_complete_without_audio_since = None;
                    if setup_done && !output_playing.load(Ordering::SeqCst) {
                        emit_state(&app, idle_phase(&mic_open), None, true, &model, &voice);
                    }
                }
                LiveCmd::OutputPlaybackChanged(playing) => {
                    if playing {
                        output_playing.store(true, Ordering::SeqCst);
                        let min_deadline = Instant::now() + Duration::from_millis(500);
                        if output_playing_deadline.is_none_or(|d| d < min_deadline) {
                            output_playing_deadline = Some(min_deadline);
                        }
                    } else {
                        let chunk_just_arrived = last_audio_chunk_at
                            .is_some_and(|t| t.elapsed() < SPEAKING_MIN_CHUNK_QUIET);
                        if !chunk_just_arrived {
                            let was = output_playing.swap(false, Ordering::SeqCst);
                            output_playing_deadline = None;
                            if was && setup_done && thinking_since.is_none() {
                                emit_state(&app, idle_phase(&mic_open), None, true, &model, &voice);
                            }
                        }
                    }
                }
                LiveCmd::ClientText(text) => {
                    if setup_done {
                        let payload = build_client_text_json(&text);
                        if let Err(e) = ws.send_text(&payload) {
                            emit_error(&app, "network", e);
                            ws.close();
                            clear_if_current(&app, gen_id, &model, &voice);
                            return;
                        }
                        thinking_since = Some(Instant::now());
                        turn_has_audio = false;
                        gen_complete_without_audio_since = None;
                        emit_state(&app, "thinking", None, true, &model, &voice);
                    } else {
                        pending_texts.push(text);
                        thinking_since = Some(Instant::now());
                    }
                }
                LiveCmd::Stop => {
                    output_playing.store(false, Ordering::SeqCst);
                    ws.close();
                    clear_if_current(&app, gen_id, &model, &voice);
                    return;
                }
            }
        }

        match ws.poll_events() {
            Ok(events) => {
                if !events.is_empty() {
                    did_work = true;
                    last_activity = Instant::now();
                }
                for ev in events {
                    match ev {
                        LiveServerEvent::SetupComplete => {
                            setup_done = true;
                            for bytes in pending_audio.drain(..) {
                                let payload = build_audio_chunk_json(&bytes);
                                if let Err(e) = ws.send_text(&payload) {
                                    emit_error(&app, "network", e);
                                    ws.close();
                                    clear_if_current(&app, gen_id, &model, &voice);
                                    return;
                                }
                            }
                            let mut sent_turn_end = false;
                            if pending_end {
                                pending_end = false;
                                if let Err(e) = ws.send_text(build_audio_end_json()) {
                                    emit_error(&app, "network", e);
                                    ws.close();
                                    clear_if_current(&app, gen_id, &model, &voice);
                                    return;
                                }
                                sent_turn_end = true;
                            }
                            for text in pending_texts.drain(..) {
                                let payload = build_client_text_json(&text);
                                if let Err(e) = ws.send_text(&payload) {
                                    emit_error(&app, "network", e);
                                    ws.close();
                                    clear_if_current(&app, gen_id, &model, &voice);
                                    return;
                                }
                                sent_turn_end = true;
                            }
                            if sent_turn_end {
                                thinking_since = Some(Instant::now());
                                turn_has_audio = false;
                                gen_complete_without_audio_since = None;
                                emit_state(&app, "thinking", None, true, &model, &voice);
                            } else if thinking_since.is_some() {
                                emit_state(&app, "thinking", None, true, &model, &voice);
                            } else {
                                emit_state(&app, idle_phase(&mic_open), None, true, &model, &voice);
                            }
                        }
                        LiveServerEvent::InputTranscript { text } => {
                            let _ = app.emit(
                                "gemini-live-transcript",
                                json!({ "role": "user", "text": text }),
                            );
                        }
                        LiveServerEvent::OutputTranscript { text } => {
                            thinking_since = None;
                            gen_complete_without_audio_since = None;
                            turn_has_output_transcript = true;
                            let _ = app.emit(
                                "gemini-live-transcript",
                                json!({ "role": "model", "text": text }),
                            );
                        }
                        LiveServerEvent::OutputText { text } => {
                            thinking_since = None;
                            gen_complete_without_audio_since = None;
                            if !turn_has_output_transcript {
                                let _ = app.emit(
                                    "gemini-live-transcript",
                                    json!({ "role": "model", "text": text }),
                                );
                            }
                        }
                        LiveServerEvent::AudioChunk {
                            base64,
                            sample_rate,
                        } => {
                            thinking_since = None;
                            gen_complete_without_audio_since = None;
                            turn_has_audio = true;
                            let now = Instant::now();
                            last_audio_chunk_at = Some(now);
                            output_playing.store(true, Ordering::SeqCst);
                            let dur_ms = estimate_pcm_duration_ms(&base64, sample_rate).max(40);
                            let base = output_playing_deadline.filter(|&d| d > now).unwrap_or(now);
                            output_playing_deadline = Some(base + Duration::from_millis(dur_ms));
                            emit_state(&app, "speaking", None, true, &model, &voice);
                            let _ = app.emit(
                                "gemini-live-audio",
                                json!({ "data": base64, "sampleRate": sample_rate }),
                            );
                        }
                        LiveServerEvent::GenerationComplete => {
                            if thinking_since.is_some() && !turn_has_audio {
                                gen_complete_without_audio_since = Some(Instant::now());
                            }
                        }
                        LiveServerEvent::Interrupted => {
                            thinking_since = None;
                            gen_complete_without_audio_since = None;
                            turn_has_audio = false;
                            turn_has_output_transcript = false;
                            output_playing.store(false, Ordering::SeqCst);
                            output_playing_deadline = None;
                            let _ = app.emit("gemini-live-interrupted", json!({}));
                            emit_state(&app, idle_phase(&mic_open), None, true, &model, &voice);
                        }
                        LiveServerEvent::TurnComplete => {
                            thinking_since = None;
                            gen_complete_without_audio_since = None;
                            turn_has_audio = false;
                            turn_has_output_transcript = false;
                            let _ = app.emit("gemini-live-turn-complete", json!({}));
                            if !output_playing.load(Ordering::SeqCst) {
                                emit_state(&app, idle_phase(&mic_open), None, true, &model, &voice);
                            }
                        }
                        LiveServerEvent::GoAway { .. } => {
                            output_playing.store(false, Ordering::SeqCst);
                            emit_error(
                                &app,
                                "other",
                                "Gemini Live closed the session. Press the shortcut to reconnect.",
                            );
                            ws.close();
                            clear_if_current(&app, gen_id, &model, &voice);
                            return;
                        }
                        LiveServerEvent::Error { message } => {
                            output_playing.store(false, Ordering::SeqCst);
                            emit_error(&app, "other", message);
                            ws.close();
                            clear_if_current(&app, gen_id, &model, &voice);
                            return;
                        }
                    }
                }
            }
            Err(LivePollError::Closed) => {
                clear_if_current(&app, gen_id, &model, &voice);
                return;
            }
            Err(LivePollError::Fatal(msg)) => {
                emit_error(&app, "network", msg);
                clear_if_current(&app, gen_id, &model, &voice);
                return;
            }
        }

        // Safety net: unmute if the webview never reported WebAudio playback completion.
        if output_playing.load(Ordering::SeqCst)
            && output_playing_deadline.is_some_and(|d| Instant::now() > d + SPEAKING_SAFETY_GRACE)
        {
            output_playing.store(false, Ordering::SeqCst);
            output_playing_deadline = None;
            if setup_done && thinking_since.is_none() {
                emit_state(&app, idle_phase(&mic_open), None, true, &model, &voice);
            }
        }

        // Exit "thinking" cleanly if generationComplete arrived without audio or turnComplete.
        if thinking_since.is_some()
            && gen_complete_without_audio_since.is_some_and(|t| t.elapsed() >= GEN_COMPLETE_GRACE)
        {
            thinking_since = None;
            gen_complete_without_audio_since = None;
            turn_has_audio = false;
            turn_has_output_transcript = false;
            let _ = app.emit("gemini-live-turn-complete", json!({}));
            if !output_playing.load(Ordering::SeqCst) {
                emit_state(&app, idle_phase(&mic_open), None, true, &model, &voice);
            }
        }

        // Never leave the pill stuck on "Gemini thinking" if the server stayed silent.
        if thinking_since.is_some_and(|t| t.elapsed() >= THINKING_TIMEOUT)
            && !output_playing.load(Ordering::SeqCst)
        {
            thinking_since = None;
            gen_complete_without_audio_since = None;
            turn_has_audio = false;
            turn_has_output_transcript = false;
            let _ = app.emit("gemini-live-turn-complete", json!({}));
            emit_state(&app, idle_phase(&mic_open), None, true, &model, &voice);
            emit_error(
                &app,
                "nothing-heard",
                "Gemini didn't reply. Hold the shortcut slightly longer while speaking.",
            );
        }

        if !mic_open.load(Ordering::SeqCst)
            && !output_playing.load(Ordering::SeqCst)
            && thinking_since.is_none()
            && last_activity.elapsed() > IDLE_TIMEOUT
        {
            ws.close();
            clear_if_current(&app, gen_id, &model, &voice);
            return;
        }

        if !did_work {
            thread::sleep(Duration::from_millis(12));
        }
    }
}

#[cfg(not(feature = "net"))]
#[allow(clippy::too_many_arguments)]
fn run_ws_worker(
    app: AppHandle,
    gen_id: u64,
    _rx: mpsc::Receiver<LiveCmd>,
    _mic_open: Arc<AtomicBool>,
    _output_playing: Arc<AtomicBool>,
    _api_key: String,
    model: String,
    voice: String,
    _system_prompt: String,
) {
    emit_error(
        &app,
        "unavailable",
        "Networking is disabled in this build (enable the `net` feature).",
    );
    clear_if_current(&app, gen_id, &model, &voice);
}

/// Handle `Ctrl + Alt + A` (or custom Agent shortcut) when `agentEngine == "gemini-live"`.
pub fn on_shortcut(app: &AppHandle, state: ShortcutState, talk_mode: TalkMode) {
    let active = lk(&app.state::<GeminiLive>().active).clone();
    match (talk_mode, state) {
        (TalkMode::Hold, ShortcutState::Pressed) => {
            if active
                .as_ref()
                .is_some_and(|h| h.mic_open.load(Ordering::SeqCst) || lk(&h.stop_mic).is_some())
            {
                return; // OS key repeat while held
            }
            if let Some(ref h) = active {
                abort_hold_tail(h);
                if h.output_playing.swap(false, Ordering::SeqCst) {
                    let _ = h.tx.send(LiveCmd::OutputPlaybackChanged(false));
                }
            }
            let _ = app.emit("gemini-live-interrupted", json!({}));
            match ensure_session(app) {
                Ok((handle, device)) => spawn_mic_pump(app, &handle, device, true),
                Err((kind, msg)) => emit_error(app, &kind, msg),
            }
        }
        (TalkMode::Hold, ShortcutState::Released) => {
            if let Some(handle) = active {
                stop_active_mic(&handle);
            }
        }
        (TalkMode::Toggle, ShortcutState::Pressed) => {
            if active.is_some() {
                stop_session(app);
            } else {
                match ensure_session(app) {
                    Ok((handle, device)) => spawn_mic_pump(app, &handle, device, false),
                    Err((kind, msg)) => emit_error(app, &kind, msg),
                }
            }
        }
        (TalkMode::Toggle, ShortcutState::Released) => {}
    }
}

pub fn stop_session(app: &AppHandle) {
    let handle = lk(&app.state::<GeminiLive>().active).take();
    if let Some(h) = handle {
        h.output_playing.store(false, Ordering::SeqCst);
        abort_hold_tail(&h);
        stop_active_mic(&h);
        let _ = h.tx.send(LiveCmd::Stop);
        let _ = app.emit("gemini-live-interrupted", json!({}));
        emit_state(app, "idle", None, false, &h.model, &h.voice);
    }
}

pub fn setup(app: &mut tauri::App) -> Result<(), Box<dyn std::error::Error>> {
    app.manage(GeminiLive::new());
    Ok(())
}

// ---------------------------------------------------------------------------------------
// Commands
// ---------------------------------------------------------------------------------------

#[tauri::command]
pub fn gemini_live_catalog() -> GeminiLiveCatalog {
    GeminiLiveCatalog {
        models: LIVE_MODELS,
        voices: LIVE_VOICES,
    }
}

#[tauri::command]
pub fn gemini_live_status(app: AppHandle) -> GeminiLiveStatus {
    let (_, raw_model, raw_voice, _) = agent_live_config(&app.state::<Dictation>());
    let default_model = normalize_live_model(&raw_model);
    let default_voice = normalize_live_voice(&raw_voice).to_string();
    let readiness_error = crate::ai::check_usable(Provider::Gemini)
        .err()
        .map(|e| e.message);
    let ready = readiness_error.is_none();

    if let Some(h) = lk(&app.state::<GeminiLive>().active).clone() {
        let phase = lk(&h.phase).clone();
        GeminiLiveStatus {
            state: phase,
            session_active: true,
            model: h.model,
            voice: h.voice,
            ready,
            readiness_error,
        }
    } else {
        GeminiLiveStatus {
            state: "idle".into(),
            session_active: false,
            model: default_model,
            voice: default_voice,
            ready,
            readiness_error,
        }
    }
}

/// Start a continuous (hands-free) Gemini Live voice session from the UI.
#[tauri::command]
pub fn gemini_live_start(app: AppHandle) -> CmdResult<GeminiLiveStatus> {
    let active = lk(&app.state::<GeminiLive>().active).clone();
    if active.is_some() {
        return Ok(gemini_live_status(app));
    }
    match ensure_session(&app) {
        Ok((handle, device)) => {
            spawn_mic_pump(&app, &handle, device, false);
            Ok(gemini_live_status(app))
        }
        Err((kind, msg)) => {
            emit_error(&app, &kind, msg.clone());
            Err(msg)
        }
    }
}

/// Stop any active Gemini Live session and close the microphone.
#[tauri::command]
pub fn gemini_live_stop(app: AppHandle) -> GeminiLiveStatus {
    stop_session(&app);
    gemini_live_status(app)
}

/// Send a typed message into the active Gemini Live session (starting one if needed).
#[tauri::command]
pub fn gemini_live_send_text(app: AppHandle, text: String) -> CmdResult<()> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Err("Enter a message first.".into());
    }
    let (handle, _) = ensure_session(&app).map_err(|(kind, msg)| {
        emit_error(&app, &kind, msg.clone());
        msg
    })?;
    let _ = app.emit(
        "gemini-live-transcript",
        json!({ "role": "user", "text": trimmed }),
    );
    handle
        .tx
        .send(LiveCmd::ClientText(trimmed.to_string()))
        .map_err(|_| "Gemini Live session closed.".to_string())
}

/// Inform Rust when WebAudio starts or finishes speaking Gemini's reply so mic capture stays
/// muted until speaker playback completes and cannot echo back into Gemini.
#[tauri::command]
pub fn gemini_live_set_output_playing(app: AppHandle, playing: bool) {
    if let Some(h) = lk(&app.state::<GeminiLive>().active).clone() {
        let _ = h.tx.send(LiveCmd::OutputPlaybackChanged(playing));
    }
}
