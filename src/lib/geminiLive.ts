// Gemini Live bidirectional voice conversation client for Agent Mode.
// Contract: src-tauri/src/gemini_live.rs.
//
// * Rust owns the WebSocket and reads the Gemini API key directly from Windows Credential Manager.
// * Rust streams 16 kHz mono PCM16 microphone audio and emits:
//   - `gemini-live-state`        { state, level?, sessionActive, model, voice }
//   - `gemini-live-audio`        { data (base64 PCM16), sampleRate }
//   - `gemini-live-interrupted`  {}
//   - `gemini-live-transcript`   { role: 'user' | 'model', text }
//   - `gemini-live-turn-complete` {}
//   - `gemini-live-error`        { kind, message }
// * This module decodes and schedules the 24 kHz PCM16 audio playback via Web Audio API.

import { get, writable } from 'svelte/store';
import { invoke, isTauri, listen } from './bridge';
import { meterLevel, type DictationFace } from './dictation';
import { DESKTOP_ONLY, errorText } from './circle';

export type GeminiLivePhase = 'idle' | 'connecting' | 'listening' | 'thinking' | 'speaking' | 'ready';

export interface GeminiLiveState {
  phase: GeminiLivePhase;
  level: number;
  sessionActive: boolean;
  model: string;
  voice: string;
  ready: boolean;
  readinessError: string | null;
  error: string | null;
  errorTone?: 'error' | 'info';
}

export interface GeminiLiveTurn {
  id: number;
  role: 'user' | 'model';
  text: string;
  complete: boolean;
}

export interface GeminiLiveModelOption {
  id: string;
  label: string;
}

export interface GeminiLiveCatalog {
  models: GeminiLiveModelOption[];
  voices: string[];
}

export const DEFAULT_GEMINI_LIVE_MODELS: GeminiLiveModelOption[] = [
  { id: 'gemini-3.8-live', label: 'Gemini 3.8 Live (recommended)' },
  { id: 'gemini-3.8-live-extended-thinking', label: 'Gemini 3.8 Live Extended Thinking' },
  { id: 'gemini-3.1-flash-live-preview', label: 'Gemini 3.1 Flash Live (Preview)' },
  { id: 'gemini-2.5-flash-native-audio-latest', label: 'Gemini 2.5 Flash Native Audio' },
];

export const DEFAULT_GEMINI_LIVE_VOICES = ['Puck', 'Charon', 'Kore', 'Fenrir', 'Aoede'];

export const geminiLiveState = writable<GeminiLiveState>({
  phase: 'idle',
  level: 0,
  sessionActive: false,
  model: 'gemini-3.8-live',
  voice: 'Puck',
  ready: false,
  readinessError: null,
  error: null,
  errorTone: 'error',
});

export const geminiLiveTurns = writable<GeminiLiveTurn[]>([]);

export const geminiLiveCatalog = writable<GeminiLiveCatalog>({
  models: DEFAULT_GEMINI_LIVE_MODELS,
  voices: DEFAULT_GEMINI_LIVE_VOICES,
});

// ---------------------------------------------------------------------------------------------
// Audio playback (24 kHz mono PCM16 LE -> Web Audio API)
// ---------------------------------------------------------------------------------------------

const ECHO_GUARD_MS = 350;

let audioCtx: AudioContext | null = null;
let nextPlayTime = 0;
const activeSources = new Set<AudioBufferSourceNode>();
let lastReportedPlaying = false;
let unmuteTimer: ReturnType<typeof setTimeout> | null = null;

function clearUnmuteTimer(): void {
  if (unmuteTimer) {
    clearTimeout(unmuteTimer);
    unmuteTimer = null;
  }
}

function notifyOutputPlaying(playing: boolean): void {
  if (lastReportedPlaying === playing) return;
  lastReportedPlaying = playing;
  if (!isTauri()) return;
  void invoke('gemini_live_set_output_playing', { playing }).catch(() => {});
}

function scheduleOutputIdleCheck(): void {
  clearUnmuteTimer();
  unmuteTimer = setTimeout(() => {
    unmuteTimer = null;
    if (activeSources.size === 0) {
      notifyOutputPlaying(false);
    }
  }, ECHO_GUARD_MS);
}

function ensureAudioContext(): AudioContext | null {
  if (typeof window === 'undefined') return null;
  const Ctor = window.AudioContext ?? (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
  if (!Ctor) return null;
  if (!audioCtx || audioCtx.state === 'closed') {
    audioCtx = new Ctor();
    nextPlayTime = 0;
  }
  if (audioCtx.state === 'suspended') {
    void audioCtx.resume().catch(() => {});
  }
  return audioCtx;
}

export function stopGeminiAudioPlayback(): void {
  clearUnmuteTimer();
  for (const src of activeSources) {
    try {
      src.stop();
    } catch {
      /* already stopped */
    }
  }
  activeSources.clear();
  nextPlayTime = 0;
  notifyOutputPlaying(false);
}

export function decodePcm16Base64ToFloat32(base64: string): Float32Array {
  const bin = atob(base64);
  const len = bin.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i += 1) bytes[i] = bin.charCodeAt(i);
  const sampleCount = Math.floor(bytes.length / 2);
  const out = new Float32Array(sampleCount);
  const view = new DataView(bytes.buffer, bytes.byteOffset, sampleCount * 2);
  for (let i = 0; i < sampleCount; i += 1) {
    out[i] = view.getInt16(i * 2, true) / 32768;
  }
  return out;
}

function playPcmChunk(base64: string, sampleRate: number): void {
  const ctx = ensureAudioContext();
  if (!ctx) return;
  let samples: Float32Array;
  try {
    samples = decodePcm16Base64ToFloat32(base64);
  } catch {
    return;
  }
  if (samples.length === 0) return;

  const rate = sampleRate >= 8000 && sampleRate <= 96000 ? sampleRate : 24000;
  const buf = ctx.createBuffer(1, samples.length, rate);
  buf.getChannelData(0).set(samples);

  const src = ctx.createBufferSource();
  src.buffer = buf;
  src.connect(ctx.destination);

  const now = ctx.currentTime;
  const startAt = Math.max(now + 0.015, nextPlayTime);
  nextPlayTime = startAt + buf.duration;

  clearUnmuteTimer();
  notifyOutputPlaying(true);
  activeSources.add(src);
  src.onended = () => {
    activeSources.delete(src);
    if (activeSources.size === 0) {
      scheduleOutputIdleCheck();
    }
  };
  src.start(startAt);
}

// ---------------------------------------------------------------------------------------------
// Pill integration
// ---------------------------------------------------------------------------------------------

export function geminiPillFace(st: GeminiLiveState): DictationFace | null {
  if (st.error) {
    return {
      kind: 'notice',
      tone: st.errorTone ?? 'error',
      label: 'Gemini Live',
      detail: st.error,
    };
  }
  if (!st.sessionActive && st.phase === 'idle') return null;
  switch (st.phase) {
    case 'connecting':
      return { kind: 'loading-model', label: 'Connecting Gemini Live' };
    case 'listening':
      return { kind: 'listening', label: 'Gemini Live', level: meterLevel(st.level) };
    case 'thinking':
      return { kind: 'transcribing', label: 'Gemini thinking' };
    case 'speaking':
      return { kind: 'transcribing', label: 'Gemini speaking' };
    case 'ready':
      return null;
    default:
      return null;
  }
}

// ---------------------------------------------------------------------------------------------
// Event reducers + wiring
// ---------------------------------------------------------------------------------------------

let turnSeq = 1;
let errorTimer: ReturnType<typeof setTimeout> | null = null;

function clearErrorTimer(): void {
  if (errorTimer) {
    clearTimeout(errorTimer);
    errorTimer = null;
  }
}

export function dismissGeminiLiveError(): void {
  clearErrorTimer();
  geminiLiveState.update((s) => (s.error ? { ...s, error: null } : s));
}

function appendTranscript(role: 'user' | 'model', text: string): void {
  if (!text) return;
  geminiLiveTurns.update((list) => {
    const last = list[list.length - 1];
    if (last && last.role === role && !last.complete) {
      const updated = { ...last, text: `${last.text}${text}` };
      return [...list.slice(0, -1), updated];
    }
    const next = list.map((t) => (t.complete ? t : { ...t, complete: true }));
    return [...next, { id: turnSeq++, role, text, complete: false }];
  });
}

function finalizeTurns(): void {
  geminiLiveTurns.update((list) => list.map((t) => (t.complete ? t : { ...t, complete: true })));
}

let wired: Promise<void> | null = null;

type StateEvent = {
  state?: string;
  level?: number;
  sessionActive?: boolean;
  model?: string;
  voice?: string;
};

type AudioEvent = {
  data?: string;
  sampleRate?: number;
};

type TranscriptEvent = {
  role?: string;
  text?: string;
};

type ErrorEvent = {
  kind?: string;
  message?: string;
};

const VALID_PHASES = new Set<GeminiLivePhase>(['idle', 'connecting', 'listening', 'thinking', 'speaking', 'ready']);

export function wireGeminiLive(): Promise<void> {
  if (!isTauri()) return Promise.resolve();
  if (!wired) {
    wired = Promise.all([
      listen<StateEvent>('gemini-live-state', (p) => {
        const raw = typeof p?.state === 'string' ? p.state : 'idle';
        const phase: GeminiLivePhase = VALID_PHASES.has(raw as GeminiLivePhase) ? (raw as GeminiLivePhase) : 'idle';
        const sessionActive = Boolean(p?.sessionActive);
        if (!sessionActive || phase === 'idle') stopGeminiAudioPlayback();
        if (sessionActive && phase !== 'idle') clearErrorTimer();
        geminiLiveState.update((s) => ({
          ...s,
          phase,
          level: typeof p?.level === 'number' && Number.isFinite(p.level) ? p.level : 0,
          sessionActive,
          model: typeof p?.model === 'string' && p.model ? p.model : s.model,
          voice: typeof p?.voice === 'string' && p.voice ? p.voice : s.voice,
          error: sessionActive && phase !== 'idle' ? null : s.error,
        }));
      }),
      listen<AudioEvent>('gemini-live-audio', (p) => {
        if (typeof p?.data === 'string' && p.data) {
          playPcmChunk(p.data, typeof p.sampleRate === 'number' ? p.sampleRate : 24000);
        }
      }),
      listen<unknown>('gemini-live-interrupted', () => {
        stopGeminiAudioPlayback();
        finalizeTurns();
      }),
      listen<TranscriptEvent>('gemini-live-transcript', (p) => {
        const role = p?.role === 'user' ? 'user' : p?.role === 'model' ? 'model' : null;
        if (role && typeof p?.text === 'string' && p.text) {
          appendTranscript(role, p.text);
        }
      }),
      listen<unknown>('gemini-live-turn-complete', () => {
        finalizeTurns();
        if (activeSources.size === 0 && !unmuteTimer) {
          scheduleOutputIdleCheck();
        }
      }),
      listen<ErrorEvent>('gemini-live-error', (p) => {
        const msg = (p?.message ?? '').trim() || 'Gemini Live connection failed.';
        const tone: 'error' | 'info' = p?.kind === 'nothing-heard' ? 'info' : 'error';
        if (tone === 'error') stopGeminiAudioPlayback();
        clearErrorTimer();
        geminiLiveState.update((s) => ({ ...s, error: msg, errorTone: tone }));
        errorTimer = setTimeout(() => {
          geminiLiveState.update((s) => (s.error === msg ? { ...s, error: null } : s));
        }, 5500);
      }),
    ])
      .then(() => undefined)
      .catch(() => {
        wired = null;
      });
  }
  return wired;
}

// ---------------------------------------------------------------------------------------------
// Commands
// ---------------------------------------------------------------------------------------------

export async function loadGeminiLiveStatus(): Promise<GeminiLiveState> {
  if (!isTauri()) return get(geminiLiveState);
  void wireGeminiLive();
  const raw = await invoke<{
    state: string;
    sessionActive: boolean;
    model: string;
    voice: string;
    ready: boolean;
    readinessError: string | null;
  }>('gemini_live_status');
  const phase: GeminiLivePhase = VALID_PHASES.has(raw.state as GeminiLivePhase)
    ? (raw.state as GeminiLivePhase)
    : 'idle';
  geminiLiveState.update((s) => ({
    ...s,
    phase,
    sessionActive: raw.sessionActive,
    model: raw.model || s.model,
    voice: raw.voice || s.voice,
    ready: raw.ready,
    readinessError: raw.readinessError,
  }));
  return get(geminiLiveState);
}

export async function loadGeminiLiveCatalog(): Promise<GeminiLiveCatalog> {
  if (!isTauri()) return get(geminiLiveCatalog);
  const raw = await invoke<GeminiLiveCatalog>('gemini_live_catalog');
  if (raw?.models?.length && raw?.voices?.length) {
    geminiLiveCatalog.set(raw);
  }
  return get(geminiLiveCatalog);
}

export async function startGeminiLive(): Promise<void> {
  if (!isTauri()) throw new Error(DESKTOP_ONLY);
  ensureAudioContext();
  void wireGeminiLive();
  try {
    await invoke('gemini_live_start');
  } catch (e) {
    throw new Error(errorText(e));
  }
}

export async function stopGeminiLive(): Promise<void> {
  stopGeminiAudioPlayback();
  finalizeTurns();
  if (!isTauri()) return;
  try {
    await invoke('gemini_live_stop');
  } catch {
    /* ignore */
  }
}

export async function sendGeminiLiveText(text: string): Promise<void> {
  if (!isTauri()) throw new Error(DESKTOP_ONLY);
  ensureAudioContext();
  void wireGeminiLive();
  try {
    await invoke('gemini_live_send_text', { text });
  } catch (e) {
    throw new Error(errorText(e));
  }
}

export function clearGeminiLiveTurns(): void {
  geminiLiveTurns.set([]);
}
