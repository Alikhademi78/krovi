import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { get } from 'svelte/store';

function encodePcm16ToBase64(samples: number[]): string {
  const buf = new Uint8Array(samples.length * 2);
  const view = new DataView(buf.buffer);
  for (let i = 0; i < samples.length; i += 1) {
    view.setInt16(i * 2, samples[i], true);
  }
  let bin = '';
  for (let i = 0; i < buf.length; i += 1) {
    bin += String.fromCharCode(buf[i]);
  }
  return btoa(bin);
}

describe('geminiLive pure helpers and non-Tauri guards', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });
  afterEach(() => {
    vi.useRealTimers();
    vi.resetModules();
    vi.doUnmock('./bridge');
  });

  it('decodes base64 little-endian PCM16 into normalized Float32 samples', async () => {
    vi.resetModules();
    const g = await import('./geminiLive');
    const b64 = encodePcm16ToBase64([0, 16384, -32768, 32767]);
    const out = g.decodePcm16Base64ToFloat32(b64);
    expect(out.length).toBe(4);
    expect(out[0]).toBeCloseTo(0, 5);
    expect(out[1]).toBeCloseTo(0.5, 5);
    expect(out[2]).toBeCloseTo(-1, 5);
    expect(out[3]).toBeCloseTo(32767 / 32768, 5);
  });

  it('maps Gemini Live phases and errors to pill faces', async () => {
    vi.resetModules();
    const g = await import('./geminiLive');
    const base = {
      phase: 'idle' as const,
      level: 0,
      sessionActive: false,
      model: 'gemini-3.8-live',
      voice: 'Puck',
      ready: true,
      readinessError: null,
      error: null,
    };

    expect(g.geminiPillFace(base)).toBeNull();
    expect(g.geminiPillFace({ ...base, phase: 'ready', sessionActive: true })).toBeNull();
    expect(g.geminiPillFace({ ...base, phase: 'connecting', sessionActive: true })).toMatchObject({
      kind: 'loading-model',
      label: 'Connecting Gemini Live',
    });
    expect(g.geminiPillFace({ ...base, phase: 'listening', level: 0.4, sessionActive: true })).toMatchObject({
      kind: 'listening',
      label: 'Gemini Live',
    });
    expect(g.geminiPillFace({ ...base, phase: 'thinking', sessionActive: true })).toMatchObject({
      kind: 'transcribing',
      label: 'Gemini thinking',
    });
    expect(g.geminiPillFace({ ...base, phase: 'speaking', sessionActive: true })).toMatchObject({
      kind: 'transcribing',
      label: 'Gemini speaking',
    });
    expect(g.geminiPillFace({ ...base, phase: 'listening', sessionActive: true, error: 'Invalid API key' })).toEqual({
      kind: 'notice',
      tone: 'error',
      label: 'Gemini Live',
      detail: 'Invalid API key',
    });
    expect(g.geminiPillFace({ ...base, phase: 'listening', sessionActive: true, error: 'Hold shortcut', errorTone: 'info' })).toEqual({
      kind: 'notice',
      tone: 'info',
      label: 'Gemini Live',
      detail: 'Hold shortcut',
    });

  });

  it('rejects live commands in non-Tauri browser mode', async () => {
    vi.resetModules();
    const g = await import('./geminiLive');
    await expect(g.startGeminiLive()).rejects.toThrow(g.DEFAULT_GEMINI_LIVE_MODELS.length > 0 ? /desktop/i : /./);
    await expect(g.sendGeminiLiveText('hello')).rejects.toThrow(/desktop/i);
    await expect(g.stopGeminiLive()).resolves.toBeUndefined();
    const st = await g.loadGeminiLiveStatus();
    expect(st.phase).toBe('idle');
  });
});

describe('geminiLive Tauri event wiring and commands', () => {
  const calls: { cmd: string; args?: Record<string, unknown> }[] = [];
  const handlers = new Map<string, (payload: unknown) => void>();

  beforeEach(() => {
    vi.useFakeTimers();
    calls.length = 0;
    handlers.clear();
    vi.resetModules();
    vi.doMock('./bridge', () => ({
      isTauri: () => true,
      listen: async (event: string, cb: (payload: unknown) => void) => {
        handlers.set(event, cb);
        return () => handlers.delete(event);
      },
      invoke: async (cmd: string, args?: Record<string, unknown>) => {
        calls.push({ cmd, args });
        switch (cmd) {
          case 'gemini_live_status':
            return {
              state: 'ready',
              sessionActive: true,
              model: 'gemini-3.8-live',
              voice: 'Aoede',
              ready: true,
              readinessError: null,
            };
          case 'gemini_live_catalog':
            return {
              models: [{ id: 'gemini-3.8-live', label: 'Gemini 3.8 Live' }],
              voices: ['Puck', 'Aoede'],
            };
          default:
            return undefined;
        }
      },
    }));
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.doUnmock('./bridge');
  });

  it('loads status and catalog and invokes start, text, and stop commands', async () => {
    const g = await import('./geminiLive');
    const st = await g.loadGeminiLiveStatus();
    expect(st).toMatchObject({
      phase: 'ready',
      sessionActive: true,
      model: 'gemini-3.8-live',
      voice: 'Aoede',
      ready: true,
    });

    const cat = await g.loadGeminiLiveCatalog();
    expect(cat.voices).toEqual(['Puck', 'Aoede']);

    await g.startGeminiLive();
    await g.sendGeminiLiveText('Summarize my day');
    await g.stopGeminiLive();

    expect(calls.map((c) => c.cmd)).toEqual([
      'gemini_live_status',
      'gemini_live_catalog',
      'gemini_live_start',
      'gemini_live_send_text',
      'gemini_live_stop',
    ]);
    expect(calls[3].args).toEqual({ text: 'Summarize my day' });
  });

  it('streams incremental user and model transcripts and handles turn completion and errors', async () => {
    const g = await import('./geminiLive');
    await g.wireGeminiLive();

    const emit = (event: string, payload: unknown) => handlers.get(event)?.(payload);

    emit('gemini-live-state', {
      state: 'listening',
      level: 0.25,
      sessionActive: true,
      model: 'gemini-3.8-live',
      voice: 'Kore',
    });
    expect(get(g.geminiLiveState)).toMatchObject({
      phase: 'listening',
      level: 0.25,
      sessionActive: true,
      voice: 'Kore',
    });

    emit('gemini-live-transcript', { role: 'user', text: 'سلام ' });
    emit('gemini-live-transcript', { role: 'user', text: 'کورتکس' });
    let turns = get(g.geminiLiveTurns);
    expect(turns).toHaveLength(1);
    expect(turns[0]).toMatchObject({ role: 'user', text: 'سلام کورتکس', complete: false });

    emit('gemini-live-transcript', { role: 'model', text: 'Hello ' });
    emit('gemini-live-transcript', { role: 'model', text: 'Cortex!' });
    turns = get(g.geminiLiveTurns);
    expect(turns).toHaveLength(2);
    expect(turns[0].complete).toBe(true);
    expect(turns[1]).toMatchObject({ role: 'model', text: 'Hello Cortex!', complete: false });

    emit('gemini-live-turn-complete', {});
    expect(get(g.geminiLiveTurns)[1].complete).toBe(true);

    emit('gemini-live-error', { kind: 'auth', message: 'API key missing' });
    expect(get(g.geminiLiveState).error).toBe('API key missing');

    // Active non-idle state clears a prior error; otherwise error auto-clears after 5.5s
    vi.advanceTimersByTime(5600);
    expect(get(g.geminiLiveState).error).toBeNull();

    emit('gemini-live-error', { kind: 'network', message: 'Socket closed' });
    expect(get(g.geminiLiveState).error).toBe('Socket closed');
    g.dismissGeminiLiveError();
    expect(get(g.geminiLiveState).error).toBeNull();

    g.clearGeminiLiveTurns();
    expect(get(g.geminiLiveTurns)).toEqual([]);
  });

  it('notifies Rust when WebAudio playback starts and unmutes after echo guard delay', async () => {
    const sources: { onended: (() => void) | null; stop: ReturnType<typeof vi.fn>; start: ReturnType<typeof vi.fn> }[] = [];
    class MockAudioContext {
      state = 'running';
      currentTime = 0;
      destination = {};
      resume = vi.fn(async () => {});
      createBuffer(_ch: number, len: number, sampleRate: number) {
        return {
          duration: len / sampleRate,
          getChannelData: () => new Float32Array(len),
        };
      }
      createBufferSource() {
        const s = {
          buffer: null as unknown,
          connect: vi.fn(),
          start: vi.fn(),
          stop: vi.fn(),
          onended: null as (() => void) | null,
        };
        sources.push(s);
        return s;
      }
    }
    vi.stubGlobal('window', { AudioContext: MockAudioContext });
    try {
      const g = await import('./geminiLive');
      await g.wireGeminiLive();

      const b64 = encodePcm16ToBase64(new Array(2400).fill(1000)); // 100 ms at 24 kHz
      handlers.get('gemini-live-audio')?.({ data: b64, sampleRate: 24000 });

      expect(calls).toContainEqual({ cmd: 'gemini_live_set_output_playing', args: { playing: true } });
      expect(sources).toHaveLength(1);

      // Even after onended, mic remains muted during the 350ms acoustic echo guard.
      sources[0].onended?.();
      expect(calls.filter((c) => c.cmd === 'gemini_live_set_output_playing')).toHaveLength(1);

      vi.advanceTimersByTime(360);
      expect(calls).toContainEqual({ cmd: 'gemini_live_set_output_playing', args: { playing: false } });

      // Nothing-heard notices render with soft info tone.
      handlers.get('gemini-live-error')?.({ kind: 'nothing-heard', message: 'Hold shortcut while speaking' });
      expect(get(g.geminiLiveState)).toMatchObject({ error: 'Hold shortcut while speaking', errorTone: 'info' });
    } finally {
      vi.unstubAllGlobals();
    }
  });
});
