import { spawn } from 'node:child_process';
import { createHash } from 'node:crypto';
import { access, mkdir, readFile } from 'node:fs/promises';
import { homedir } from 'node:os';
import { join } from 'node:path';

import { VOICE_ENABLED, demo } from './config.ts';

/**
 * The voice-over, spoken locally by Kokoro.
 *
 * Every line a scene narrates through `actor.say` comes here. Nothing is sent
 * anywhere: Kokoro-82M runs as ONNX on the CPU out of a cache directory, so a
 * corpus can be re-cut on a train and the audio for a private app never
 * leaves the machine. That is the same bargain the rest of the studio keeps
 * with the seeded data, and it would be strange to break it for the
 * narration.
 *
 * Two things make this cheap enough to leave switched on. Lines are cached by
 * the hash of their own text, so the dark pass of a scene speaks nothing at
 * all: it says exactly what the light pass said, and the wav is already
 * there. And synthesis for a whole scene happens in one process, because
 * loading the model costs about four seconds and saying a sentence costs
 * about one.
 *
 * Install is deliberately not automatic. A studio that silently pip-installs a
 * hundred megabytes the first time somebody films is a studio that does
 * something surprising on a metered connection, so a missing model is a clear
 * message and a silent corpus rather than a download.
 */

export type Mark = { at: number; text: string };

/**
 * One cache for every demo on the machine, model and cut lines alike.
 *
 * Per-app was the first instinct and it is wrong twice over. The model is a
 * quarter of a gigabyte and a Python environment beside it, identical for
 * every app that ever films, so a second app would pay the whole install
 * again to get a byte-for-byte copy of the first one's. And the lines cannot
 * collide: `keyFor` hashes the respelt text together with the voice, the
 * language, the speed, both pause lengths and the target loudness, so two
 * apps land on the same file only when they asked the same model to say the
 * same words the same way — in which case the file they want is the file
 * that is already there.
 *
 * `DEMO_VOICE_HOME` still separates them for anyone who wants that.
 */
const VOICE_HOME =
  process.env.DEMO_VOICE_HOME ?? join(homedir(), '.cache/demo-voice');
const PYTHON = join(VOICE_HOME, '.venv/bin/python');
const MODEL = join(VOICE_HOME, 'kokoro.onnx');
const VOICES = join(VOICE_HOME, 'voices.bin');
const LINES = join(VOICE_HOME, 'lines');

/**
 * The voice and pace, adopted from the settings `arikchakma/gpu-time`
 * narrates with, since they were already tuned and sound better than anything
 * arrived at here. `demo.config.ts`'s own `voice` commits an app to a choice;
 * `DEMO_VOICE_NAME`/`DEMO_VOICE_SPEED` override either one at film time, the
 * same way they'd override these defaults with no config at all.
 *
 * Not only the voice. The three numbers under it are the difference between
 * a model reading a string and something worth listening to: a shade under
 * full speed so the sentences are deliberate rather than clipped, and real
 * pauses at sentence and clause boundaries so the prose has joints. Kokoro
 * defaults to 0.25 and 0.1 there, which reads as slightly ponderous over a
 * demo; these are tighter.
 */
const VOICE = process.env.DEMO_VOICE_NAME ?? demo.voice?.name ?? 'af_heart';
const LANG = process.env.DEMO_VOICE_LANG ?? 'en-us';
/**
 * Full speed, with short joints.
 *
 * Not a taste call: a demo line is one sentence about one thing, and at less
 * than full speed the deliberateness reads as a recording rather than someone
 * talking. The pauses matter more than the rate. Kokoro defaults to a quarter
 * of a second between sentences and a tenth between clauses, which is
 * measured against prose being read aloud; over a demo it drags, so both are
 * tighter here. Every number is overridable per app and per run.
 */
const SPEED = Number(process.env.DEMO_VOICE_SPEED ?? demo.voice?.speed ?? 1);
const SENTENCE_PAUSE = Number(process.env.DEMO_VOICE_SENTENCE_PAUSE ?? 0.14);
const CLAUSE_PAUSE = Number(process.env.DEMO_VOICE_CLAUSE_PAUSE ?? 0.06);

/**
 * Every line levelled to the same loudness.
 *
 * Kokoro's output level drifts with the phonemes in a line, so a corpus of
 * raw synthesis has one sentence twice as loud as the next and no amount of
 * mixing hides it. Normalising each line to a target RMS fixes that at the
 * source; the clamp is there so a very quiet line is lifted rather than
 * amplified into its own noise floor.
 */
const TARGET_RMS = Number(process.env.DEMO_VOICE_RMS ?? 0.12);

function exists(path: string): Promise<boolean> {
  return access(path).then(
    () => true,
    () => false
  );
}

/** Whether the local model is installed. Checked once, reported once. */
export async function voiceReady(): Promise<boolean> {
  if (!VOICE_ENABLED) {
    return false;
  }
  const found = await Promise.all([
    exists(PYTHON),
    exists(MODEL),
    exists(VOICES),
  ]);
  return found.every(Boolean);
}

export function voiceMissingMessage(): string {
  return (
    `  no voice-over: ${VOICE_HOME} has no Kokoro model.\n` +
    `  to install it (about 250MB, once):\n` +
    `    uv venv --python 3.12 ${join(VOICE_HOME, '.venv')}\n` +
    `    uv pip install --python ${PYTHON} kokoro-onnx soundfile\n` +
    `    curl -L -o ${MODEL} https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx\n` +
    `    curl -L -o ${VOICES} https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin\n` +
    `  or film silent with DEMO_VOICE=0.\n`
  );
}

/**
 * Includes the model's own size, so swapping the int8 model for fp32 does not
 * quietly serve back lines the old one spoke. Everything that changes how a
 * line sounds belongs in its name.
 */
/**
 * Respellings for words the phonemiser guesses the wrong way.
 *
 * English heteronyms are spelled once and said two ways, and espeak has to
 * pick from the letters alone: it once read "your tasks live right here" as
 * the adjective, rhyming with hive, which is the sort of thing that makes a
 * demo sound machine-made in a single syllable. Handing it a respelling is
 * more reliable than hand-phonemising a whole line, and it keeps scene files
 * readable: the script says "live" and the model is told "liv".
 *
 * `demo.config.ts`'s own `voice.saidAs` is merged on top of this table, so an
 * app adds only the words its own scenes actually mispronounce — never
 * speculative entries for words that might theoretically be ambiguous.
 */
const BUILT_IN_SAID_AS: [RegExp, string][] = [
  // The verb. "Live stream" would want the other reading, so this is scoped
  // to the sense a narration line means, and a line that wants the adjective
  // has to be written around it.
  [/\blive\b/gi, 'liv'],
];

const SAID_AS: [RegExp, string][] = [
  ...BUILT_IN_SAID_AS,
  ...(demo.voice?.saidAs ?? []),
];

function speakable(text: string): string {
  return SAID_AS.reduce((said, [word, as_]) => said.replace(word, as_), text);
}

function keyFor(text: string, model: string): string {
  return createHash('sha1')
    .update(
      `${model}|${VOICE}|${LANG}|${SPEED}|${SENTENCE_PAUSE}|${CLAUSE_PAUSE}|${TARGET_RMS}|${text}`
    )
    .digest('hex')
    .slice(0, 16);
}

const SYNTH = `
import json, sys, os
import numpy as np
import soundfile as sf

try:
    import espeakng_loader
    from phonemizer.backend.espeak.wrapper import EspeakWrapper
    EspeakWrapper.set_library(espeakng_loader.get_library_path())
    EspeakWrapper.set_data_path(espeakng_loader.get_data_path())
except Exception:
    pass

from kokoro_onnx import Kokoro

job = json.load(sys.stdin)
kokoro = Kokoro(job["model"], job["voices"])

for line in job["lines"]:
    if os.path.exists(line["path"]):
        continue
    samples, rate = kokoro.create(
        line["text"],
        voice=job["voice"],
        speed=job["speed"],
        lang=job["lang"],
        sentence_pause=job["sentencePause"],
        clause_pause=job["clausePause"],
    )
    rms = float(np.sqrt(np.mean(samples ** 2)))
    if rms > 0:
        samples = samples * np.clip(job["targetRms"] / rms, 0.55, 1.6)
    sf.write(line["path"], samples, rate)

print(json.dumps({"ok": True}))
`;

function runPython(job: unknown): Promise<void> {
  return new Promise((resolve, reject) => {
    const child = spawn(PYTHON, ['-c', SYNTH], {
      stdio: ['pipe', 'pipe', 'pipe'],
    });
    let stderr = '';
    child.stderr.on('data', (chunk: Buffer) => {
      stderr += chunk.toString();
    });
    child.on('error', reject);
    child.on('close', (code) =>
      code === 0
        ? resolve()
        : reject(new Error(`kokoro exited ${code}\n${stderr.slice(-800)}`))
    );
    child.stdin.end(JSON.stringify(job));
  });
}

export type SpokenLine = Mark & { path: string; seconds: number };

/**
 * Speaks whatever is not already spoken, and answers with every line's file.
 *
 * The duration comes back from the wav header rather than from the model,
 * because it is what the mix needs: a line that runs past the end of the clip
 * has to be noticed, and the only way to notice is to know how long it
 * actually is.
 */
export async function speak(marks: Mark[]): Promise<SpokenLine[]> {
  if (marks.length === 0) {
    return [];
  }

  await mkdir(LINES, { recursive: true });

  const { stat } = await import('node:fs/promises');
  const model = String((await stat(MODEL)).size);

  // Keyed and synthesised on the respelling, so changing the table
  // invalidates only the lines it actually touches.
  const lines = marks.map((mark) => ({
    ...mark,
    said: speakable(mark.text),
    path: join(LINES, `${keyFor(speakable(mark.text), model)}.wav`),
  }));

  const wanted = lines.filter(
    (line, at) => lines.findIndex((other) => other.path === line.path) === at
  );
  const absent = await Promise.all(
    wanted.map(async (line) => ((await exists(line.path)) ? null : line))
  );
  const todo = absent.filter(
    (line): line is (typeof wanted)[number] => line !== null
  );

  if (todo.length > 0) {
    await runPython({
      model: MODEL,
      voices: VOICES,
      voice: VOICE,
      lang: LANG,
      speed: SPEED,
      sentencePause: SENTENCE_PAUSE,
      clausePause: CLAUSE_PAUSE,
      targetRms: TARGET_RMS,
      lines: todo.map((line) => ({ text: line.said, path: line.path })),
    });
  }

  return Promise.all(
    lines.map(async (line) => ({
      ...line,
      seconds: await wavSeconds(line.path),
    }))
  );
}

/**
 * Length straight out of the RIFF header.
 *
 * Shelling out to ffprobe for this would be three seconds of process startup
 * across a corpus to read a number that is at a known byte offset of a file
 * this module just wrote.
 */
async function wavSeconds(path: string): Promise<number> {
  const buffer = await readFile(path);
  const rate = buffer.readUInt32LE(24);
  const bytesPerSecond = buffer.readUInt32LE(28);

  for (let at = 12; at + 8 <= buffer.length;) {
    const id = buffer.toString('ascii', at, at + 4);
    const size = buffer.readUInt32LE(at + 4);
    if (id === 'data') {
      return bytesPerSecond > 0 ? size / bytesPerSecond : size / (rate * 2);
    }
    at += 8 + size + (size % 2);
  }

  return 0;
}

/**
 * The ffmpeg arguments that lay the narration over a silent clip.
 *
 * Each line is delayed to its own mark and the lot are mixed, rather than
 * concatenated with silence between: a mark is a moment in the choreography,
 * and the audio has to land there even if the line before it ran long.
 * `normalize=0` is what keeps `amix` from quietly ducking every line in
 * proportion to how many there are.
 */
export function mixArgs(
  silent: string,
  lines: SpokenLine[],
  out: string,
  videoSeconds: number,
  videoArgs: string[]
): { args: string[]; overrun: number } {
  const overrun = Math.max(
    0,
    ...lines.map((line) => line.at + line.seconds - videoSeconds)
  );

  const inputs = lines.flatMap((line) => ['-i', line.path]);
  const delays = lines
    .map(
      (line, at) =>
        `[${at + 1}:a]adelay=${Math.max(0, Math.round(line.at * 1000))}:all=1[d${at}]`
    )
    .join(';');
  const heads = lines.map((_, at) => `[d${at}]`).join('');
  const mix =
    lines.length === 1
      ? `${heads}anull[spoken]`
      : `${heads}amix=inputs=${lines.length}:normalize=0:dropout_transition=0[spoken]`;

  /**
   * A held last frame, rather than a last word cut in half — or bleeding
   * into the next clip.
   *
   * `actor.say` already holds long enough for its own line in the ordinary
   * case, so this is the safety net for the closing line of a scene, where
   * there is nothing after it to hold. Padding means filtering the video,
   * and filtering means the stream cannot be copied, so the encoder arguments
   * come back in here from the recorder rather than being guessed at: it is
   * the same encode the silent clip had, run once more with a longer tail.
   *
   * Any overrun pads, not just one past a tolerance. `stitchTour` concatenates
   * every clip in a theme with a stream copy, so a clip whose audio track runs
   * even slightly longer than its video track has narration that keeps
   * playing after the concat demuxer has already moved on to the next file's
   * timeline — heard as the tail of one scene's line overlapping the start of
   * the next one's. Padding the video to match the audio, however small the
   * gap, is what keeps a clip's own duration honest for concatenation.
   *
   * Without padding the video is copied untouched, which is the common path
   * and costs nothing.
   */
  const padding = overrun > 0;
  const video = padding
    ? [
        '-filter_complex',
        `${delays};${mix};[0:v]tpad=stop_mode=clone:stop_duration=${overrun.toFixed(3)}[v]`,
        '-map',
        '[v]',
        ...videoArgs,
      ]
    : ['-filter_complex', `${delays};${mix}`, '-map', '0:v', '-c:v', 'copy'];

  return {
    overrun,
    args: [
      '-y',
      '-hide_banner',
      '-loglevel',
      'error',
      '-i',
      silent,
      ...inputs,
      ...video,
      '-map',
      '[spoken]',
      '-c:a',
      'aac',
      '-b:a',
      '160k',
      '-ar',
      '48000',
      '-movflags',
      '+faststart',
      out,
    ],
  };
}

/**
 * Roughly how long Kokoro will take to say something, before it has said it.
 *
 * Needed because the hold has to be chosen while the scene is running and the
 * wav does not exist until afterwards. Measured at three words a second for
 * this voice at speed 1, which is close enough that the padding path above
 * almost never runs: the point is only that the pointer should not race ahead
 * of the narration, and half a second either way does not change that.
 */
export function spokenFor(text: string, minimum = 0): number {
  const words = text.trim().split(/\s+/).filter(Boolean).length;
  const seconds = words / (3.7 * SPEED) + 0.4;
  return Math.max(minimum, Math.round(seconds * 1000));
}
