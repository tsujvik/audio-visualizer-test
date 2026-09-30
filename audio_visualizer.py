"""
audio visualizer: MICROPHONE, AUDIO FILE, or a HAND-PLAYED REVERB SYNTH.

Install:  py -m pip install numpy sounddevice soundfile pygame-ce
Hand synth also needs Python 3.12 plus opencv-python and mediapipe==0.10.21:
          py -3.12 -m pip install numpy sounddevice soundfile pygame-ce opencv-python mediapipe==0.10.21
Run:      py audio_visualizer.py        (use  py -3.12 audio_visualizer.py  for hand synth)

Menu:      click a row, or press 1 (microphone) / 2 (audio file) / 3 (hand synth) /
           4 (rhythm game), ESC to quit
Visuals:   ESC returns to the menu
File mode: SPACE = pause/resume, LEFT/RIGHT = seek 5 seconds
           Supported files: wav, mp3, flac, ogg

Hand synth (a big, dreamy, reverb-soaked pad):
    RIGHT hand: height = pitch (snapped to a scale), pinch open/closed = brightness
    LEFT hand:  height = volume,                      pinch open/closed = reverb amount
    (pinch = distance between thumb tip and index fingertip)
    S = change scale, LEFT/RIGHT = change key, SPACE = mute

Rhythm game: pick any song, it builds a chart from the music itself, and you play it.
    D F J K = hit the four lanes (tap short notes, HOLD the key down on long ones until
    the tail ends), UP/DOWN = note speed, LEFT/RIGHT = nudge timing,
    ESC = pause (ENTER resume, R restart, Q quit)
"""

import bisect
import faulthandler
import math
import os
import random
import threading
import time

import numpy as np
import pygame
import sounddevice as sd
import soundfile as sf

faulthandler.enable()      # if Python ever crashes hard, print where (in the terminal)

# Hand synth needs extra packages. If they're missing (or you're on a Python
# version MediaPipe doesn't support), the mic and file modes still work fine.
try:
    import cv2
    import mediapipe as mp
    mp_hands = mp.solutions.hands
    HAND_ERROR = None
except Exception as err:
    cv2 = mp = mp_hands = None
    HAND_ERROR = f"{type(err).__name__}: {err}"

# ---------------------------------------------------------------- settings --
BLOCK_SIZE = 2048        # samples analysed per frame (bigger = better bass detail)
MIC_SAMPLE_RATE = 44100
INPUT_DEVICE = None      # None = system default mic, or a device number from sd.query_devices()
START_TIMEOUT = 10       # seconds to wait for audio/camera to open before giving up
NUM_BARS = 48
MIN_FREQ = 40
MAX_FREQ = 12000
TILT_DB = 12             # file mode only: boost highs by up to this many dB
FALL_SPEED = 0.03        # how fast bars drop per frame

WIDTH, HEIGHT = 1120, 660
FPS = 60
SEEK_SECONDS = 5

# --- hand synth settings ---
SYNTH_SAMPLE_RATE = 44100
AUDIO_BLOCK = 1024           # the reverb works in blocks this size (about 23 ms)
MASTER_VOLUME = 0.22         # overall loudness (0..1)
NUM_OCTAVES = 3              # pitch range of the right hand
DETUNE_CENTS = [-13, -6.5, 0, 6.5, 13]   # 5 slightly detuned voices = wide, lush pad
NUM_HARMONICS = 20           # how many overtones each voice can have
SUB_LEVEL = 0.4              # deep sine one octave below
ATTACK = 0.06                # how fast notes swell in (per block, smaller = slower)
RELEASE = 0.03               # how fast notes fade out
GLIDE = 0.2                  # how fast the pitch slides between notes
REVERB_SECONDS = 3.0         # length of the reverb tail (lower it if audio crackles)
CAMERA_INDEX = 0
# The usable band of the camera image (0 = top, 1 = bottom). Keeping the
# extremes out of range means you don't have to reach the very edge of the frame.
Y_TOP, Y_BOTTOM = 0.12, 0.88
HYSTERESIS = 0.15            # stops notes flickering when your hand sits on a boundary
DETUNE_RATIOS = 2 ** (np.array(DETUNE_CENTS) / 1200)
HARMONICS = np.arange(1, NUM_HARMONICS + 1)

SCALES = {
    "Minor pentatonic": [0, 3, 5, 7, 10],
    "Major pentatonic": [0, 2, 4, 7, 9],
    "Major": [0, 2, 4, 5, 7, 9, 11],
    "Natural minor": [0, 2, 3, 5, 7, 8, 10],
    "Chromatic": list(range(12)),
    "Free glide": None,      # no snapping, like a real theremin
}
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# ------------------------------------------------------------------- theme --
# Light editorial look: paper white, dot grid, sky-blue panels, black ink, and
# one small red accent.
PAPER = (243, 245, 248)
INK = (14, 16, 24)
BLUE = (28, 108, 236)
SKY_TOP = (66, 146, 244)
SKY_BOTTOM = (214, 233, 252)
GREY = (146, 154, 168)
DOT = (198, 204, 214)
RED = (236, 64, 84)
WHITE = (255, 255, 255)

PANEL = pygame.Rect(40, 96, WIDTH - 80, HEIGHT - 96 - 60)   # main visual panel
HAND_PANEL = pygame.Rect(340, 96, WIDTH - 340 - 40, HEIGHT - 96 - 60)
MENU_PANEL = pygame.Rect(580, 96, WIDTH - 580 - 40, HEIGHT - 96 - 60)

FONT_TITLE = FONT_ROW = FONT_UI = FONT_MONO = FONT_TINY = FONT_HUGE = FONT_BIG = None


def init_theme():
    """Load fonts (call after pygame.init). Falls back to pygame's default font."""
    global FONT_TITLE, FONT_ROW, FONT_UI, FONT_MONO, FONT_TINY, FONT_HUGE, FONT_BIG
    sans = "helveticaneue,helvetica,arial,segoeui"
    mono = "consolas,menlo,couriernew,monospace"
    FONT_TITLE = pygame.font.SysFont(sans, 52, bold=True)
    FONT_ROW = pygame.font.SysFont(sans, 40, bold=True)
    FONT_UI = pygame.font.SysFont(sans, 22, bold=True)
    FONT_HUGE = pygame.font.SysFont(sans, 104, bold=True)
    FONT_BIG = pygame.font.SysFont(sans, 72, bold=True)
    FONT_MONO = pygame.font.SysFont(mono, 15)
    FONT_TINY = pygame.font.SysFont(mono, 12)


_cache = {}
_text_cache = {}


def lerp_color(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def render_text(font, s, color):
    key = (id(font), s, color)
    surf = _text_cache.get(key)
    if surf is None:
        if len(_text_cache) > 600:
            _text_cache.clear()
        surf = font.render(s, True, color)
        _text_cache[key] = surf
    return surf


def draw_text(screen, font, s, pos, color=INK, anchor="topleft"):
    surf = render_text(font, s, color)
    rect = surf.get_rect(**{anchor: pos})
    screen.blit(surf, rect)
    return rect


def draw_label(screen, font, s, pos, color=INK, bg=PAPER, anchor="topleft", pad=5):
    """Text on a little paper sticker, so it stays readable over anything."""
    surf = render_text(font, s, color)
    rect = surf.get_rect(**{anchor: pos})
    pygame.draw.rect(screen, bg, rect.inflate(pad * 2, pad))
    screen.blit(surf, rect)
    return rect


def draw_cross(screen, x, y, size=6, color=INK):
    pygame.draw.line(screen, color, (x - size, y), (x + size, y), 1)
    pygame.draw.line(screen, color, (x, y - size), (x, y + size), 1)


def get_paper():
    """Off-white background with a faint dot grid (built once)."""
    if "paper" not in _cache:
        surf = pygame.Surface((WIDTH, HEIGHT))
        surf.fill(PAPER)
        for x in range(20, WIDTH, 20):
            for y in range(20, HEIGHT, 20):
                surf.fill(DOT, (x, y, 2, 2))
        _cache["paper"] = surf
    return _cache["paper"]


def get_sky(size):
    """Blue sky gradient with soft clouds (built once per size)."""
    key = ("sky", tuple(size))
    if key not in _cache:
        w, h = size
        surf = pygame.Surface((w, h))
        for y in range(h):
            pygame.draw.line(surf, lerp_color(SKY_TOP, SKY_BOTTOM, y / max(1, h - 1)),
                             (0, y), (w, y))
        # Draw blobs on a tiny surface, then scale up: the smoothing makes clouds.
        small = pygame.Surface((max(2, w // 10), max(2, h // 10)), pygame.SRCALPHA)
        rng = random.Random(3)
        sw, sh = small.get_size()
        for _ in range(26):
            pygame.draw.circle(small, (255, 255, 255, rng.randrange(60, 150)),
                               (rng.randrange(0, sw), rng.randrange(sh // 5, sh)),
                               rng.randrange(3, 9))
        surf.blit(pygame.transform.smoothscale(small, (w, h)), (0, 0))
        _cache[key] = surf
    return _cache[key]


def draw_panel(screen, rect):
    screen.blit(get_sky(rect.size), rect.topleft)
    pygame.draw.rect(screen, INK, rect, 1)
    for cx, cy in ((rect.left - 14, rect.top - 14), (rect.right + 14, rect.top - 14),
                   (rect.left - 14, rect.bottom + 14), (rect.right + 14, rect.bottom + 14)):
        draw_cross(screen, cx, cy)


def draw_chrome(screen, title, info_lines, footer_left="", footer_right=""):
    """The frame every screen shares: paper, title, ruler, footer."""
    screen.blit(get_paper(), (0, 0))

    # black strip down the left edge, like a film-leader marker
    pygame.draw.rect(screen, INK, (0, 200, 12, 210))
    for k in range(6):
        pygame.draw.rect(screen, INK, (18, 206 + k * 9, 7, 3))
    pygame.draw.polygon(screen, INK, [(18, 430), (18, 442), (28, 436)])
    pygame.draw.polygon(screen, INK, [(18, 448), (18, 460), (28, 454)])

    draw_text(screen, FONT_TITLE, title, (40, 20))
    for i, line in enumerate(info_lines):
        draw_text(screen, FONT_TINY, line, (WIDTH - 40, 24 + i * 15), INK, "topright")

    # ruler under the title
    y = 82
    pygame.draw.line(screen, INK, (40, y), (WIDTH - 40, y), 1)
    for x in range(40, WIDTH - 39, 10):
        pygame.draw.line(screen, INK, (x, y), (x, y + (7 if (x - 40) % 50 == 0 else 3)), 1)
    pygame.draw.rect(screen, BLUE, (40, y - 3, 76, 6))

    draw_text(screen, FONT_TINY, footer_left, (40, HEIGHT - 34), INK)
    draw_text(screen, FONT_TINY, footer_right, (WIDTH - 40, HEIGHT - 34), INK, "topright")


# ---------------------------------------------------------------- analysis --
WINDOW = np.hanning(BLOCK_SIZE).astype(np.float32)

def build_bar_bins(sample_rate):
    """Work out which FFT bins belong to each bar (depends on the sample rate)."""
    freqs = np.fft.rfftfreq(BLOCK_SIZE, d=1.0 / sample_rate)
    edges = np.logspace(np.log10(MIN_FREQ), np.log10(MAX_FREQ), NUM_BARS + 1)
    bins = []
    for i in range(NUM_BARS):
        lo = np.searchsorted(freqs, edges[i])
        hi = max(lo + 1, np.searchsorted(freqs, edges[i + 1]))
        bins.append((lo, hi))
    return bins


def compute_bars(samples, bar_bins, db_floor, db_ceil, tilt):
    """Turn a chunk of samples into NUM_BARS values between 0.0 and 1.0."""
    spectrum = np.abs(np.fft.rfft(samples * WINDOW)) / BLOCK_SIZE
    levels = np.array([spectrum[lo:hi].max() for lo, hi in bar_bins])
    db = 20 * np.log10(levels + 1e-9) + tilt
    return np.clip((db - db_floor) / (db_ceil - db_floor), 0.0, 1.0)


def pad_to_block(segment):
    if len(segment) < BLOCK_SIZE:
        segment = np.pad(segment, (0, BLOCK_SIZE - len(segment)))
    return segment


# ----------------------------------------------------------- audio sources --
# Every source offers the same interface, so the visualizer doesn't care which
# one it's drawing:  start(), stop(), get_segment(), handle_key(key),
# plus the attributes name, hint, sample_rate, finished, db_floor, db_ceil, tilt.
# The Source base class adds optional hooks that only some sources use.

class Source:
    finished = False
    panel = PANEL                # where the sky, bars and camera are drawn
    bar_max_frac = 0.92          # how much of the panel the tallest bar may use
    show_glitch = True           # the little blue "glitch" bars in the corner

    def update(self):
        """Called once per frame before drawing."""

    def draw_background(self, screen, panel):
        """Draw under the bars (the hand synth draws its camera here)."""

    def draw_overlay(self, screen, panel):
        """Draw extra things on top of the bars."""

    def handle_key(self, key):
        pass


class MicSource(Source):
    """Live audio from the default microphone."""

    name = "microphone"
    hint = "ESC = MENU"
    db_floor = -90       # mic signals are quieter, so the range is more sensitive
    db_ceil = -40
    tilt = np.zeros(NUM_BARS)   # no frequency boost for the mic
    sample_rate = MIC_SAMPLE_RATE
    finished = False

    def __init__(self):
        self.buffer = np.zeros(BLOCK_SIZE, dtype=np.float32)
        self.stream = None

    def _callback(self, indata, frames, time_info, status):
        self.buffer = indata[:, 0].copy()

    def start(self):
        self.stream = sd.InputStream(device=INPUT_DEVICE, channels=1,
                                     samplerate=self.sample_rate,
                                     blocksize=BLOCK_SIZE, callback=self._callback)
        self.stream.start()

    def stop(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()

    def get_segment(self):
        return pad_to_block(self.buffer)

    def draw_overlay(self, screen, panel):
        # blinking red dot + LIVE
        rect = draw_label(screen, FONT_TINY, "LIVE INPUT", (panel.right - 18, panel.top + 16),
                          INK, PAPER, "topright")
        if (pygame.time.get_ticks() // 600) % 2 == 0:
            pygame.draw.circle(screen, RED, (rect.left - 12, rect.centery), 4)


class FileSource(Source):
    """Plays an audio file and hands the visualizer whatever is playing now."""

    hint = "SPACE = PAUSE     LEFT / RIGHT = SEEK 5S     ESC = MENU"
    db_floor = -80
    db_ceil = -20
    tilt = np.linspace(0, TILT_DB, NUM_BARS)

    def __init__(self, path):
        base = os.path.splitext(os.path.basename(path))[0].lower()
        self.name = base if len(base) <= 26 else base[:24] + ".."
        self.data, self.sample_rate = sf.read(path, dtype="float32", always_2d=True)
        self.mono = self.data.mean(axis=1)    # merged channels, for analysis
        self.pos = 0                          # index of the next sample to play
        self.paused = False
        self.finished = False
        self.lag = 0
        self.stream = None

    def _callback(self, outdata, frames, time_info, status):
        """Runs on a background thread whenever the speakers need more audio."""
        outdata.fill(0)
        if self.paused:
            return
        chunk = self.data[self.pos:self.pos + frames]
        outdata[:len(chunk)] = chunk
        self.pos += len(chunk)
        if len(chunk) < frames:               # ran out of song
            self.finished = True
            raise sd.CallbackStop

    def start(self):
        self.stream = sd.OutputStream(samplerate=self.sample_rate,
                                      channels=self.data.shape[1],
                                      callback=self._callback)
        self.stream.start()
        self.lag = int(self.stream.latency * self.sample_rate)

    def stop(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()

    def get_segment(self):
        if self.paused:
            return np.zeros(BLOCK_SIZE, dtype=np.float32)
        start = max(0, self.pos - self.lag)
        return pad_to_block(self.mono[start:start + BLOCK_SIZE])

    def handle_key(self, key):
        if key == pygame.K_SPACE:
            self.paused = not self.paused
        elif key in (pygame.K_LEFT, pygame.K_RIGHT):
            direction = 1 if key == pygame.K_RIGHT else -1
            new_pos = self.pos + direction * SEEK_SECONDS * self.sample_rate
            self.pos = int(np.clip(new_pos, 0, len(self.data) - 1))

    def draw_overlay(self, screen, panel):
        total = len(self.data)
        frac = self.pos / max(1, total)
        elapsed = int(self.pos / self.sample_rate)
        length = int(total / self.sample_rate)
        stamp = f"{elapsed // 60:02d}:{elapsed % 60:02d} / {length // 60:02d}:{length % 60:02d}"
        draw_label(screen, FONT_TINY, stamp, (panel.right - 18, panel.top + 16),
                   INK, PAPER, "topright")
        if self.paused:
            draw_label(screen, FONT_TINY, "PAUSED", (panel.right - 18, panel.top + 42),
                       WHITE, RED, "topright")
        # thin progress line along the top of the panel
        x0, x1, y = panel.left + 18, panel.right - 18, panel.top + 8
        pygame.draw.line(screen, INK, (x0, y), (x1, y), 1)
        px = int(x0 + (x1 - x0) * frac)
        pygame.draw.line(screen, BLUE, (x0, y), (px, y), 3)
        pygame.draw.rect(screen, INK, (px - 3, y - 4, 6, 8))


# -------------------------------------------------------- hand synth helpers --
def clip01(x):
    return max(0.0, min(1.0, x))


def build_notes(root, degrees):
    """All the MIDI notes of a scale across NUM_OCTAVES, plus the top root."""
    notes = [root + 12 * octave + d for octave in range(NUM_OCTAVES) for d in degrees]
    notes.append(root + 12 * NUM_OCTAVES)
    return notes


def pick_index(height, count, current):
    """Turn hand height (0..1) into a note index, sticking to the current
    note until the hand moves clearly into the next one."""
    raw = height * count
    if current is not None and abs(raw - (current + 0.5)) < 0.5 + HYSTERESIS:
        return current
    return int(min(raw, count - 1))


def midi_to_freq(midi):
    return 440.0 * 2 ** ((midi - 69) / 12)


def note_name(midi):
    m = int(round(midi))
    return f"{NOTE_NAMES[m % 12]}{m // 12 - 1}"


def hand_info(landmarks):
    """Pull the numbers we care about out of MediaPipe's 21 landmarks."""
    pts = [(lm.x, lm.y) for lm in landmarks.landmark]
    palm_y = pts[9][1]                                    # middle-finger knuckle
    height = 1 - clip01((palm_y - Y_TOP) / (Y_BOTTOM - Y_TOP))
    hand_size = math.dist(pts[0], pts[9]) or 1e-6         # wrist -> knuckle
    pinch = math.dist(pts[4], pts[8]) / hand_size         # thumb tip -> index tip
    openness = clip01((pinch - 0.2) / 0.8)                # 0 = pinched, 1 = wide open
    return {"pts": pts, "height": height, "openness": openness}


def draw_meter(screen, label, value, x, y, w, color=BLUE):
    """A thin editorial slider: label, percentage, line with a square handle."""
    draw_text(screen, FONT_TINY, label, (x, y), INK)
    draw_text(screen, FONT_TINY, f"{int(value * 100):3d}%", (x + w, y), INK, "topright")
    ly = y + 26
    pygame.draw.line(screen, GREY, (x, ly), (x + w, ly), 1)
    end = int(x + w * value)
    pygame.draw.line(screen, color, (x, ly), (end, ly), 4)
    pygame.draw.rect(screen, INK, (end - 4, ly - 6, 8, 12))


# ------------------------------------------------------------------ reverb --
class Reverb:
    """A big, dark, stereo reverb. The sound is convolved with a synthetic room
    (a burst of noise that fades away over a few seconds), done in the frequency
    domain block by block so it's fast enough to run live."""

    def __init__(self, sample_rate, block, seconds):
        self.block = block
        n = int(seconds * sample_rate)
        self.parts = -(-n // block)                     # number of blocks in the room
        length = self.parts * block
        self.H = []
        for seed in (7, 11):                            # a different room for each ear
            ir = self._make_ir(length, sample_rate, seed)
            blocks = ir.reshape(self.parts, block)
            self.H.append(np.fft.rfft(blocks, n=2 * block, axis=1).astype(np.complex64))
        # Remembers the spectrum of the most recent input blocks.
        self.fdl = np.zeros((self.parts, block + 1), dtype=np.complex64)
        self.idx = 0
        self.prev_in = np.zeros(block, dtype=np.float32)

    @staticmethod
    def _make_ir(n, sr, seed):
        """The room's 'fingerprint': noise that fades out, darker as it goes."""
        rng = np.random.default_rng(seed)
        t = np.arange(n) / sr
        seconds = n / sr
        noise = rng.standard_normal(n)
        freqs = np.fft.rfftfreq(n, 1 / sr)
        dark = np.fft.irfft(np.fft.rfft(noise) / (1 + (freqs / 2500.0) ** 2), n)
        dark *= noise.std() / dark.std()
        # Dark noise lingers for the whole tail, bright noise dies quickly.
        ir = dark * np.exp(-t / (seconds / 6.9)) + 0.5 * noise * np.exp(-t / 0.12)
        ir *= 1 - np.exp(-t / 0.02)                     # soft onset
        pre_delay = int(0.015 * sr)
        ir = np.roll(ir, pre_delay)
        ir[:pre_delay] = 0
        return ir / np.sqrt(np.sum(ir ** 2))            # keep the loudness unchanged

    def process(self, x):
        """Feed one block of mono audio, get back (left, right) reverb."""
        frame = np.concatenate((self.prev_in, x))
        self.prev_in = x.copy()
        self.idx = (self.idx + 1) % self.parts
        self.fdl[self.idx] = np.fft.rfft(frame)
        idx = self.idx
        outs = []
        for H in self.H:
            # Each stored block is multiplied by the matching slice of the room.
            acc = (self.fdl[idx::-1] * H[:idx + 1]).sum(axis=0)
            if idx + 1 < self.parts:
                acc += (self.fdl[:idx:-1] * H[idx + 1:]).sum(axis=0)
            y = np.fft.irfft(acc, n=2 * self.block)[self.block:]
            outs.append(y.astype(np.float32))
        return outs[0], outs[1]


# ----------------------------------------------------------- hand synth source --
class HandSynthSource(Source):
    """A synthesizer you play with your hands. It makes the sound itself, and
    the visualizer draws the spectrum of what it's producing."""

    name = "hand synth"
    hint = "S = SCALE     < / > = KEY     SPACE = MUTE     ESC = MENU"
    db_floor = -80
    db_ceil = -18
    tilt = np.zeros(NUM_BARS)
    sample_rate = SYNTH_SAMPLE_RATE
    panel = HAND_PANEL
    bar_max_frac = 0.30
    show_glitch = False

    def __init__(self):
        if HAND_ERROR:
            print("\nHand synth isn't available:", HAND_ERROR)
            print("It needs Python 3.12 and:  py -3.12 -m pip install opencv-python mediapipe==0.10.21")
            print("(If MediaPipe says it has no 'solutions', that's the fix.)\n")
            raise RuntimeError("needs mediapipe + opencv (run with Python 3.12)")

        # audio state (the audio thread reads the target_* values)
        self.freq = 220.0            # current (smoothed) frequency
        self.amp = 0.0               # current (smoothed) volume
        self.target_freq = 220.0
        self.target_amp = 0.0
        self.brightness = 0.5        # 0 = soft and dark, 1 = bright and buzzy
        self.reverb_mix = 0.6        # 0 = dry, 1 = all reverb
        rng = np.random.default_rng()
        self.phases = rng.uniform(0, 2 * np.pi, len(DETUNE_RATIOS))   # one per voice
        self.sub_phase = 0.0
        self.reverb = Reverb(self.sample_rate, AUDIO_BLOCK, REVERB_SECONDS)
        self.history = np.zeros(BLOCK_SIZE, dtype=np.float32)   # feeds the visualizer
        self.stream = None

        # vision state
        self.cap = None
        self.hands = None
        self.camera = None           # latest camera picture, as a pygame surface
        self.wash = None             # white veil that gives the camera a high-key look
        self.cam_x, self.cam_y = self.panel.x, self.panel.y
        self.cam_w, self.cam_h = self.panel.w, self.panel.h
        self.found = {}              # {'Left'/'Right': hand info}

        # musical state
        self.scale_i = 0
        self.root = 48               # C3
        self.note_idx = None
        self.muted = False
        self.vals = {"pitch": 0.5, "vol": 0.6, "bright": 0.5, "verb": 0.55}
        self.notes = []
        self.midi = 48
        self.playing = False

    # ---- audio ----------------------------------------------------------
    def _callback(self, outdata, frames, time_info, status):
        if frames != AUDIO_BLOCK:            # the reverb needs fixed-size blocks
            outdata.fill(0)
            return
        sr = self.sample_rate

        # Glide toward the targets a little each block so changes never click.
        f_end = self.freq * (self.target_freq / self.freq) ** GLIDE
        freq = np.linspace(self.freq, f_end, frames)
        self.freq = f_end

        # Slow swell in, slower fade out: this is what makes it feel like a pad.
        rate = ATTACK if self.target_amp > self.amp else RELEASE
        a_end = self.amp + (self.target_amp - self.amp) * rate
        amp = np.linspace(self.amp, a_end, frames)
        self.amp = a_end

        # Five slightly detuned voices. Accumulating phase keeps each wave
        # continuous when the pitch changes.
        inc = 2 * np.pi * DETUNE_RATIOS[:, None] * freq[None, :] / sr
        phase = self.phases[:, None] + np.cumsum(inc, axis=1)          # (voices, frames)
        self.phases = phase[:, -1] % (2 * np.pi)

        # Brightness = how quickly the overtones fade out (like a low-pass filter).
        # Overtones that would go past the speaker's limit are switched off.
        w = HARMONICS ** (-(1.0 + 3.0 * (1.0 - self.brightness)))
        n_ok = max(1, int(0.45 * sr / (freq.max() * DETUNE_RATIOS.max())))
        w[n_ok:] = 0.0
        rms = math.sqrt(np.sum(w ** 2) / 2)

        harm = np.sin(HARMONICS[None, :, None] * phase[:, None, :])    # (voices, harmonics, frames)
        wave = (w @ harm.sum(axis=0)) / (rms * math.sqrt(len(DETUNE_RATIOS)))

        # A deep sine one octave down for weight.
        sub_phase = self.sub_phase + np.cumsum(2 * np.pi * (freq * 0.5) / sr)
        self.sub_phase = sub_phase[-1] % (2 * np.pi)
        wave = wave + SUB_LEVEL * 1.4 * np.sin(sub_phase)

        dry = (wave * amp * MASTER_VOLUME).astype(np.float32)

        # The reverb keeps running even when the note is silent, so the tail
        # rings out after you take your hand away.
        wet_l, wet_r = self.reverb.process(dry)
        m = self.reverb_mix
        left = (1 - m) * dry + m * wet_l
        right = (1 - m) * dry + m * wet_r
        outdata[:, 0] = np.tanh(left)
        outdata[:, 1] = np.tanh(right)

        mono = ((left + right) * 0.5).astype(np.float32)
        self.history = np.concatenate((self.history[frames:], mono))

    def start(self):
        backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(CAMERA_INDEX, backend)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(CAMERA_INDEX)
        if not self.cap.isOpened():
            raise RuntimeError("couldn't open the webcam (is another app using it?)")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.hands = mp_hands.Hands(max_num_hands=2, model_complexity=0,
                                    min_detection_confidence=0.6,
                                    min_tracking_confidence=0.5)

        self.stream = sd.OutputStream(samplerate=self.sample_rate, channels=2,
                                      blocksize=AUDIO_BLOCK, latency="low",
                                      callback=self._callback)
        self.stream.start()

    def stop(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()
        if self.hands:
            self.hands.close()
        if self.cap:
            self.cap.release()

    def get_segment(self):
        return self.history

    # ---- vision ---------------------------------------------------------
    def _track(self, frame):
        """Find the hands in a camera frame. Also builds the picture to show."""
        frame = cv2.flip(frame, 1)                        # mirror, like a selfie
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.hands.process(rgb)

        found = {}
        if results.multi_hand_landmarks:
            for lms, handed in zip(results.multi_hand_landmarks,
                                   results.multi_handedness):
                label = handed.classification[0].label    # "Left" or "Right"
                found[label] = hand_info(lms)

        # Fit the picture inside the panel, keeping its shape.
        h, w = rgb.shape[:2]
        scale = min(self.panel.w / w, self.panel.h / h)
        self.cam_w, self.cam_h = int(w * scale), int(h * scale)
        self.cam_x = self.panel.x + (self.panel.w - self.cam_w) // 2
        self.cam_y = self.panel.y + (self.panel.h - self.cam_h) // 2
        surface = pygame.image.frombuffer(rgb.tobytes(), (w, h), "RGB")
        self.camera = pygame.transform.smoothscale(surface, (self.cam_w, self.cam_h))
        if (self.wash is None or self.wash.get_width() != self.cam_w
                or self.wash.get_height() != self.cam_h):
            self.wash = pygame.Surface((self.cam_w, self.cam_h))
            self.wash.fill(WHITE)
            self.wash.set_alpha(95)
        return found

    # ---- per-frame update -------------------------------------------------
    def update(self):
        ok, frame = self.cap.read()
        if ok:
            self.found = self._track(frame)
        else:
            self.found = {}
            self.camera = None

        right, left = self.found.get("Right"), self.found.get("Left")
        vals = self.vals

        # Hands -> synth controls (smoothed to remove jitter).
        targets = {
            "pitch": right["height"] if right else vals["pitch"],
            "bright": right["openness"] if right else vals["bright"],
            "vol": 0.1 + 0.9 * left["height"] if left else 0.6,
            "verb": left["openness"] if left else 0.55,
        }
        for key, target in targets.items():
            vals[key] += (target - vals[key]) * 0.35

        degrees = SCALES[list(SCALES)[self.scale_i]]
        if degrees is None:                               # free glide
            self.midi = self.root + vals["pitch"] * 12 * NUM_OCTAVES
            self.notes = []
            self.note_idx = None
        else:
            self.notes = build_notes(self.root, degrees)
            if self.note_idx is not None and self.note_idx >= len(self.notes):
                self.note_idx = None
            self.note_idx = pick_index(vals["pitch"], len(self.notes), self.note_idx)
            self.midi = self.notes[self.note_idx]

        self.playing = bool(right) and not self.muted
        self.target_freq = midi_to_freq(self.midi)
        self.target_amp = vals["vol"] if self.playing else 0.0
        self.brightness = vals["bright"]
        self.reverb_mix = 0.25 + 0.7 * vals["verb"]

    def handle_key(self, key):
        if key == pygame.K_SPACE:
            self.muted = not self.muted
        elif key == pygame.K_s:
            self.scale_i = (self.scale_i + 1) % len(SCALES)
            self.note_idx = None
        elif key == pygame.K_RIGHT:
            self.root = min(60, self.root + 1)
            self.note_idx = None
        elif key == pygame.K_LEFT:
            self.root = max(36, self.root - 1)
            self.note_idx = None

    # ---- drawing ----------------------------------------------------------
    def draw_background(self, screen, panel):
        if self.camera is not None:
            screen.blit(self.camera, (self.cam_x, self.cam_y))
            screen.blit(self.wash, (self.cam_x, self.cam_y))

    def _draw_hand(self, screen, pts, label):
        color = BLUE if label == "Right" else INK
        px = [(self.cam_x + int(x * self.cam_w), self.cam_y + int(y * self.cam_h))
              for x, y in pts]
        for a, b in mp_hands.HAND_CONNECTIONS:
            pygame.draw.line(screen, color, px[a], px[b], 2)
        for i, p in enumerate(px):                        # hollow joints, like line art
            r = 5 if i in (4, 8, 12, 16, 20) else 3
            pygame.draw.circle(screen, PAPER, p, r)
            pygame.draw.circle(screen, color, p, r, 2)
        pygame.draw.line(screen, RED, px[4], px[8], 3)    # the pinch
        tag = "R / PITCH + BRIGHTNESS" if label == "Right" else "L / VOLUME + REVERB"
        draw_label(screen, FONT_TINY, tag, (px[0][0], px[0][1] + 14), color, PAPER, "midtop")

    def _draw_ladder(self, screen, panel):
        """Note markers along the right edge so you can see where each note lives."""
        count = len(self.notes)
        x0 = panel.right - 74
        for i in range(count):
            h = (i + 0.5) / count
            y = int(self.cam_y + self.cam_h * (Y_TOP + (Y_BOTTOM - Y_TOP) * (1 - h)))
            active = i == self.note_idx
            pygame.draw.line(screen, BLUE if active else INK,
                             (x0, y), (panel.right - 18, y), 4 if active else 1)
            if active:
                draw_label(screen, FONT_UI, note_name(self.midi), (x0 - 8, y), INK, PAPER,
                           "midright")

    def draw_overlay(self, screen, panel):
        for label, info in self.found.items():
            self._draw_hand(screen, info["pts"], label)
        if self.notes:
            self._draw_ladder(screen, panel)

        playing = self.playing
        # big note name, left column
        note = (note_name(self.midi) + ".") if playing else "--."
        draw_text(screen, FONT_HUGE, note, (34, 96), INK if playing else GREY)
        freq_text = f"{midi_to_freq(self.midi):7.1f} HZ" if playing else "SILENT"
        draw_text(screen, FONT_MONO, freq_text, (40, 210), INK)
        pygame.draw.line(screen, INK, (40, 244), (300, 244), 1)

        draw_meter(screen, "VOLUME / LEFT HEIGHT", self.vals["vol"], 40, 264, 260)
        draw_meter(screen, "BRIGHTNESS / RIGHT PINCH", self.vals["bright"], 40, 320, 260)
        draw_meter(screen, "REVERB / LEFT PINCH", self.vals["verb"], 40, 376, 260)

        pygame.draw.line(screen, INK, (40, 438), (300, 438), 1)
        scale = list(SCALES)[self.scale_i].lower()
        draw_text(screen, FONT_UI, f"{scale} / {NOTE_NAMES[self.root % 12].lower()}", (40, 450))
        for i, line in enumerate(("RIGHT HAND = PITCH", "LEFT HAND = VOLUME",
                                  "PINCH = THUMB + INDEX")):
            draw_text(screen, FONT_TINY, line, (40, 492 + i * 17), INK)

        if not self.found.get("Right"):
            draw_label(screen, FONT_MONO, "RAISE YOUR RIGHT HAND TO PLAY",
                       (panel.centerx, panel.top + 60), INK, PAPER, "midtop", pad=14)
        if self.muted:
            draw_label(screen, FONT_MONO, "MUTED", (panel.centerx, panel.top + 100),
                       WHITE, RED, "midtop", pad=14)


# ----------------------------------------------------------------- drawing --
SEG_H, SEG_GAP = 7, 3           # one bar is a stack of little blocks
_STEP = SEG_H + SEG_GAP


def get_bar_sprite(bar_w, max_segments):
    """A full-height stack of blocks, drawn once and cut down to size per bar."""
    key = ("bar", bar_w, max_segments)
    if key not in _cache:
        h = max_segments * _STEP
        sprite = pygame.Surface((bar_w, h), pygame.SRCALPHA)
        for j in range(max_segments):
            pygame.draw.rect(sprite, INK, (0, h - (j + 1) * _STEP + SEG_GAP, bar_w, SEG_H))
        _cache[key] = sprite
    return _cache[key]


def draw_bars(screen, smoothed, peaks, area, max_frac=0.92):
    """Segmented black bars with a blue top block and a red peak marker."""
    slot = area.w / NUM_BARS
    bar_w = max(3, int(slot * 0.58))
    max_segments = max(1, int(area.h * max_frac / _STEP))
    sprite = get_bar_sprite(bar_w, max_segments)
    for i in range(NUM_BARS):
        x = int(area.left + i * slot + (slot - bar_w) / 2)
        n = int(round(smoothed[i] * max_segments))
        if n > 0:
            hh = n * _STEP
            screen.blit(sprite, (x, area.bottom - hh), (0, sprite.get_height() - hh, bar_w, hh))
            pygame.draw.rect(screen, BLUE,
                             (x, area.bottom - n * _STEP + SEG_GAP, bar_w, SEG_H))
        peak_n = max(n, int(round(peaks[i] * max_segments)))
        if peak_n > 0:
            peak_y = area.bottom - (peak_n + 1) * _STEP + SEG_GAP
            pygame.draw.rect(screen, RED, (x, peak_y + 2, bar_w, 3))


def draw_axis(screen, area):
    """Baseline with a tick per band and frequency markers underneath."""
    slot = area.w / NUM_BARS
    pygame.draw.line(screen, INK, (area.left, area.bottom + 3), (area.right, area.bottom + 3), 1)
    for i in range(NUM_BARS):
        cx = int(area.left + (i + 0.5) * slot)
        pygame.draw.line(screen, INK, (cx, area.bottom + 3), (cx, area.bottom + 6), 1)
    for f, text in ((100, "100 HZ"), (1000, "1 KHZ"), (10000, "10 KHZ")):
        idx = NUM_BARS * math.log(f / MIN_FREQ) / math.log(MAX_FREQ / MIN_FREQ)
        x = int(area.left + idx * slot)
        pygame.draw.line(screen, INK, (x, area.bottom + 3), (x, area.bottom + 14), 1)
        draw_text(screen, FONT_TINY, text, (x + 4, area.bottom + 10), INK)


def draw_glitch(screen, panel, smoothed):
    """Little blue rectangles in the corner that twitch with the music."""
    for k in range(7):
        level = float(smoothed[(k * 6) % NUM_BARS])
        y = panel.top + 40 + k * 11
        width = 16 + int(level * 150)
        pygame.draw.rect(screen, BLUE if k % 3 else INK, (panel.left + 18, y, width, 4))
        pygame.draw.rect(screen, WHITE, (panel.left + 18 + width + 6, y, 10 + (k * 7) % 22, 4))


# -------------------------------------------------------------------- menu --
def pick_file():
    """Open a file dialog. Returns '' if the user cancels."""
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askopenfilename(
        title="Choose a song",
        filetypes=[("Audio files", "*.wav *.mp3 *.flac *.ogg"), ("All files", "*.*")],
    )
    root.destroy()
    return path


def draw_flow_lines(screen, panel, t, energy):
    """Thin animated line art across the menu's sky panel."""
    screen.set_clip(panel)
    xs = range(panel.left, panel.right + 8, 8)
    for k in range(9):
        pts = []
        for x in xs:
            u = (x - panel.left) / panel.w
            y = (panel.centery + (k - 4) * 26
                 + (38 + 30 * energy) * math.sin(u * 6.0 + t * 0.9 + k * 0.55)
                 + 16 * math.sin(u * 15.0 - t * 0.6 + k * 1.3))
            pts.append((x, y))
        pygame.draw.aalines(screen, BLUE if k in (2, 6) else INK, False, pts)
    screen.set_clip(None)


def run_menu(screen, clock, message=""):
    """Show the start menu. Returns 'mic', 'file', 'hand', 'rhythm', or None (quit)."""
    hand_desc = ("a reverb pad played with your hands" if not HAND_ERROR
                 else "needs python 3.12 + mediapipe")
    rows = [
        ("01", "microphone", "live input from your default mic", "mic"),
        ("02", "audio file", "play a song and watch it move", "file"),
        ("03", "hand synth", hand_desc, "hand"),
        ("04", "rhythm game", "turn any song into a game", "rhythm"),
    ]
    rects = [pygame.Rect(40, 126 + i * 118, 500, 108) for i in range(len(rows))]

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return None
                if event.key in (pygame.K_1, pygame.K_m):
                    return "mic"
                if event.key in (pygame.K_2, pygame.K_f):
                    return "file"
                if event.key in (pygame.K_3, pygame.K_h):
                    return "hand"
                if event.key in (pygame.K_4, pygame.K_g):
                    return "rhythm"
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                for (_, _, _, choice), rect in zip(rows, rects):
                    if rect.collidepoint(event.pos):
                        return choice

        t = pygame.time.get_ticks() / 1000
        mouse = pygame.mouse.get_pos()
        hovered = next((i for i, r in enumerate(rects) if r.collidepoint(mouse)), None)

        draw_chrome(screen, "audio visualizer.",
                    ["SYS / AV-04", "4 MODES", f"{NUM_BARS} BANDS / LOG SCALE"],
                    "1 / 2 / 3 / 4 = SELECT     ESC = QUIT", f"{clock.get_fps():.0f} FPS")

        # sky panel with flowing line art
        draw_panel(screen, MENU_PANEL)
        draw_flow_lines(screen, MENU_PANEL, t, 0.6 if hovered is not None else 0.0)
        draw_label(screen, FONT_TINY, "SIGNAL / 4 MODES", (MENU_PANEL.left + 16, MENU_PANEL.top + 14))
        for i, line in enumerate(("every frequency,", "turned into light.")):
            draw_label(screen, FONT_TINY, line, (MENU_PANEL.left + 16, MENU_PANEL.bottom - 52 + i * 22))

        # the three options
        for i, ((num, label, desc, _), rect) in enumerate(zip(rows, rects)):
            hot = i == hovered
            if hot:
                pygame.draw.rect(screen, BLUE, rect)
            else:
                pygame.draw.line(screen, INK, rect.topleft, rect.topright, 1)
            fg = WHITE if hot else INK
            draw_text(screen, FONT_MONO, num, (rect.left + 16, rect.top + 10), WHITE if hot else BLUE)
            draw_text(screen, FONT_ROW, label, (rect.left + 14, rect.top + 30), fg)
            draw_text(screen, FONT_TINY, desc.upper(), (rect.left + 16, rect.bottom - 24),
                      WHITE if hot else INK)
            if hot:
                draw_text(screen, FONT_UI, ">>", (rect.right - 18, rect.centery), WHITE, "midright")
        pygame.draw.line(screen, INK, rects[-1].bottomleft, rects[-1].bottomright, 1)

        if message:
            draw_label(screen, FONT_TINY, "ERR / " + message.upper(), (40, 594),
                       WHITE, RED, "topleft", pad=10)

        pygame.display.flip()
        clock.tick(FPS)


# ---------------------------------------------------------- rhythm charting --
# Turns a song into a playable chart: finds the moments where something
# "happens" in the music (drum hits, plucks, chord stabs), then picks which of
# them become notes, and in which lane, for each difficulty.

LANES = 4
LANE_KEYS = (pygame.K_d, pygame.K_f, pygame.K_j, pygame.K_k)
LANE_LABELS = ("D", "F", "J", "K")
DIFFICULTIES = {
    # gap = shortest time between notes, keep = share of the strongest hits used,
    # hold_min = shortest long note (seconds), hold_share = most notes that may be long
    "easy":   {"gap": 0.42, "keep": 0.40, "doubles": 0.00, "hold_min": 0.60, "hold_share": 0.30},
    "normal": {"gap": 0.26, "keep": 0.70, "doubles": 0.00, "hold_min": 0.50, "hold_share": 0.22},
    "hard":   {"gap": 0.15, "keep": 0.95, "doubles": 0.12, "hold_min": 0.42, "hold_share": 0.18},
}
MAX_HOLD = 2.4               # longest a long note can be (seconds)
ONSET_BIAS = 0.003           # seconds added to detected hit times (measured on test clicks)


def stft_mag(x, n_fft, hop):
    """Short-time Fourier transform magnitudes, shape (frames, n_fft/2 + 1)."""
    x = np.asarray(x, dtype=np.float32)
    if len(x) < n_fft:
        x = np.pad(x, (0, n_fft - len(x)))
    frames = 1 + (len(x) - n_fft) // hop
    win = np.hanning(n_fft).astype(np.float32)
    out = np.empty((frames, n_fft // 2 + 1), dtype=np.float32)
    starts = hop * np.arange(frames)
    offsets = np.arange(n_fft)
    for s in range(0, frames, 256):              # in batches, to keep memory small
        e = min(frames, s + 256)
        out[s:e] = np.abs(np.fft.rfft(x[starts[s:e, None] + offsets[None, :]] * win, axis=1))
    return out


def estimate_tempo(env, fr):
    """Tempo from how the hits repeat (autocorrelation), just for display."""
    n = len(env)
    if n < fr * 6 or env.max() <= 0:
        return 100.0
    e = env - env.mean()
    ac = np.fft.irfft(np.abs(np.fft.rfft(e, 2 * n)) ** 2)[:n]
    lags = np.arange(1, n)
    bpm = 60.0 * fr / lags
    prior = np.exp(-0.5 * (np.log2(bpm / 110.0) / 0.8) ** 2)
    score = np.where((bpm >= 60) & (bpm <= 180), ac[1:] * prior, -np.inf)
    return float(60.0 * fr / lags[int(np.argmax(score))])


def measure_sustain(level, flux, i, fr, max_len=3.0):
    """How long the sound that started at frame i keeps going: until it fades
    to a fraction of its peak, or until something new hits in the same band."""
    peak = level[i:i + 8].max()
    onset_flux = flux[max(0, i - 1):i + 4].max()   # how sharp this note's own attack was
    if peak <= 0 or onset_flux <= 0:
        return 0.0
    end = min(len(level), i + int(max_len * fr))
    j, quiet = i + 4, 0
    while j < end:
        if flux[j] > 0.35 * onset_flux:         # a fresh attack: the old note is over
            break
        quiet = quiet + 1 if level[j] < 0.4 * peak else 0
        if quiet >= 3:                          # faded away
            j -= 2
            break
        j += 1
    return (j - i) / fr


def detect_onsets(x, sr):
    """Find the moments where new sound starts. Returns (onsets, envelope, frames_per_second)."""
    n_fft, hop = 512, 256
    mag = stft_mag(x, n_fft, hop)
    fr = sr / hop
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    logm = np.log1p(30.0 * mag)                                        # loudness-like scale
    flux = np.maximum(0.0, np.diff(logm, axis=0, prepend=logm[:1]))    # only sound getting louder
    bands = [(40, 150), (150, 500), (500, 2200), (2200, 9000)]         # kick / low / mid / hats
    band_flux = np.stack([flux[:, (freqs >= lo) & (freqs < hi)].sum(axis=1)
                          for lo, hi in bands], axis=1)
    level = np.stack([mag[:, (freqs >= lo) & (freqs < hi)].sum(axis=1)
                      for lo, hi in bands], axis=1)                    # how loud each band is
    norm = band_flux / (np.percentile(band_flux, 95, axis=0) + 1e-9)   # each band on its own scale
    env = np.convolve((norm * np.array([1.0, 1.0, 0.8, 0.6])).sum(axis=1),
                      [0.25, 0.5, 0.25], mode="same")
    win = max(3, int(0.5 * fr))
    local = np.convolve(env, np.ones(win) / win, mode="same")          # what's "normal" nearby
    threshold = local * 1.3 + 0.3
    peaks = np.where((env[1:-1] >= env[:-2]) & (env[1:-1] > env[2:])
                     & (env[1:-1] > threshold[1:-1]))[0] + 1
    onsets = []
    for i in peaks:
        onsets.append({"t": (i * hop + n_fft / 2) / sr + ONSET_BIAS,
                       "strength": float(env[i] - local[i]),
                       "bands": norm[max(0, i - 1):i + 2].max(axis=0)})
    if onsets:                                                          # spread the lanes evenly
        mean_bands = np.mean([o["bands"] for o in onsets], axis=0) + 1e-9
        for o, i in zip(onsets, peaks):
            o["rel"] = o["bands"] / mean_bands
            k = int(np.argmax(o["rel"]))                                # the band this hit lives in
            o["sustain"] = measure_sustain(level[:, k], band_flux[:, k], int(i), fr)
    return onsets, env, fr


def build_chart(onsets, difficulty, duration, seed):
    """Pick the notes for one difficulty: strongest hits first, spaced by the
    difficulty's minimum gap, each put in the lane that matches its sound."""
    cfg = DIFFICULTIES[difficulty]
    order = sorted(range(len(onsets)), key=lambda i: -onsets[i]["strength"])
    taken = []                                                          # sorted times
    chosen = []
    for i in order[: int(len(order) * cfg["keep"])]:
        t = onsets[i]["t"]
        if t < 0.8 or t > duration - 0.3:
            continue
        k = bisect.bisect_left(taken, t)
        if k > 0 and t - taken[k - 1] < cfg["gap"]:
            continue
        if k < len(taken) and taken[k] - t < cfg["gap"]:
            continue
        taken.insert(k, t)
        chosen.append(i)
    chosen.sort(key=lambda i: onsets[i]["t"])

    rng = random.Random(seed)
    top = np.percentile([onsets[i]["strength"] for i in chosen], 88) if chosen else 0
    notes, last_lane, run, last_t = [], -1, 0, -9.0
    recent = []                                                         # the last few lanes used
    for i in chosen:
        o = onsets[i]
        # a note goes where its sound points, but lanes used a lot lately are less attractive
        balanced = [o["rel"][k] / (1 + 0.6 * recent.count(k)) for k in range(LANES)]
        prefs = [int(b) for b in np.argsort(-np.array(balanced))]
        lane = prefs[0]
        if lane == last_lane and (o["t"] - last_t < 0.35 or run >= 2):
            lane = prefs[1]                                             # don't hammer one lane
        run = run + 1 if lane == last_lane else 0
        recent = (recent + [lane])[-6:]
        notes.append({"t": float(o["t"]), "lane": lane, "len": 0.0, "_s": o.get("sustain", 0.0)})
        if cfg["doubles"] and o["strength"] >= top and rng.random() < cfg["doubles"] * 4:
            notes.append({"t": float(o["t"]), "lane": (lane + 2) % LANES, "len": 0.0, "_s": 0.0})
        last_lane, last_t = lane, o["t"]

    # Long notes: the hits whose sound lasts longest, up to this difficulty's share.
    candidates = sorted((n for n in notes if n["_s"] >= cfg["hold_min"] + 0.1),
                        key=lambda n: -n["_s"])
    for n in candidates[: int(len(notes) * cfg["hold_share"])]:
        n["len"] = min(n["_s"] - 0.1, MAX_HOLD, duration - 0.3 - n["t"])
    for lane in range(LANES):                                           # nothing may overlap a long note
        in_lane = sorted((n for n in notes if n["lane"] == lane), key=lambda n: n["t"])
        for a, b in zip(in_lane, in_lane[1:]):
            if a["len"] > 0:
                a["len"] = min(a["len"], b["t"] - a["t"] - 0.30)
    for n in notes:
        n["len"] = round(n["len"], 3) if n["len"] >= cfg["hold_min"] else 0.0
        del n["_s"]
    return notes


def to_stereo(data):
    if data.shape[1] == 1:
        data = np.repeat(data, 2, axis=1)
    elif data.shape[1] > 2:
        data = data[:, :2]
    return np.ascontiguousarray(data, dtype=np.float32)


def analyze_rhythm(path, report=None):
    """Read a song and build its charts. Returns a dict for the game."""
    def say(frac, text):
        if report:
            report(frac, text)

    say(0.03, "reading the file")
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    duration = len(data) / sr
    if duration < 10:
        raise ValueError("that song is too short (it needs at least 10 seconds)")
    mono = data.mean(axis=1)
    factor = max(1, int(round(sr / 22050)))                             # analyse at ~22 kHz: faster
    if factor > 1:
        usable = (len(mono) // factor) * factor
        analysis = mono[:usable].reshape(-1, factor).mean(axis=1)
    else:
        analysis = mono
    sra = sr / factor

    say(0.15, "listening for hits")
    onsets, env, fr = detect_onsets(analysis, sra)
    if len(onsets) < 20:
        raise ValueError("couldn't find enough beats in that song to build a chart")

    say(0.7, "finding the tempo")
    bpm = estimate_tempo(env, fr)

    say(0.8, "building the charts")
    seed = len(data) % 100003
    charts = {d: build_chart(onsets, d, duration, seed) for d in DIFFICULTIES}
    density = {}
    for d, notes in charts.items():
        counts, _ = np.histogram([n["t"] for n in notes], bins=60, range=(0, duration))
        density[d] = [float(c) / (duration / 60.0) for c in counts]     # notes per second

    say(1.0, "done")
    return {"name": os.path.splitext(os.path.basename(path))[0], "duration": duration,
            "sr": sr, "stereo": to_stereo(data), "mono": mono, "bpm": bpm,
            "hits_found": len(onsets), "charts": charts, "density": density}


# ------------------------------------------------------------- rhythm game --
LEAD_IN_SECONDS = 3.0                 # countdown before the song starts
WINDOW_PERFECT, WINDOW_GREAT, WINDOW_OK = 0.045, 0.090, 0.135   # seconds either side of a note
HOLD_AUTO = 0.05                       # holding until this close to the tail completes it
HOLD_TAIL_WINDOW = 0.15                # letting go this close to the tail still counts
HOLD_POINTS_PER_SECOND = 220           # bonus for every second a long note is held
POINTS = {"perfect": 300, "great": 200, "ok": 100, "miss": 0}
ACCURACY_WEIGHT = {"perfect": 1.0, "great": 0.75, "ok": 0.4, "miss": 0.0}
JUDGE_COLOR = {"perfect": BLUE, "great": INK, "ok": GREY, "miss": RED}
FIELD = pygame.Rect(380, 96, 360, 504)                          # the playfield panel
LANE_W, LANE_GAP = 84, 4
HIT_Y = FIELD.bottom - 96                                       # where notes should be hit
GAME_FPS = 120                                                  # higher = tighter timing
RHYTHM = {"offset": 0.0, "speed": 560.0}                        # kept between songs


def lane_x(lane):
    return FIELD.x + 6 + lane * (LANE_W + LANE_GAP)


def grade_for(accuracy):
    for limit, grade in ((0.95, "S"), (0.90, "A"), (0.80, "B"), (0.70, "C")):
        if accuracy >= limit:
            return grade
    return "D"


class RhythmGame:
    """One play-through: plays the song, keeps the clock, judges your key presses."""

    def __init__(self, song, difficulty):
        self.song = song
        self.data = song["stereo"]
        self.sr = song["sr"]
        self.difficulty = difficulty
        self.notes = [{"t": n["t"], "lane": n["lane"], "len": n.get("len", 0.0),
                       "end": n["t"] + n.get("len", 0.0), "state": None}
                      for n in song["charts"][difficulty]]
        self.lanes = [[n for n in self.notes if n["lane"] == lane] for lane in range(LANES)]
        for lane_notes in self.lanes:
            lane_notes.sort(key=lambda n: n["t"])
        self.lane_times = [[n["t"] for n in lane_notes] for lane_notes in self.lanes]
        self.ptr = [0] * LANES                       # first note in each lane not yet judged
        self.down = [False] * LANES                  # which keys are held right now
        self.active = [None] * LANES                 # the long note being held in each lane
        self.holds_total = sum(1 for n in self.notes if n["len"] > 0)
        self.holds_done = 0
        self.total_judgments = len(self.notes) + self.holds_total    # a long note is judged twice
        self._tick_t = None
        self._tick_carry = 0.0

        self.offset = RHYTHM["offset"]               # shifts the notes against the music
        self.speed = RHYTHM["speed"]                 # how fast notes fall (pixels per second)

        # the song clock: position in the audio, plus how far we are into the current block
        self.pos = -int(LEAD_IN_SECONDS * self.sr)   # negative = countdown
        self.paused = False
        self.frozen = None
        self.finished = False
        self.stream = None
        self.latency = 0.05
        self._cb_pos = self.pos
        self._cb_wall = time.perf_counter()
        self._last_t = -LEAD_IN_SECONDS

        self.score = 0
        self.combo = 0
        self.max_combo = 0
        self.counts = {"perfect": 0, "great": 0, "ok": 0, "miss": 0}
        self.errors = []                             # how early/late each hit was (seconds)
        self.judgement = None                        # (name, wall time, error) for the popup
        self.flash = [0.0] * LANES                   # when each lane last got hit

    # ---- audio ----------------------------------------------------------
    def _callback(self, outdata, frames, time_info, status):
        outdata.fill(0)
        start = self.pos
        if not self.paused:
            lo, hi = max(start, 0), min(start + frames, len(self.data))
            if hi > lo:
                outdata[lo - start:hi - start] = self.data[lo:hi]
            self.pos = start + frames
        self._cb_pos, self._cb_wall = start, time.perf_counter()
        try:                                         # how long until this block is heard
            lat = time_info.outputBufferDacTime - time_info.currentTime
        except Exception:
            lat = 0.0
        if 0.0 < lat < 0.5:
            self.latency += (lat - self.latency) * 0.1

    def start(self):
        self.stream = sd.OutputStream(samplerate=self.sr, channels=2, blocksize=1024,
                                      latency="low", callback=self._callback)
        self.stream.start()
        try:
            if 0.0 < self.stream.latency < 0.5:
                self.latency = float(self.stream.latency)
        except Exception:
            pass

    def stop(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()

    def set_paused(self, paused):
        if paused and not self.paused:
            self.frozen = self.raw_time()
        elif not paused and self.paused:
            self._last_t = self.frozen if self.frozen is not None else self._last_t
            self.frozen = None
        self.paused = paused

    # ---- the clock ------------------------------------------------------
    def raw_time(self):
        """Where the music is right now, in seconds (what you're hearing)."""
        if self.paused and self.frozen is not None:
            return self.frozen
        t = self._cb_pos / self.sr + (time.perf_counter() - self._cb_wall) - self.latency
        self._last_t = max(self._last_t, t)          # never runs backwards
        return self._last_t

    def now(self):
        return self.raw_time() + self.offset

    # ---- judging --------------------------------------------------------
    def _register(self, name, error=None, label=None):
        self.counts[name] += 1
        if name == "miss":
            self.combo = 0
        else:
            self.score += int(POINTS[name] * (1 + min(self.combo, 40) * 0.05))
            self.combo += 1
            self.max_combo = max(self.max_combo, self.combo)
            if error is not None:
                self.errors.append(error)
        self.judgement = (name, time.perf_counter(), error, label)

    def _miss_note(self, note):
        """A note nobody played. A long note counts as two misses: head and tail."""
        note["state"] = "miss"
        self._register("miss")
        if note["len"] > 0:
            self._register("miss")

    def _finish_hold(self, lane, name):
        note = self.active[lane]
        note["state"] = "done"
        self.active[lane] = None
        self.holds_done += 1
        self._register(name)
        self.flash[lane] = time.perf_counter()

    def _drop_hold(self, lane):
        note = self.active[lane]
        note["state"] = "dropped"
        self.active[lane] = None
        self._register("miss", label="DROPPED")

    def press(self, lane, t=None):
        """A key went down. Hit the nearest note in that lane if it's close enough."""
        t = self.now() if t is None else t
        self.down[lane] = True
        if self.active[lane]:
            return                                   # already holding something in this lane
        notes, i = self.lanes[lane], self.ptr[lane]
        while i < len(notes):
            note = notes[i]
            error = t - note["t"]                    # positive = late
            if error < -WINDOW_OK:
                break                                # too early: ignore the press
            i += 1
            if error > WINDOW_OK:                    # this note was missed already
                self._miss_note(note)
                continue
            name = ("perfect" if abs(error) <= WINDOW_PERFECT
                    else "great" if abs(error) <= WINDOW_GREAT else "ok")
            if note["len"] > 0:
                note["state"] = "holding"            # now keep the key down until the tail
                self.active[lane] = note
            else:
                note["state"] = name
            self._register(name, error)
            self.flash[lane] = time.perf_counter()
            break
        self.ptr[lane] = i

    def release(self, lane, t=None):
        """A key came up. If it was holding a long note, did it last long enough?"""
        t = self.now() if t is None else t
        self.down[lane] = False
        note = self.active[lane]
        if not note:
            return
        remaining = note["end"] - t
        if remaining <= HOLD_AUTO:
            self._finish_hold(lane, "perfect")
        elif remaining <= HOLD_TAIL_WINDOW:
            self._finish_hold(lane, "great")
        else:
            self._drop_hold(lane)                    # let go too early

    def update(self):
        """Unplayed notes that slid past the hit line become misses; long notes tick along."""
        t = self.now()
        for lane in range(LANES):
            notes, i = self.lanes[lane], self.ptr[lane]
            while i < len(notes) and notes[i]["t"] < t - WINDOW_OK:
                self._miss_note(notes[i])
                i += 1
            self.ptr[lane] = i

        dt = 0.0 if self._tick_t is None else min(0.1, max(0.0, t - self._tick_t))
        self._tick_t = t
        for lane in range(LANES):
            note = self.active[lane]
            if not note:
                continue
            if t >= note["end"] - HOLD_AUTO:         # made it to the end of the tail
                if self.down[lane]:
                    self._finish_hold(lane, "perfect")
                else:
                    self._drop_hold(lane)
            else:
                self._tick_carry += HOLD_POINTS_PER_SECOND * dt
        whole = int(self._tick_carry)
        self.score += whole
        self._tick_carry -= whole

        if self.pos >= len(self.data) + int(0.8 * self.sr):
            for lane in range(LANES):                # the song ended while a key was down
                if self.active[lane]:
                    if self.down[lane]:
                        self._finish_hold(lane, "perfect")
                    else:
                        self._drop_hold(lane)
            self.finished = True

    # ---- results --------------------------------------------------------
    def accuracy(self, judged_only=False):
        judged = sum(self.counts.values()) if judged_only else self.total_judgments
        if judged == 0:
            return 1.0
        return sum(ACCURACY_WEIGHT[k] * v for k, v in self.counts.items()) / judged

    def result(self):
        acc = self.accuracy()
        return {"score": self.score, "max_combo": self.max_combo, "counts": dict(self.counts),
                "accuracy": acc, "grade": grade_for(acc), "errors": list(self.errors),
                "total": self.total_judgments, "notes": len(self.notes),
                "holds": self.holds_total, "holds_done": self.holds_done,
                "difficulty": self.difficulty}

    def spectrum_chunk(self, t):
        """A chunk of the song around time t, for the little spectrum display."""
        i = int(t * self.sr)
        if i < 0 or i >= len(self.song["mono"]):
            return np.zeros(BLOCK_SIZE, dtype=np.float32)
        return pad_to_block(self.song["mono"][i:i + BLOCK_SIZE])


# ------------------------------------------------------- rhythm game screens --
def fit_font(text, width, *fonts):
    """The largest of the given fonts in which the text still fits the width."""
    for font in fonts:
        if font.size(text)[0] <= width:
            return font
    return fonts[-1]


def fmt_time(seconds):
    seconds = int(seconds)
    return f"{seconds // 60}:{seconds % 60:02d}"


def run_rhythm_analysis(screen, clock, path):
    """Chart the song in a helper thread while the window shows progress.
    Returns ('ok', song), ('error', message), ('menu', None) or ('quit', None)."""
    state = {"frac": 0.0, "text": "starting", "result": None, "error": None}

    def report(frac, text):
        state["frac"], state["text"] = frac, text

    def work():
        try:
            state["result"] = analyze_rhythm(path, report)
        except Exception as err:
            state["error"] = err

    thread = threading.Thread(target=work, daemon=True)
    thread.start()
    name = os.path.basename(path)
    name = name if len(name) <= 44 else name[:42] + ".."
    shown = 0.0

    while thread.is_alive():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "quit", None
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return "menu", None
        shown += (state["frac"] - shown) * 0.08
        draw_chrome(screen, "rhythm game.", ["BUILDING A CHART", "HITS / LANES / TEMPO"],
                    "ESC = CANCEL", f"{clock.get_fps():.0f} FPS")
        draw_panel(screen, PANEL)
        cx, cy = PANEL.center
        draw_label(screen, FONT_MONO, "LISTENING TO THE SONG", (cx, cy - 78), INK, PAPER, "midtop", pad=14)
        draw_label(screen, FONT_UI, name.lower(), (cx, cy - 44), INK, PAPER, "midtop", pad=14)
        bar = pygame.Rect(cx - 220, cy + 14, 440, 14)
        pygame.draw.rect(screen, PAPER, bar)
        pygame.draw.rect(screen, BLUE, (bar.x, bar.y, int(bar.w * min(1.0, shown)), bar.h))
        pygame.draw.rect(screen, INK, bar, 1)
        draw_label(screen, FONT_TINY, state["text"].upper(), (cx, cy + 42), INK, PAPER, "midtop")
        pygame.display.flip()
        clock.tick(FPS)

    if state["error"] is not None:
        return "error", f"Couldn't build a chart: {state['error']}"
    return "ok", state["result"]


def run_song_select(screen, clock, song, difficulty="normal"):
    """Pick a difficulty. Returns ('play', difficulty), ('menu', None) or ('quit', None)."""
    names = list(DIFFICULTIES)
    index = names.index(difficulty)
    duration = song["duration"]
    title = song["name"]
    title = title if len(title) <= 40 else title[:38] + ".."
    rects = [pygame.Rect(40, 226 + i * 104, 500, 92) for i in range(len(names))]

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "quit", None
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return "menu", None
                if event.key == pygame.K_RETURN:
                    return "play", names[index]
                if event.key in (pygame.K_UP, pygame.K_LEFT):
                    index = (index - 1) % len(names)
                elif event.key in (pygame.K_DOWN, pygame.K_RIGHT):
                    index = (index + 1) % len(names)
                elif event.key in (pygame.K_1, pygame.K_2, pygame.K_3):
                    index = {pygame.K_1: 0, pygame.K_2: 1, pygame.K_3: 2}[event.key]
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                for i, rect in enumerate(rects):
                    if rect.collidepoint(event.pos):
                        if i == index:
                            return "play", names[index]
                        index = i

        draw_chrome(screen, "rhythm game.",
                    [f"{fmt_time(duration)} LONG", f"~{song['bpm']:.0f} BPM", f"{song['hits_found']} HITS FOUND"],
                    "UP / DOWN = CHOOSE     ENTER = PLAY     ESC = MENU", f"{clock.get_fps():.0f} FPS")

        draw_text(screen, FONT_TINY, "SONG", (40, 100), GREY)
        shown_title = title.lower() + "."
        title_font = fit_font(shown_title, 500, FONT_BIG, FONT_TITLE, FONT_ROW, FONT_UI)
        draw_text(screen, title_font, shown_title, (36, 112 + (72 - title_font.get_height()) // 2), INK)
        draw_text(screen, FONT_MONO, "CHOOSE A DIFFICULTY", (40, 196), BLUE)

        for i, (name, rect) in enumerate(zip(names, rects)):
            hot = i == index
            notes = len(song["charts"][name])
            holds = sum(1 for n in song["charts"][name] if n.get("len", 0) > 0)
            per_sec = notes / duration
            if hot:
                pygame.draw.rect(screen, BLUE, rect)
            else:
                pygame.draw.line(screen, INK, rect.topleft, rect.topright, 1)
            draw_text(screen, FONT_MONO, f"0{i + 1}", (rect.left + 16, rect.top + 10), WHITE if hot else BLUE)
            draw_text(screen, FONT_ROW, name, (rect.left + 14, rect.top + 28), WHITE if hot else INK)
            draw_text(screen, FONT_TINY, f"{notes} NOTES / {holds} HOLDS / {per_sec:.1f} PER SECOND",
                      (rect.left + 16, rect.bottom - 22), WHITE if hot else INK)
        pygame.draw.line(screen, INK, rects[-1].bottomleft, rects[-1].bottomright, 1)

        # right: how busy the chart is across the song
        draw_panel(screen, MENU_PANEL)
        draw_label(screen, FONT_TINY, f"NOTE DENSITY / {names[index].upper()}",
                   (MENU_PANEL.left + 16, MENU_PANEL.top + 14))
        dens = song["density"][names[index]]
        peak = max(1.0, max(song["density"][n][k] for n in names for k in range(60)))
        base, width = MENU_PANEL.top + 300, MENU_PANEL.w - 32
        for k, value in enumerate(dens):
            h = int(value / peak * 180)
            bx = MENU_PANEL.left + 16 + int(k * width / 60)
            pygame.draw.rect(screen, INK if k % 5 else BLUE, (bx, base - h, max(2, width // 60 - 2), h))
        pygame.draw.line(screen, INK, (MENU_PANEL.left + 16, base + 3), (MENU_PANEL.right - 16, base + 3), 1)
        draw_text(screen, FONT_TINY, "START", (MENU_PANEL.left + 16, base + 10), INK)
        draw_text(screen, FONT_TINY, fmt_time(duration), (MENU_PANEL.right - 16, base + 10), INK, "topright")

        draw_label(screen, FONT_TINY, "KEYS / D F J K", (MENU_PANEL.left + 16, MENU_PANEL.top + 340))
        for lane in range(LANES):
            box = pygame.Rect(MENU_PANEL.left + 16 + lane * 60, MENU_PANEL.top + 366, 52, 52)
            pygame.draw.rect(screen, PAPER, box)
            pygame.draw.rect(screen, INK, box, 1)
            draw_text(screen, FONT_UI, LANE_LABELS[lane], box.center, INK, "center")
        draw_label(screen, FONT_TINY,
                   f"PERFECT +-{int(WINDOW_PERFECT * 1000)} MS   GREAT +-{int(WINDOW_GREAT * 1000)} MS   OK +-{int(WINDOW_OK * 1000)} MS",
                   (MENU_PANEL.left + 16, MENU_PANEL.top + 440))
        draw_label(screen, FONT_TINY, "IN GAME: UP/DOWN SPEED   LEFT/RIGHT SYNC   ESC PAUSE",
                   (MENU_PANEL.left + 16, MENU_PANEL.top + 466))
        draw_label(screen, FONT_TINY, "LONG NOTES: HOLD THE KEY UNTIL THE TAIL ENDS",
                   (MENU_PANEL.left + 16, MENU_PANEL.top + 490))

        pygame.display.flip()
        clock.tick(FPS)


def draw_note(screen, lx, note, now, speed, holding=False):
    """One note: a block, plus a body and tail cap if it's a long note."""
    head_y = int(HIT_Y - (note["t"] - now) * speed)
    if note["len"] > 0:
        tail_y = int(HIT_Y - (note["end"] - now) * speed)
        bottom = HIT_Y if holding else head_y
        if bottom > tail_y:
            body = pygame.Rect(lx + 24, tail_y, LANE_W - 48, bottom - tail_y)
            pygame.draw.rect(screen, BLUE if holding else (150, 192, 250), body)
            pygame.draw.rect(screen, INK, body, 2)
        pygame.draw.rect(screen, INK, (lx + 7, tail_y - 6, LANE_W - 14, 8))      # the tail cap
    if holding:
        head_y = HIT_Y                                   # the head waits on the line
    pygame.draw.rect(screen, INK, (lx + 7, head_y - 11, LANE_W - 14, 22))
    pygame.draw.rect(screen, RED if holding else BLUE, (lx + 7, head_y - 11, LANE_W - 14, 5))


def draw_field(screen, game, now):
    """The playfield: sky, lanes, falling notes, hit line, key boxes, flashes."""
    if "field" not in _cache:
        surf = get_sky(FIELD.size).copy()
        for lane in range(LANES):
            stripe = pygame.Surface((LANE_W, FIELD.h), pygame.SRCALPHA)
            stripe.fill((255, 255, 255, 80 if lane % 2 == 0 else 34))
            surf.blit(stripe, (lane_x(lane) - FIELD.x, 0))
        _cache["field"] = surf
    screen.blit(_cache["field"], FIELD.topleft)
    pygame.draw.rect(screen, INK, FIELD, 1)
    for cx, cy in ((FIELD.left - 14, FIELD.top - 14), (FIELD.right + 14, FIELD.top - 14),
                   (FIELD.left - 14, FIELD.bottom + 14), (FIELD.right + 14, FIELD.bottom + 14)):
        draw_cross(screen, cx, cy)

    wall = time.perf_counter()
    held = pygame.key.get_pressed()
    screen.set_clip(FIELD)
    for lane in range(LANES):
        lx = lane_x(lane)
        # notes still to be played, between just-passed and just-appearing
        times, notes = game.lane_times[lane], game.lanes[lane]
        lookahead = (HIT_Y - FIELD.top + 30) / game.speed
        lo = bisect.bisect_left(times, now - 0.25)
        hi = bisect.bisect_right(times, now + lookahead)
        for note in notes[lo:hi]:
            if note["state"] is None:
                draw_note(screen, lx, note, now, game.speed)
        if game.active[lane]:                            # a long note being held right now
            pygame.draw.rect(screen, BLUE, (lx + 8, HIT_Y - 60, LANE_W - 16, 60))
            draw_note(screen, lx, game.active[lane], now, game.speed, holding=True)
        # flash when a note is hit
        age = wall - game.flash[lane]
        if age < 0.2:
            h = int(70 * (1 - age / 0.2))
            pygame.draw.rect(screen, BLUE, (lx + 8, HIT_Y - h, LANE_W - 16, h))
    screen.set_clip(None)

    pygame.draw.line(screen, INK, (FIELD.left, HIT_Y), (FIELD.right - 1, HIT_Y), 3)
    for lane in range(LANES):                                   # key boxes
        box = pygame.Rect(lane_x(lane) + 4, HIT_Y + 16, LANE_W - 8, 50)
        down = bool(held[LANE_KEYS[lane]])
        pygame.draw.rect(screen, BLUE if down else PAPER, box)
        pygame.draw.rect(screen, INK, box, 1)
        draw_text(screen, FONT_UI, LANE_LABELS[lane], box.center, WHITE if down else INK, "center")

    if game.judgement and wall - game.judgement[1] < 0.5:      # PERFECT / GREAT / OK / MISS
        name, _, error, label = game.judgement
        text = label or (name.upper() + ("" if error is None else f"  {error * 1000:+.0f}"))
        draw_label(screen, FONT_UI, text, (FIELD.centerx, HIT_Y - 130), JUDGE_COLOR[name]
                   if name != "miss" else WHITE, PAPER if name != "miss" else RED, "midtop", pad=14)

    if now < 0:                                                 # countdown
        digit = str(int(-now) + 1)
        draw_label(screen, FONT_HUGE, digit, (FIELD.centerx, FIELD.top + 150), INK, PAPER, "midtop", pad=24)
        draw_label(screen, FONT_MONO, "GET READY", (FIELD.centerx, FIELD.top + 290), INK, PAPER, "midtop", pad=12)
    elif now < 0.6:
        draw_label(screen, FONT_UI, "GO", (FIELD.centerx, FIELD.top + 200), WHITE, BLUE, "midtop", pad=18)


def run_game(screen, clock, song, difficulty):
    """Play one song. Returns ('done', result), ('retry', None), ('menu', None) or ('quit', None)."""
    game = RhythmGame(song, difficulty)
    bar_bins = build_bar_bins(song["sr"])
    smoothed, peaks = np.zeros(NUM_BARS), np.zeros(NUM_BARS)
    spectrum_area = pygame.Rect(40, 490, 300, 110)
    title = song["name"] if len(song["name"]) <= 30 else song["name"][:28] + ".."
    outcome = ("menu", None)
    game.start()
    try:
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return "quit", None
                if event.type == pygame.KEYUP:
                    if event.key in LANE_KEYS and not game.paused:
                        game.release(LANE_KEYS.index(event.key))
                    continue
                if event.type != pygame.KEYDOWN:
                    continue
                if game.paused:
                    if event.key in (pygame.K_RETURN, pygame.K_ESCAPE):
                        game.set_paused(False)
                        pressed = pygame.key.get_pressed()               # let go while paused?
                        for lane in range(LANES):
                            if game.down[lane] and not pressed[LANE_KEYS[lane]]:
                                game.release(lane)
                    elif event.key == pygame.K_r:
                        return "retry", None
                    elif event.key == pygame.K_q:
                        return "menu", None
                    continue
                if event.key == pygame.K_ESCAPE:
                    game.set_paused(True)
                elif event.key in LANE_KEYS:
                    game.press(LANE_KEYS.index(event.key))
                elif event.key == pygame.K_UP:
                    game.speed = RHYTHM["speed"] = min(1000.0, game.speed + 40)
                elif event.key == pygame.K_DOWN:
                    game.speed = RHYTHM["speed"] = max(280.0, game.speed - 40)
                elif event.key == pygame.K_LEFT:
                    game.offset = RHYTHM["offset"] = max(-0.3, game.offset - 0.005)
                elif event.key == pygame.K_RIGHT:
                    game.offset = RHYTHM["offset"] = min(0.3, game.offset + 0.005)

            if not game.paused:
                if any(game.active):                     # safety net if a key-up event never arrives
                    pressed = pygame.key.get_pressed()
                    for lane in range(LANES):
                        if game.active[lane] and game.down[lane] and not pressed[LANE_KEYS[lane]]:
                            game.release(lane)
                game.update()
            if game.finished:
                return "done", game.result()

            now = game.now()
            target = compute_bars(game.spectrum_chunk(game.raw_time()), bar_bins, -80, -20,
                                  np.linspace(0, TILT_DB, NUM_BARS))
            if game.paused:
                target = np.zeros(NUM_BARS)
            smoothed = np.clip(np.where(target > smoothed, target, smoothed - FALL_SPEED), 0.0, 1.0)
            peaks = np.clip(np.where(smoothed > peaks, smoothed, peaks - 0.008), 0.0, 1.0)

            judged = sum(game.counts.values())
            draw_chrome(screen, "rhythm game.",
                        [title.upper(), f"{game.difficulty.upper()} / {len(game.notes)} NOTES / {game.holds_total} HOLDS", f"~{song['bpm']:.0f} BPM"],
                        "D F J K = HIT / HOLD     ESC = PAUSE     UP / DOWN = SPEED     LEFT / RIGHT = SYNC",
                        f"{clock.get_fps():.0f} FPS")
            draw_field(screen, game, now)

            # ---- left column: score, combo, accuracy, spectrum ----------------------
            draw_text(screen, FONT_TINY, "SCORE", (40, 104), GREY)
            draw_text(screen, FONT_TITLE, f"{game.score:07d}", (38, 120), INK)
            draw_text(screen, FONT_TINY, "COMBO", (40, 200), GREY)
            draw_text(screen, FONT_HUGE, str(game.combo), (34, 212), BLUE if game.combo >= 10 else INK)
            draw_text(screen, FONT_TINY, "ACCURACY", (40, 342), GREY)
            draw_text(screen, FONT_TITLE, f"{game.accuracy(True) * 100:5.1f}%", (38, 358), INK)
            draw_text(screen, FONT_MONO, f"BEST COMBO {game.max_combo}", (40, 430), INK)
            draw_bars(screen, smoothed, peaks, spectrum_area, 0.85)

            # ---- right column: judgements, progress, settings ------------------------
            x = 780
            draw_text(screen, FONT_TINY, "JUDGEMENTS", (x, 104), GREY)
            for i, name in enumerate(("perfect", "great", "ok", "miss")):
                y = 128 + i * 34
                pygame.draw.rect(screen, JUDGE_COLOR[name], (x, y + 4, 14, 14))
                draw_text(screen, FONT_UI, name.upper(), (x + 26, y), INK)
                draw_text(screen, FONT_UI, str(game.counts[name]), (x + 300, y), INK, "topright")
            pygame.draw.line(screen, INK, (x, 276), (x + 300, 276), 1)
            draw_text(screen, FONT_TINY, "PROGRESS", (x, 290), GREY)
            frac = min(1.0, max(0.0, game.raw_time() / song["duration"]))
            pygame.draw.line(screen, GREY, (x, 322), (x + 300, 322), 1)
            pygame.draw.line(screen, BLUE, (x, 322), (x + int(300 * frac), 322), 4)
            pygame.draw.rect(screen, INK, (x + int(300 * frac) - 4, 316, 8, 12))
            draw_text(screen, FONT_MONO, f"{fmt_time(max(0, game.raw_time()))} / {fmt_time(song['duration'])}", (x, 336), INK)
            draw_text(screen, FONT_MONO,
                      f"HITS {judged}/{game.total_judgments}   HOLDS {game.holds_done}/{game.holds_total}",
                      (x, 360), INK)
            pygame.draw.line(screen, INK, (x, 396), (x + 300, 396), 1)
            draw_text(screen, FONT_TINY, "SETTINGS", (x, 408), GREY)
            draw_text(screen, FONT_MONO, f"SPEED   {int(game.speed)} PX/S", (x, 430), INK)
            draw_text(screen, FONT_MONO, f"SYNC    {game.offset * 1000:+.0f} MS", (x, 454), INK)
            draw_text(screen, FONT_TINY, "SYNC: IF YOU HIT EARLY OR LATE", (x, 492), GREY)
            draw_text(screen, FONT_TINY, "ON PURPOSE, NUDGE IT WITH LEFT/RIGHT", (x, 508), GREY)

            if game.paused:
                draw_label(screen, FONT_TITLE, "paused.", (FIELD.centerx, FIELD.top + 150), INK, PAPER, "midtop", pad=30)
                draw_label(screen, FONT_TINY, "ENTER = RESUME     R = RESTART     Q = MENU",
                           (FIELD.centerx, FIELD.top + 240), INK, PAPER, "midtop", pad=14)

            pygame.display.flip()
            clock.tick(GAME_FPS)
    finally:
        game.stop()
    return outcome


def run_results(screen, clock, song, result):
    """Show the score. Returns 'retry', 'select', 'menu' or 'quit'."""
    errors = result["errors"]
    mean_ms = float(np.mean(errors)) * 1000 if len(errors) >= 15 else None
    hist, _ = np.histogram([e * 1000 for e in errors], bins=15, range=(-135, 135))
    hist_peak = max(1, int(hist.max()))
    counts = result["counts"]
    note = ""
    title = song["name"] if len(song["name"]) <= 36 else song["name"][:34] + ".."

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "quit"
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return "menu"
                if event.key in (pygame.K_RETURN, pygame.K_r):
                    return "retry"
                if event.key == pygame.K_d:
                    return "select"
                if event.key == pygame.K_a:
                    if mean_ms is None:
                        note = "NEED AT LEAST 15 HITS TO MEASURE YOUR TIMING"
                    else:
                        RHYTHM["offset"] = float(np.clip(RHYTHM["offset"] - mean_ms / 1000, -0.3, 0.3))
                        note = f"SYNC SET TO {RHYTHM['offset'] * 1000:+.0f} MS"

        draw_chrome(screen, "results.",
                    [title.upper(), f"{result['difficulty'].upper()} / {result['notes']} NOTES / {result['holds']} HOLDS", "SONG COMPLETE"],
                    "ENTER = PLAY AGAIN     D = CHANGE DIFFICULTY     A = APPLY SYNC     ESC = MENU",
                    f"{clock.get_fps():.0f} FPS")

        draw_text(screen, FONT_TINY, "GRADE", (40, 100), GREY)
        draw_text(screen, FONT_HUGE, result["grade"] + ".", (34, 112), BLUE if result["grade"] in "SA" else INK)
        draw_text(screen, FONT_TINY, "SCORE", (300, 108), GREY)
        draw_text(screen, FONT_TITLE, f"{result['score']:07d}", (298, 124), INK)
        draw_text(screen, FONT_TINY, "ACCURACY", (300, 196), GREY)
        draw_text(screen, FONT_UI, f"{result['accuracy'] * 100:.1f}%", (300, 212), INK)

        draw_text(screen, FONT_TINY, "BEST COMBO", (40, 262), GREY)
        draw_text(screen, FONT_TITLE, str(result["max_combo"]), (38, 278), INK)
        draw_text(screen, FONT_TINY, "HOLDS DONE", (300, 262), GREY)
        draw_text(screen, FONT_TITLE, f"{result['holds_done']}/{result['holds']}", (298, 278), INK)
        pygame.draw.line(screen, INK, (40, 360), (540, 360), 1)
        for i, name in enumerate(("perfect", "great", "ok", "miss")):
            y = 378 + i * 42
            pygame.draw.rect(screen, JUDGE_COLOR[name], (40, y + 5, 16, 16))
            draw_text(screen, FONT_ROW, name, (72, y - 8), INK)
            draw_text(screen, FONT_ROW, str(counts[name]), (540, y - 8), INK, "topright")

        draw_panel(screen, MENU_PANEL)
        draw_label(screen, FONT_TINY, "HOW EARLY OR LATE YOU WERE / MS", (MENU_PANEL.left + 16, MENU_PANEL.top + 14))
        base, width = MENU_PANEL.top + 250, MENU_PANEL.w - 32
        for k, value in enumerate(hist):
            h = int(value / hist_peak * 170)
            bx = MENU_PANEL.left + 16 + int(k * width / 15)
            pygame.draw.rect(screen, BLUE if k == 7 else INK, (bx, base - h, width // 15 - 4, h))
        pygame.draw.line(screen, INK, (MENU_PANEL.left + 16, base + 3), (MENU_PANEL.right - 16, base + 3), 1)
        mid = MENU_PANEL.left + 16 + width // 2
        pygame.draw.line(screen, RED, (mid, base - 180), (mid, base + 10), 2)
        draw_text(screen, FONT_TINY, "EARLY", (MENU_PANEL.left + 16, base + 12), INK)
        draw_text(screen, FONT_TINY, "ON TIME", (mid, base + 12), INK, "midtop")
        draw_text(screen, FONT_TINY, "LATE", (MENU_PANEL.right - 16, base + 12), INK, "topright")

        if mean_ms is None:
            draw_label(screen, FONT_TINY, "TOO FEW HITS TO MEASURE YOUR TIMING", (MENU_PANEL.left + 16, MENU_PANEL.top + 320))
        else:
            word = "LATE" if mean_ms > 0 else "EARLY"
            draw_label(screen, FONT_UI, f"{abs(mean_ms):.0f} MS {word} ON AVERAGE", (MENU_PANEL.left + 16, MENU_PANEL.top + 314))
            draw_label(screen, FONT_TINY, "IF THAT FEELS OFF, PRESS A TO SHIFT THE NOTES", (MENU_PANEL.left + 16, MENU_PANEL.top + 360))
            draw_label(screen, FONT_TINY, "BY THAT MUCH, OR NUDGE IT DURING A SONG.", (MENU_PANEL.left + 16, MENU_PANEL.top + 384))
        if note:
            draw_label(screen, FONT_TINY, note, (MENU_PANEL.left + 16, MENU_PANEL.top + 430), WHITE, BLUE, "topleft", pad=10)

        pygame.display.flip()
        clock.tick(FPS)


def run_rhythm(screen, clock):
    """The whole rhythm-game flow. Returns ('menu'|'quit'|'error', message)."""
    path = pick_file()
    if not path:
        return "menu", None
    status, song = run_rhythm_analysis(screen, clock, path)
    if status != "ok":
        return status, song
    difficulty = "normal"
    while True:
        action, difficulty = run_song_select(screen, clock, song, difficulty)
        if action != "play":
            return action, None
        while True:
            outcome, result = run_game(screen, clock, song, difficulty)
            if outcome == "quit":
                return "quit", None
            if outcome == "menu":
                return "menu", None
            if outcome == "retry":
                continue
            choice = run_results(screen, clock, song, result)
            if choice == "retry":
                continue
            if choice == "select":
                break
            return choice, None


# -------------------------------------------------------------- visualizer --
def start_source(screen, clock, source, info):
    """Open the audio (and camera) in a helper thread while the window keeps
    updating. If the audio driver gets stuck, you get an error after a few
    seconds instead of a frozen window. Returns None when ready, or 'menu' /
    'quit' if you gave up while it was opening."""
    state = {"error": None, "abandoned": False}

    def work():
        try:
            source.start()
        except Exception as err:
            state["error"] = err
            return
        if state["abandoned"]:            # you left before it finished opening
            source.stop()

    thread = threading.Thread(target=work, daemon=True)
    thread.start()
    began = time.time()

    while thread.is_alive():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                state["abandoned"] = True
                return "quit"
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                state["abandoned"] = True
                return "menu"

        draw_chrome(screen, source.name + ".", info, source.hint, f"{clock.get_fps():.0f} FPS")
        draw_panel(screen, source.panel)
        draw_label(screen, FONT_MONO, f"OPENING {source.name.upper()} ...",
                   source.panel.center, INK, PAPER, "center", pad=18)
        pygame.display.flip()
        clock.tick(FPS)

        if time.time() - began > START_TIMEOUT:
            state["abandoned"] = True
            print(f"\n{source.name} didn't open within {START_TIMEOUT}s. Where each thread is stuck:")
            faulthandler.dump_traceback(all_threads=True)
            raise RuntimeError(f"{source.name} didn't respond after {START_TIMEOUT}s "
                               "(see the terminal for details)")

    if state["error"] is not None:
        raise state["error"]
    return None


def run_visualizer(screen, clock, source):
    """Draw the bars for one source. Returns 'menu' or 'quit'."""
    pygame.display.set_caption(f"{source.name} - audio visualizer")

    bar_bins = build_bar_bins(source.sample_rate)
    smoothed = np.zeros(NUM_BARS)
    peaks = np.zeros(NUM_BARS)
    result = "menu"
    panel = source.panel
    area = pygame.Rect(panel.left + 28, panel.top + 28, panel.w - 56, panel.h - 28 - 46)
    info = [f"SR {source.sample_rate} HZ", f"{NUM_BARS} BANDS / FFT {BLOCK_SIZE}",
            f"RANGE {MIN_FREQ}-{MAX_FREQ} HZ"]

    try:
        outcome = start_source(screen, clock, source, info)
        if outcome:
            return outcome
        running = True
        while running and not source.finished:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    result, running = "quit", False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        result, running = "menu", False
                    else:
                        source.handle_key(event.key)

            source.update()
            target = compute_bars(source.get_segment(), bar_bins,
                                  source.db_floor, source.db_ceil, source.tilt)

            # fast up -> down
            smoothed = np.where(target > smoothed, target, smoothed - FALL_SPEED)
            smoothed = np.clip(smoothed, 0.0, 1.0)
            peaks = np.where(smoothed > peaks, smoothed, peaks - 0.008)
            peaks = np.clip(peaks, 0.0, 1.0)

            draw_chrome(screen, source.name + ".", info, source.hint,
                        f"{clock.get_fps():.0f} FPS")
            draw_panel(screen, panel)
            source.draw_background(screen, panel)
            if source.show_glitch:
                draw_glitch(screen, panel, smoothed)
            draw_bars(screen, smoothed, peaks, area, source.bar_max_frac)
            draw_axis(screen, area)
            draw_label(screen, FONT_TINY, "SPECTRUM / LOG SCALE", (panel.left + 18, panel.top + 16))
            source.draw_overlay(screen, panel)

            pygame.display.flip()
            clock.tick(FPS)
    finally:
        source.stop()

    return result


# -------------------------------------------------------------------- main --
def main():
    pygame.display.init()        # not pygame.init(): its audio system isn't used here
    pygame.font.init()           # and can get in the way of sounddevice
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    init_theme()
    clock = pygame.time.Clock()
    message = ""

    while True:
        pygame.display.set_caption("audio visualizer")
        choice = run_menu(screen, clock, message)
        message = ""
        if choice is None:
            break

        if choice == "rhythm":
            try:
                action, payload = run_rhythm(screen, clock)
            except Exception as err:
                message = f"Couldn't run the game: {err}"[:100]
                continue
            if action == "quit":
                break
            if action == "error":
                message = str(payload)[:100]
            continue

        try:
            if choice == "mic":
                source = MicSource()
            elif choice == "hand":
                source = HandSynthSource()
            else:
                path = pick_file()
                if not path:
                    continue                  # dialog cancelled, back to menu
                source = FileSource(path)
        except Exception as err:
            what = "that file" if choice == "file" else "that mode"
            message = f"Couldn't open {what}: {err}"[:100]
            continue

        try:
            result = run_visualizer(screen, clock, source)
        except Exception as err:
            message = f"Couldn't start: {err}"[:100]
            continue

        if result == "quit":
            break

    pygame.quit()


if __name__ == "__main__":
    main()