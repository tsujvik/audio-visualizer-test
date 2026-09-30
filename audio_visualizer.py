"""
Menu:      click a row, or press 1 (microphone) / 2 (audio file) / 3 (hand synth), ESC to quit
Visuals:   ESC returns to the menu
File mode: SPACE = pause/resume, LEFT/RIGHT = seek 5 seconds
           Supported files: wav, mp3, flac, ogg

Hand synthL:
    RIGHT hand: height = pitch (snapped to a scale), pinch open/closed = brightness
    LEFT hand:  height = volume,                      pinch open/closed = reverb amount
    (pinch = distance between thumb tip and index fingertip)
    S = change scale, LEFT/RIGHT = change key, SPACE = mute
"""

import math
import os
import random

import numpy as np
import pygame
import sounddevice as sd
import soundfile as sf

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

# ------------------------------------------------------------------- ui --
PAPER = (243, 245, 248)
INK = (14, 16, 24)
BLUE = (28, 108, 236)
SKY_TOP = (66, 146, 244)
SKY_BOTTOM = (214, 233, 252)
GREY = (146, 154, 168)
DOT = (198, 204, 214)
RED = (236, 64, 84)
WHITE = (255, 255, 255)

PANEL = pygame.Rect(40, 96, WIDTH - 80, HEIGHT - 96 - 60)   #MAINMENU
HAND_PANEL = pygame.Rect(340, 96, WIDTH - 340 - 40, HEIGHT - 96 - 60)
MENU_PANEL = pygame.Rect(580, 96, WIDTH - 580 - 40, HEIGHT - 96 - 60)

FONT_TITLE = FONT_ROW = FONT_UI = FONT_MONO = FONT_TINY = FONT_HUGE = None


def init_theme():
    """Load fonts (call after pygame.init). Falls back to pygame's default font."""
    global FONT_TITLE, FONT_ROW, FONT_UI, FONT_MONO, FONT_TINY, FONT_HUGE
    sans = "helveticaneue,helvetica,arial,segoeui"
    mono = "consolas,menlo,couriernew,monospace"
    FONT_TITLE = pygame.font.SysFont(sans, 52, bold=True)
    FONT_ROW = pygame.font.SysFont(sans, 40, bold=True)
    FONT_UI = pygame.font.SysFont(sans, 22, bold=True)
    FONT_HUGE = pygame.font.SysFont(sans, 104, bold=True)
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
        # background
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

    # black strip down the left edge
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
        self.stream = sd.InputStream(channels=1, samplerate=self.sample_rate,
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

        f_end = self.freq * (self.target_freq / self.freq) ** GLIDE
        freq = np.linspace(self.freq, f_end, frames)
        self.freq = f_end

        rate = ATTACK if self.target_amp > self.amp else RELEASE
        a_end = self.amp + (self.target_amp - self.amp) * rate
        amp = np.linspace(self.amp, a_end, frames)
        self.amp = a_end

        inc = 2 * np.pi * DETUNE_RATIOS[:, None] * freq[None, :] / sr
        phase = self.phases[:, None] + np.cumsum(inc, axis=1)          # (voices, frames)
        self.phases = phase[:, -1] % (2 * np.pi)

        w = HARMONICS ** (-(1.0 + 3.0 * (1.0 - self.brightness)))
        n_ok = max(1, int(0.45 * sr / (freq.max() * DETUNE_RATIOS.max())))
        w[n_ok:] = 0.0
        rms = math.sqrt(np.sum(w ** 2) / 2)

        harm = np.sin(HARMONICS[None, :, None] * phase[:, None, :])    # (voices, harmonics, frames)
        wave = (w @ harm.sum(axis=0)) / (rms * math.sqrt(len(DETUNE_RATIOS)))

        sub_phase = self.sub_phase + np.cumsum(2 * np.pi * (freq * 0.5) / sr)
        self.sub_phase = sub_phase[-1] % (2 * np.pi)
        wave = wave + SUB_LEVEL * 1.4 * np.sin(sub_phase)

        dry = (wave * amp * MASTER_VOLUME).astype(np.float32)

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
    """Show the start menu. Returns 'mic', 'file', 'hand', or None (quit)."""
    hand_desc = ("a reverb pad played with your hands" if not HAND_ERROR
                 else "needs python 3.12 + mediapipe")
    rows = [
        ("01", "microphone", "live input from your default mic", "mic"),
        ("02", "audio file", "play a song and watch it move", "file"),
        ("03", "hand synth", hand_desc, "hand"),
    ]
    rects = [pygame.Rect(40, 130 + i * 146, 500, 128) for i in range(len(rows))]

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
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                for (_, _, _, choice), rect in zip(rows, rects):
                    if rect.collidepoint(event.pos):
                        return choice

        t = pygame.time.get_ticks() / 1000
        mouse = pygame.mouse.get_pos()
        hovered = next((i for i, r in enumerate(rects) if r.collidepoint(mouse)), None)

        draw_chrome(screen, "audio visualizer.",
                    ["SYS / AV-04", "3 INPUT MODES", f"{NUM_BARS} BANDS / LOG SCALE"],
                    "1 / 2 / 3 = SELECT     ESC = QUIT", f"{clock.get_fps():.0f} FPS")

        # sky panel with flowing line art
        draw_panel(screen, MENU_PANEL)
        draw_flow_lines(screen, MENU_PANEL, t, 0.6 if hovered is not None else 0.0)
        draw_label(screen, FONT_TINY, "SIGNAL / 3 INPUT MODES", (MENU_PANEL.left + 16, MENU_PANEL.top + 14))
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
            draw_text(screen, FONT_MONO, num, (rect.left + 16, rect.top + 16), WHITE if hot else BLUE)
            draw_text(screen, FONT_ROW, label, (rect.left + 14, rect.top + 44), fg)
            draw_text(screen, FONT_TINY, desc.upper(), (rect.left + 16, rect.bottom - 28),
                      WHITE if hot else INK)
            if hot:
                draw_text(screen, FONT_UI, ">>", (rect.right - 18, rect.centery), WHITE, "midright")
        pygame.draw.line(screen, INK, rects[-1].bottomleft, rects[-1].bottomright, 1)

        if message:
            draw_label(screen, FONT_TINY, "ERR / " + message.upper(), (40, 568),
                       WHITE, RED, "topleft", pad=10)

        pygame.display.flip()
        clock.tick(FPS)


# -------------------------------------------------------------- visualizer --
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
        source.start()
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
    pygame.init()
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