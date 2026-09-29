import numpy as np
import sounddevice as sd
import pygame

SAMPLE_RATE = 44100
BLOCK_SIZE = 2048
NUM_BARS = 48
MIN_FREQ = 40
MAX_FREQ = 12000
DB_FLOOR = -90
DB_CEIL = -20
FALL_SPEED = 0.03

WIDTH, HEIGHT = 1000, 500
FPS = 60

audio_buffer = np.zeros(BLOCK_SIZE, dtype=np.float32)

def audio_callback(indata, frames, time_info, status):
    global audio_buffer
    audio_buffer = indata[:, 0].copy()

fft_freqs = np.fft.rfftfreq(BLOCK_SIZE, d=1.0 / SAMPLE_RATE)
bar_edges = np.logspace(np.log10(MIN_FREQ), np.log10(MAX_FREQ), NUM_BARS + 1)

bar_bins = []
for i in range(NUM_BARS):
    lo = np.searchsorted(fft_freqs, bar_edges[i])
    hi = max(lo + 1, np.searchsorted(fft_freqs, bar_edges[i + 1]))
    bar_bins.append((lo, hi))

window = np.hanning(BLOCK_SIZE).astype(np.float32)