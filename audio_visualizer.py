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