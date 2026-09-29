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

def compute_bars(samples):
    spectrum = np.abs(np.fft.rfft(samples * window)) / BLOCK_SIZE
    levels = np.array([spectrum[lo:hi].max() for lo, hi in bar_bins])
    db = 20 * np.log10(levels + 1e-9)
    return np.clip((db - DB_FLOOR) / (DB_CEIL - DB_FLOOR), 0.0, 1.0)

def bar_color(i, height):
    color = pygame.Color(0)
    hue = (i / NUM_BARS) * 300
    color.hsva = (hue, 85, 45 + 55 * height, 100)
    return color


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Audio Visualizer")
    clock = pygame.time.Clock()

    smoothed = np.zeros(NUM_BARS)
    peaks = np.zeros(NUM_BARS)
    slot = WIDTH / NUM_BARS
    bar_w = max(2, int(slot * 0.75))

    with sd.InputStream(channels=1, samplerate=SAMPLE_RATE,
                        blocksize=BLOCK_SIZE, callback=audio_callback):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False

            target = compute_bars(audio_buffer)

            smoothed = np.where(target > smoothed, target, smoothed - FALL_SPEED)
            smoothed = np.clip(smoothed, 0.0, 1.0)

            peaks = np.where(smoothed > peaks, smoothed, peaks - 0.008)
            peaks = np.clip(peaks, 0.0, 1.0)

            screen.fill((10, 10, 18))
            for i in range(NUM_BARS):
                h = int(smoothed[i] * (HEIGHT - 20))
                x = int(i * slot + (slot - bar_w) / 2)
                pygame.draw.rect(screen, bar_color(i, smoothed[i]),
                                 (x, HEIGHT - h, bar_w, h), border_radius=3)

                cap_y = HEIGHT - int(peaks[i] * (HEIGHT - 20)) - 4
                pygame.draw.rect(screen, (235, 235, 245), (x, cap_y, bar_w, 3))

            pygame.display.flip()
            clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()
