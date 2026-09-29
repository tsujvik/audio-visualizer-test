import sounddevice as sd
import numpy as np

print(sd.query_devices())
print("\nDefault input device index:", sd.default.device[0])

def cb(indata, frames, time_info, status):
    if status:
        print(status)
    level = float(np.abs(indata).max())
    print(f"{level:.4f} " + "#" * int(level * 300))

with sd.InputStream(channels=1, callback=cb):
    sd.sleep(5000)