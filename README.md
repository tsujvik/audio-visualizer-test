# audio-visualizer-test

installation:
git clone https://github.com/tsujvik/audio-visualizer-test.git
cd audio-visualizer-test
py -m pip install numpy sounddevice soundfile pygame-ce

hand synth needs python 3.12 and mediapipe to work. everything else works ok without them
in order to install 3.12 and mediapipe:
py -3.12 -m pip install numpy sounddevice soundfile pygame-ce opencv-python mediapipe==0.10.21

using the app:
py audio_visualizer.py
using the app for hand synth:
py -3.12 audio_visualizer.py

to ensure there is mic input, use mic_test.py
