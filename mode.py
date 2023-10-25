import librosa
import numpy as np
import matplotlib.pyplot as plt
import math
import os
import random

directory = "../../../wav"
files = [f for f in os.listdir(directory) if f.endswith(".wav")]
random_file = random.choice(files)
random_path = os.path.join(directory, random_file)
print(random_path)

y, sr = librosa.load(random_path)
y_harmonic, y_percussive = librosa.effects.hpss(y)
# y_harmonic, y_percussive = librosa.effects.hpss(y[2006550:2271150])
y = y_harmonic
stft = librosa.stft(y)
stft[:17,:] = np.zeros((17,stft.shape[-1]))
y = librosa.istft(stft)
f0, voiced_flag, voiced_probs = librosa.pyin(y,
                                             fmin=librosa.note_to_hz('C3'),
                                             fmax=librosa.note_to_hz('C7'))
times = librosa.times_like(f0)

note = "A Bb B C C# D Eb E F F# G Ab".split()
plot = [0] * 12
last = float("nan")
for i in range(len(f0)):
    if not np.isnan(f0[i]):
        f0[i] = 2**(round(math.log2(f0[i]/220)*12)/12)*220
        pitch = round(math.log2(f0[i]/220)*12)%12
        if pitch == last:
            plot[pitch] += 1
        last = pitch
    else:
        last = float("nan")
possibility = np.zeros(12)
for j in range(12):
    possibility[j] = plot[j] // 2
    for i in (0,2,4,5,7,9,11):
        possibility[j] += plot[(i+j)%12]
print(plot)
print(possibility)
print(note[np.argmax(possibility)])

D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
fig, ax = plt.subplots()
img = librosa.display.specshow(D, x_axis='time', y_axis='log', ax=ax)
ax.set(title='pYIN fundamental frequency estimation')
fig.colorbar(img, ax=ax, format="%+2.f dB")
ax.plot(times, f0, label='f0', color='cyan', linewidth=3)
ax.legend(loc='upper right')
plt.show()

from pydub import AudioSegment
from pydub.playback import play

def frequencies_to_audio(note_frequencies, sample_rate, frame_duration):
    audio = AudioSegment.empty()
    time = 0.0
    for i, freq in enumerate(note_frequencies):
        if not np.isnan(freq):
            t = np.linspace(0, frame_duration, int(sample_rate * frame_duration), endpoint=False)
            note = 0.5 * np.sin(2 * np.pi * freq * t)
            note = (note * 32767).astype(np.int16)
            audio_segment = AudioSegment.from_mono_audiosegments(AudioSegment(
                data=note.tobytes(order='C'),
                sample_width=note.itemsize,
                frame_rate=sample_rate,
                channels=1
            ))

            fade_duration = 10
            audio_segment = audio_segment.fade_in(fade_duration).fade_out(fade_duration)
            audio += audio_segment
        else:
            duration = frame_duration
            silence = AudioSegment.silent(duration=int(duration * 1000))

            if i < len(note_frequencies) - 1 and not np.isnan(note_frequencies[i + 1]):
                silence = silence.fade_out(10)
            audio += silence
        time += frame_duration
    return audio

audio = frequencies_to_audio(f0, sr, 1/sr*1000)

play(audio)

