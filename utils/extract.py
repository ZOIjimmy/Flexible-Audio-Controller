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

note = "A Bb B C C# D Eb E F F# G Ab Am Bbm Bm Cm C#m Dm Ebm Em Fm F#m Gm Abm".split()
plot = [0] * 12
last = float("nan")
conseg = 0
for i in range(len(f0)):
    if not np.isnan(f0[i]):
        f0[i] = 2**(round(math.log2(f0[i]/220)*12)/12)*220
        pitch = round(math.log2(f0[i]/220)*12)%12
        if pitch == last:
            conseg += 1
        else:
            conseg = 0
        if conseg >= 3:
            plot[pitch] += 1
        last = pitch
    else:
        last = float("nan")
print(plot)

coe1 = [0.02772601, -0.021885277499999998, 0.016057425833333333, -0.024805963333333337, 0.022774495000000002, 0.0022130216666666667, -0.03432571416666667, 0.017960493333333338, -0.012172340833333331, 0.015860745833333332, -0.0144441075, 0.002726601666666667]
coe2 = [0.02142058083333333, -0.02470891, 0.008429393333333333, 0.0227024525, -0.017118801666666666, 0.0179762675, -0.024228155833333334, 0.022496708333333334, 0.0016113599999999998, -0.03209938666666667, 0.01231384, -0.0064807350000000005]

def predict_multiclass(features):
    probabilities = []
    for i in range(12):
        p = np.dot(np.roll(coe1, i), features) + 0.35785841749999997
        probabilities.append(p)
    for i in range(12):
        p = np.dot(np.roll(coe2, i), features) - 0.3578584166666667
        probabilities.append(p)
    predicted = np.argmax(probabilities)
    return predicted

predicted = predict_multiclass(plot)
print(note[predicted])
# print(plot)
# print(possibility)
# print(note[np.argmax(possibility)])
# exit()

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

