import librosa
import librosa
import numpy as np
import matplotlib.pyplot as plt
import math
import os
import random

directory = "../../../wav"
files = [f for f in os.listdir(directory) if f.endswith(".wav")]

note = "A Bb B C C# D Eb E F F# G Ab Am Bbm Bm Cm C#m Dm Ebm Em Fm F#m Gm Abm".split()

def extract(file):
    y, sr = librosa.load(file)
    y, y_percussive = librosa.effects.hpss(y)
    # y_mono = librosa.to_mono(y)
    # chroma = librosa.feature.chroma_cqt(y=y_mono, sr=sr)
    # _sum = chroma.sum(axis=1)
    # print(_sum)
    # mode1 = _sum.argmax()
    # if _sum[(mode1+3)%12] + _sum[(mode1-4)%12] > _sum[(mode1+4)%12] + _sum[(mode1-3)%12]:
        # mode1 = (mode1+3) % 12 + 12
    # else:
        # mode1 = (mode1+3) % 12
    stft = librosa.stft(y)
    stft[:17,:] = np.zeros((17,stft.shape[-1]))
    y = librosa.istft(stft)
    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C3'), fmax=librosa.note_to_hz('C7'))
    times = librosa.times_like(f0)

    plot = [0] * 12
    last = float("nan")
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
    return plot

for file in files:
    print(file)
    path = os.path.join(directory, file)
    print(extract(path))
