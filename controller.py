import numpy as np
import librosa
import pyaudio
import soundfile

ARRAY_SIZE = int(1e10)

def speed_modify(filename, formula):
    waveform, sr = librosa.load(filename, sr=None, mono=False)
    stft = librosa.stft(waveform)
    channels, _, stft_len = stft.shape

    time_steps = []
    t = 0
    while t < ARRAY_SIZE:
        x = t / stft_len
        v = stft_len * eval(formula)
        if v > stft_len or v < 0:
            break
        else:
            time_steps.append(min(v, stft_len-1))
        t += 1

    padding = [(0, 0) for _ in stft.shape]
    padding[-1] = (0, 2)
    stft_pad = np.pad(stft, padding, "constant")
    shape = list(stft.shape)
    shape[-1] = len(time_steps)
    stft_stretch = np.zeros_like(stft, shape=shape)
    phase = np.angle(stft[..., 0])

    for t, step in enumerate(time_steps):
        left = stft_pad[..., int(step)]
        right = stft_pad[..., int(step)+1]
        frac = np.mod(step, 1.0)
        mag = (1 - frac) * np.abs(left) + frac * np.abs(right)
        stft_stretch[..., t] = (np.cos(phase) + 1j * np.sin(phase)) * mag
        phase += np.angle(right) - np.angle(left)

    y_stretch = librosa.istft(stft_stretch, dtype=waveform.dtype).transpose()
    # y_stretch = y_stretch.reshape(-1, 1, order='F').ravel()
    
    # p = pyaudio.PyAudio()
    # stream = p.open(format=pyaudio.paFloat32, channels=channels, rate=sr, output=True)
    # stream.write(y_stretch.tobytes())

    # stream.stop_stream()
    # stream.close()
    # p.terminate()

    soundfile.write('good.wav', y_stretch, sr)

if __name__ == '__main__':

    filename = "psy.wav"
    formula = "1 - 4*x**2 - 2e-1*x"
    # formula = "1 + 3.5*x**2 - 3.8*x"
    # formula = "1 - 5*x"
    speed_modify(filename, formula)
