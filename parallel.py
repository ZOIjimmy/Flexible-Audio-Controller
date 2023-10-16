import multiprocessing.managers
import multiprocessing
import numpy as np
import librosa
import pyaudio

ARRAY_SIZE = int(1e10)
MAX_ITER = 1000
MAX_PROCECCOR = 4
chunk_size = 128

def process(stft, steps, i, dtype):
    padding = [(0, 0) for _ in stft.shape]
    padding[-1] = (0, 2)
    stft2 = np.pad(stft, padding, "constant")

    shape = list(stft.shape)
    shape[-1] = len(steps)
    stft_stretch = np.zeros_like(stft, shape=shape)
    phase = np.angle(stft[..., 0]) # TODO

    for t, step in enumerate(steps):
        left = stft2[..., int(step)]
        right = stft2[..., int(step)+1]
        frac = np.mod(step, 1.0)
        mag = (1 - frac) * np.abs(left) + frac * np.abs(right)
        stft_stretch[..., t] = (np.cos(phase) + 1j * np.sin(phase)) * mag
        phase += np.angle(right) - np.angle(left)

    y_stretch = librosa.istft(stft_stretch, dtype=dtype)
    y_stretch = y_stretch.reshape(-1, 1, order='F').ravel()
    shm = multiprocessing.shared_memory.SharedMemory(name="buffer")
    dst = np.ndarray(shape=(ARRAY_SIZE,), dtype=np.float32, buffer=shm.buf)
    dst[i*chunk_size*1024:i*chunk_size*1024+y_stretch.shape[0]] = y_stretch[:]
    # print(y_stretch.shape[0], i)
    return y_stretch.shape[0]


def speed_modify(filename, formula):
    waveform, sr = librosa.load(filename, sr=None, mono=False)
    stft = librosa.stft(waveform)
    channels, _, stft_len = stft.shape

    d_size = np.dtype(np.float32).itemsize * np.prod((ARRAY_SIZE,))
    shm = multiprocessing.shared_memory.SharedMemory(create=True, size=d_size, name="buffer")
    dst = np.ndarray((ARRAY_SIZE,), dtype=np.float32, buffer=shm.buf)
    pool = multiprocessing.Pool(processes=MAX_PROCECCOR)
    futures = []

    for it in range(MAX_ITER):
        steps = []
        t = it * chunk_size
        while True:
            x = t / stft.shape[-1]
            v = stft.shape[-1] * eval(formula)
            if v > stft.shape[-1] or v < 0 or t >= (it+1) * chunk_size:
                break
            else:
                steps.append(min(v, stft.shape[-1]-1))
            t += 1
        if len(steps) == 0:
            break

        futures.append(pool.apply_async(process, (stft, steps, it, waveform.dtype)))
        
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paFloat32, channels=channels, rate=sr, output=True)
    start = 0
    for i in range(len(futures)):
        l = futures[i].get()
        stream.write(dst[start:start+l].tobytes())
        start += l

    pool.close()
    pool.join()
    shm.close()
    shm.unlink()
    stream.stop_stream()
    stream.close()
    p.terminate()

if __name__ == '__main__':

    filename = "psy.wav"
    # formula = "1 - 4*x**2 - 2e-1*x"
    # formula = "1 + 3.5*x**2 - 3.8*x"
    formula = "1 - 2*x"

    speed_modify(filename, formula)
