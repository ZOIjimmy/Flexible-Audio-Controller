import numpy as np
import librosa
import pyaudio
import soundfile

ARRAY_SIZE = int(1e10)
MAX_ITER = 10000
chunk_size = 8     # 0.1 second in sample rate 44100

def __overlap_add(y, ytmp, hop_length):
    n_fft = ytmp.shape[-2]
    for frame in range(ytmp.shape[-1]):
        sample = frame * hop_length
        # if N > y.shape[-1] - sample:
            # N = y.shape[-1] - sample
        y[..., sample : (sample + n_fft)] += ytmp[..., :n_fft, frame]

# modified from librosa istft
def istft(stft_matrix: np.ndarray, remain: np.ndarray, hop_length = None, win_length = None, n_fft = None, window = "hann", last = False) -> np.ndarray:
    if n_fft is None:
        n_fft = 2 * (stft_matrix.shape[-2] - 1)
    if win_length is None:
        win_length = n_fft
    if hop_length is None:
        hop_length = int(win_length // 4)

    ifft_window = librosa.filters.get_window(window, win_length, fftbins=True)
    ifft_window = librosa.util.pad_center(ifft_window, size=n_fft)
    ifft_window = librosa.util.expand_to(ifft_window, ndim=stft_matrix.ndim, axes=-2)

    n_frames = stft_matrix.shape[-1]
    dtype = librosa.util.dtype_c2r(stft_matrix.dtype)

    shape = list(stft_matrix.shape[:-2])
    expected_signal_len = n_fft + hop_length * (n_frames - 1)
    shape.append(expected_signal_len)
    expected_signal_len -= n_fft

    y = np.zeros(shape, dtype=dtype)
    fft = librosa.get_fftlib()
    start_frame = int(np.ceil((n_fft // 2) / hop_length))
    ytmp = ifft_window * fft.irfft(stft_matrix[..., :start_frame], n=n_fft, axis=-2)

    shape[-1] = n_fft + hop_length * (start_frame - 1)
    head_buffer = np.zeros(shape, dtype=dtype)
    if len(remain) > 0:
        head_buffer[:,:remain.shape[-1]] = remain[:,:]

    __overlap_add(head_buffer, ytmp, hop_length)

    if y.shape[-1] < shape[-1] - n_fft // 2:
        y[..., :] = head_buffer[..., n_fft // 2 : y.shape[-1] + n_fft // 2]
    else:
        y[..., : shape[-1]] = head_buffer[:,:]

    offset = start_frame * hop_length
    n_columns = int(librosa.util.MAX_MEM_BLOCK // (np.prod(stft_matrix.shape[:-1]) * stft_matrix.itemsize))
    n_columns = max(n_columns, 1)

    frame = 0
    for bl_s in range(start_frame, n_frames, n_columns):
        bl_t = min(bl_s + n_columns, n_frames)
        ytmp = ifft_window * fft.irfft(stft_matrix[..., bl_s:bl_t], n=n_fft, axis=-2)
        __overlap_add(y[..., frame * hop_length + offset :], ytmp, hop_length)
        frame += bl_t - bl_s

    # normalize
    if len(remain) > 0:
        ifft_win_sum = librosa.filters.window_sumsquare(window=window,n_frames=n_frames+4,win_length=win_length,n_fft=n_fft,hop_length=hop_length,dtype=dtype)[n_fft:]
        start = 0
    else:
        ifft_win_sum = librosa.filters.window_sumsquare(window=window,n_frames=n_frames,win_length=win_length,n_fft=n_fft,hop_length=hop_length,dtype=dtype)
        start = n_fft // 2

    ifft_win_sum = librosa.util.fix_length(ifft_win_sum[..., start:], size=y.shape[-1])
    approx_nonzero_indices = ifft_win_sum > librosa.util.tiny(ifft_win_sum)
    if last:
        expected_signal_len += n_fft
    else:
        approx_nonzero_indices[expected_signal_len:] = False
    y[..., approx_nonzero_indices] /= ifft_win_sum[approx_nonzero_indices]

    return y, expected_signal_len

def speed_modify(filename, formula="x", mode="play"):
    waveform, sr = librosa.load(filename, sr=None, mono=False)
    stft = librosa.stft(waveform)
    channels, _, stft_len = stft.shape

    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paFloat32, channels=channels, rate=sr, output=True)
    futures = []
    phase = np.angle(stft[..., 0])

    padding = [(0, 0) for _ in stft.shape]
    padding[-1] = (0, 2)
    stft2 = np.pad(stft, padding, "constant")
    remain = []
    start = 0
    full = np.zeros(shape=(ARRAY_SIZE, 2), dtype=np.float32)

    for it in range(MAX_ITER):
        steps = []
        t = it * chunk_size
        while True:
            x = t / stft.shape[-1]
            v = stft.shape[-1] * eval(formula)
            if v > stft.shape[-1] or v < 0:
                steps.append(None)
                break
            elif t >= (it+1) * chunk_size:
                steps.append(min(v, stft.shape[-1]-1))
                break
            else:
                steps.append(min(v, stft.shape[-1]-1))
            t += 1
        if len(steps) == 1:
            break

        shape = list(stft.shape)
        shape[-1] = len(steps)
        stft_stretch = np.zeros_like(stft, shape=shape)
    
        for t, step in enumerate(steps):
            if step == None:
                stft_stretch[..., t] = 0
            else:
                left = stft2[..., int(step)]
                right = stft2[..., int(step)+1]
                frac = np.mod(step, 1.0)
                mag = (1 - frac) * np.abs(left) + frac * np.abs(right)
                stft_stretch[..., t] = (np.cos(phase) + 1j * np.sin(phase)) * mag
                phase += np.angle(right) - np.angle(left)

        result, expected_len = istft(stft_stretch, remain)
        stft_stretch, remain = result[..., :expected_len], result[..., expected_len:]
        y_stretch = stft_stretch.transpose()

        if mode == "play":
            stream.write(y_stretch.tobytes())
        elif mode == "save":
            full[start:start+y_stretch.shape[0], ...] = y_stretch
        start += y_stretch.shape[0]

    if mode == "save":
        soundfile.write('audio2.wav', full[:start], sr)

    stream.stop_stream()
    stream.close()
    p.terminate()

if __name__ == '__main__':

    filename = "psy.wav"
    formula = "1 - 4*x**2 - 2e-1*x"
    # formula = "1 + 3.5*x**2 - 3.8*x"
    # formula = "1 - 3*x"

    speed_modify(filename, formula, "play")
