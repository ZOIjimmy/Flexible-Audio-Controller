import numpy as np
import librosa
import pyaudio
import soundfile
import keyboard
import threading
import numexpr as ne

ARRAY_SIZE = int(1e10)
MAX_ITER = 10000
chunk_size = 8 # about 0.1 second in sample rate 44100

# TODO: volume, channels, echo, eq

class AudioSpeedController:
    def __init__(self):
        self.stft = np.array([])
        self.phase = np.array([])
        self.y = np.array([])
        self.remain = []
        self.formula = None
        self.speed = 1
        self.accel = 0
        self.x = 0

    def key_callback(self, e):
        kb = keyboard._pressed_events
        if 123 in kb:
            self.speed -= 0.1
        if 124 in kb:
            self.speed += 0.1
        if 125 in kb:
            self.accel -= 0.1
        if 126 in kb:
            self.accel += 0.1

    def play(self, stream, i):
        stream.write(self.y.tobytes())

    def __overlap_add(self, y, ytmp, hop_length):
        n_fft = ytmp.shape[-2]
        for frame in range(ytmp.shape[-1]):
            sample = frame * hop_length
            y[..., sample : (sample + n_fft)] += ytmp[..., :n_fft, frame]

    # modified from librosa istft
    def istft(self, stft_matrix, hop_length = None, win_length = None, n_fft = None, window = "hann", last = False) -> np.ndarray:
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
        if len(self.remain) > 0:
            head_buffer[:,:self.remain.shape[-1]] = self.remain[:,:]

        self.__overlap_add(head_buffer, ytmp, hop_length)

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
            self.__overlap_add(y[..., frame * hop_length + offset :], ytmp, hop_length)
            frame += bl_t - bl_s

        # normalize
        if len(self.remain) > 0:
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

        self.remain = y[..., expected_signal_len:]
        y[..., approx_nonzero_indices] /= ifft_win_sum[approx_nonzero_indices]
        self.y = y[..., :expected_signal_len].transpose()

    def calculate(self, it):
        steps = []
        t = it * chunk_size
        if self.formula:
            while True:
                x = t / self.shape[-1]
                v = self.shape[-1] * ne.evaluate(self.formula)
                if v > self.shape[-1] or v < 0:
                    steps.append(None)
                    break
                else:
                    steps.append(min(v, self.shape[-1]-1))
                if t >= (it+1) * chunk_size:
                    break
                t += 1
        else:
            print('\rspeed: {:.5f}, accel: {:.5f}'.format(self.speed, self.accel), end="")
            while True:
                self.x += self.speed
                self.speed += self.accel/self.shape[-1]
                if self.x > self.shape[-1] or self.x < 0:
                    steps.append(None)
                    break
                else:
                    steps.append(min(self.x, self.shape[-1]-1))
                if t >= (it+1) * chunk_size:
                    break
                t += 1

        if len(steps) == 1:
            self.y = np.array([])
            return

        shape = list(self.shape)
        shape[-1] = len(steps)
        stretch = np.zeros(shape, dtype=np.complex64)
    
        for t, step in enumerate(steps):
            if step == None:
                stretch[..., t] = 0
            else:
                left = self.stft[..., int(step)]
                right = self.stft[..., int(step)+1]
                frac = np.mod(step, 1.0)
                mag = (1 - frac) * np.abs(left) + frac * np.abs(right)
                stretch[..., t] = (np.cos(self.phase) + 1j * np.sin(self.phase)) * mag
                self.phase += np.angle(right) - np.angle(left)

        self.istft(stretch)

    def speed_modify(self, filename, formula=None, mode="play", param=(1,0,False)):
        waveform, sr = librosa.load(filename, sr=None, mono=False)
        self.stft = librosa.stft(waveform)
        self.shape = self.stft.shape
        self.phase = np.angle(self.stft[..., 0])
        self.formula = formula
        self.speed, self.accel, reverse = param
        if reverse:
            self.x = self.shape[-1]
        else:
            self.x = 0

        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paFloat32, channels=self.shape[0], rate=sr, output=True)

        padding = [(0, 0) for _ in self.shape]
        padding[-1] = (0, 2)
        self.stft = np.pad(self.stft, padding, "constant")

        start = 0
        full = np.zeros(shape=(ARRAY_SIZE, 2), dtype=np.float32)

        for it in range(MAX_ITER):
            self.calculate(it)
            if len(self.y) == 0:
                break

            if mode == "play":
                if it > 0:
                    thrd.join()
                thrd = threading.Thread(target=self.play, args=(stream, 0))
                thrd.start()
            elif mode == "save":
                full[start:start+self.y.shape[0], ...] = self.y
            start += self.y.shape[0]

        if mode == "save":
            soundfile.write('audio2.wav', full[:start], sr)
        else:
            thrd.join()

        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == '__main__':

    filename = "psy.wav"
    formula = "1 - 4*x**2 - 2e-1*x"
    # formula = "1 + 3.5*x**2 - 3.8*x"
    # formula = "1 - 3*x"

    controller = AudioSpeedController()
    keyboard.hook(controller.key_callback)
    controller.speed_modify(filename, formula=formula, mode="play")
    # controller.speed_modify(filename, mode="play", param=(-0.2, -8, True))
