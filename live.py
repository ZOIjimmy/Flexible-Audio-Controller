#!/usr/bin/env python3
import numpy as np
import librosa
import pyaudio
import soundfile
import keyboard
import threading
import numexpr as ne
import argparse

ARRAY_SIZE = int(1e10)
MAX_ITER = 10000
chunk_size = 8 # about 0.1 second in sample rate 44100

# TODO: channels, echo, eq

class AudioSpeedController:
    def __init__(self):
        self.stft = np.array([])
        self.phase = np.array([])
        self.y = np.array([])
        self.remain = []
        self.speed_formula = None
        self.speed = 1
        self.accel = 0
        self.volume_formula = None
        self.volume = 1
        self.x = 0

    def key_callback(self, e):
        kb = keyboard._pressed_events
        if 0 in kb: # a
            self.speed -= 0.1
        if 1 in kb: # s
            self.speed += 0.1
        if 2 in kb: # d
            self.accel -= 0.1
        if 13 in kb: # w
            self.accel += 0.1
        if 123 in kb: # left
            pass
        if 124 in kb: # right
            pass
        if 125 in kb: # down
            self.volume -= 0.1
        if 126 in kb: # up
            self.volume += 0.1

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
        while True:
            x = t / self.shape[-1]
            if self.speed_formula:
                self.x = self.shape[-1] * ne.evaluate(self.speed_formula)
            else:
                # print('\rspeed: {:.5f}, accel: {:.5f}'.format(self.speed, self.accel), end="")
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
                if self.volume_formula:
                    # print('volume: {:.1f}'.format(self.volume), end="")
                    x = step/self.shape[-1]
                    mag2 = ne.evaluate(self.volume_formula)
                    mag *= max(mag2, 0)
                else:
                    mag *= max(self.volume, 0)
                stretch[..., t] = (np.cos(self.phase) + 1j * np.sin(self.phase)) * mag
                self.phase += np.angle(right) - np.angle(left)

        self.istft(stretch)

    def speed_modify(self, filename, speed=(1,0,False), volume=1, mode="play"):
        waveform, sr = librosa.load(filename, sr=None, mono=False)
        self.stft = librosa.stft(waveform)
        self.shape = self.stft.shape
        self.phase = np.angle(self.stft[..., 0])
        if type(speed) is str:
            self.speed_formula = speed
        elif type(speed) is tuple and len(speed) == 3:
            self.speed, self.accel, reverse = speed
            if reverse:
                self.x = self.shape[-1]
            else:
                self.x = 0
        else:
            raise TypeError("invalid speed param")
        if type(volume) is str:
            self.volume_formula = volume
        elif type(volume) is int or type(volume) is float:
            self.volume = volume
        else:
            raise TypeError("invalid volume param")

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
    parser = argparse.ArgumentParser(
                    prog='live.py',
                    description='play or save the modified music',
                    epilog='')
    parser.add_argument('filename')
    parser.add_argument('-s', '--speed', help="A formula with variable x like '1-3*x'. Overwrite other speed arguments.")
    parser.add_argument('-i', '--initial_velocity', default=-0.2)
    parser.add_argument('-a', '--initial_acceleration', default=-8)
    parser.add_argument('-r', '--reversed', default=True)
    parser.add_argument('-v', '--volume', help="A formula with variable x like 'x**10'. Constant is accepted.", default=1)
    parser.add_argument('-m', '--mode', help="'play' or 'save'", default='play')
    args = parser.parse_args()
    if not args.speed:
        args.speed = (args.initial_velocity, args.initial_acceleration, args.reversed)

    # formula = "1 - 4*x**2 - 2e-1*x"
    # formula = "1 + 3.5*x**2 - 3.8*x"
    # formula = "1 - 3*x"
    # speed = (-0.2, -8, True) 

    controller = AudioSpeedController()
    keyboard.hook(controller.key_callback)
    # controller.speed_modify(filename, speed=formula, volume="x", mode="play")
    controller.speed_modify(args.filename, speed=args.speed, volume=args.volume, mode=args.mode)
