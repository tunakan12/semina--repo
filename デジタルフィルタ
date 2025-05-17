import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as sig
import matplotlib.pyplot as plt

# 1. バタワース型ディジタルフィルタ設計 
fileName = 'C:/Users/yuuya/Downloads/speech1.wav'
Fs, data = wav.read(fileName)
x = data[:16000*5]  
N_list = [4, 6, 8]  
N = 6  
Cutoff = 1500.0
Wn = Cutoff / (Fs / 2.0)

#2. フィルタの周波数特性
plt.figure(figsize=(8,5))
for n in N_list:
    b, a = sig.butter(n, Wn, 'low', analog=False)
    w, h = sig.freqz(b, a)
    plt.plot(w/np.pi, 20*np.log10(abs(h)), label=f'N={n}')
plt.title('Butterworth Filter Frequency Response')
plt.xlabel('Normalized Frequency [×π rad/sample]')
plt.ylabel('Amplitude [dB]')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

#3. フィルタリング（N=6） 
b, a = sig.butter(N, Wn, 'low', analog=False)
y = sig.lfilter(b, a, x)
z = np.int16(y)
wav.write('filterOut.wav', Fs, z)

#4. 入力・出力波形のスペクトル比較 
def plot_spectrum(signal, Fs, title):
    Nsig = len(signal)
    f = np.fft.fftfreq(Nsig, d=1/Fs)
    spectrum = np.abs(np.fft.fft(signal))
    plt.plot(f[:Nsig//2], spectrum[:Nsig//2])
    plt.title(title)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude')

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plot_spectrum(x, Fs, 'Input Spectrum')
plt.subplot(1, 2, 2)
plot_spectrum(y, Fs, 'Output Spectrum')
plt.tight_layout()
plt.show()

#5. 発展課題: 各次数での周波数特性・インパルス応答 
impulse_len_list = [100, 64] 
for impulse_len in impulse_len_list:
    plt.figure(figsize=(8, 5))
    impulse = np.zeros(impulse_len)
    impulse[0] = 1  
    for n in N_list:
        b, a = sig.butter(n, Wn, 'low', analog=False)
        response = sig.lfilter(b, a, impulse)
        plt.plot(response, label=f'N={n}')
    plt.title(f'Impulse Response (Impulse Length={impulse_len})')
    plt.xlabel('n (sample)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()
