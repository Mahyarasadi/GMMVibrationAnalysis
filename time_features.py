import numpy as np
from scipy.stats import skew
import matplotlib.pyplot as plt

# --- Functions ---
def signal_entropy(signal, bins=50):

    # histogram
    hist, _ = np.histogram(signal, bins=bins)

    # probability distribution
    p = hist / np.sum(hist)

    # remove zeros
    p = p[p > 0]

    # Shannon entropy
    entropy = -np.sum(p * np.log2(p))

    return entropy

def rms(x):
    x = x - np.nanmean(x)
    return np.std(x)

def shape_factor(x):
    return rms(x) / np.mean(np.abs(x))

def impulse_factor(signal):
    """
    Impulse Factor = max(abs(signal)) / mean(abs(signal))
    """
    return np.max(np.abs(signal)) / np.mean(np.abs(signal))

def margin_factor(signal):
    """
    Margin Factor = max(abs(signal)) / (RMS(signal))^0.5
    """
    rms_val = np.sqrt(np.mean(signal**2))
    return np.max(np.abs(signal)) / (rms_val**0.5)


def crest(x):
    crest = rms(x) / np.max(np.abs(x))
    return crest

def calc_fft(signal, fs):
    # FFT scaled to RMS and Hanning windowed 
    N = len(signal)
    window = np.hanning(N)
    windowed_signal = window * signal
    window_weight = np.sum(window) / N
    windowed_signal /= window_weight
    fft_amp = np.fft.rfft(windowed_signal)
    fft_amp = (np.sqrt(2)/N) * np.abs(fft_amp)
    fft_amp[0] = fft_amp[0] / np.sqrt(2)
    freqs = np.fft.rfftfreq(N, 1/fs)
    return freqs, fft_amp

# Sampling parameters
fs = 1000        # sampling frequency (Hz)
T = 1            # signal duration (seconds)
time = np.linspace(0, T, fs*T, endpoint=False)

# Signal 1: multi-sine (smooth waveform)
signal1 = (
    1*np.sin(2*np.pi*50*time) +
    0.8*np.sin(2*np.pi*120*time) +
    0.3*np.sin(2*np.pi*250*time)
)

# Signal 2: asymmetric waveform (introduces skewness)
signal2 = (
    1.0*np.sin(2*np.pi*50*time) +
    0.8*np.sin(2*np.pi*100*time)**3 +   # nonlinear distortion
    0.2*np.sin(2*np.pi*200*time)
)
signal2 = (
    1.0*np.sin(2*np.pi*50*time) +
    0.4*np.sin(2*np.pi*120*time)
)

# introduce asymmetry
signal2 = np.where(signal2 > 0, signal2, 0.3*signal2)
signal2 = signal2 - np.mean(signal2)
target_rms = 1
signal1 = signal1 * (target_rms / rms(signal1))
signal2 = signal2 * (target_rms / rms(signal2))



# --- Calculations ---
rms1 = rms(signal1)
sf1 = shape_factor(signal1)
skew1 = skew(signal1)
mean1 = np.mean(signal1)
entropy1 = signal_entropy(signal1)
impulse1 = impulse_factor(signal1)
margin1 = margin_factor(signal1)
crest1 = crest(signal1)


rms2 = rms(signal2)
sf2 = shape_factor(signal2)
skew2 = skew(signal2)
mean2 = np.mean(signal2)
entropy2 = signal_entropy(signal2)
impulse2 = impulse_factor(signal2)
margin2 = margin_factor(signal2)
crest2 = crest(signal2)

freqs, amp1 = calc_fft(signal1, fs)
freqs, amp2 = calc_fft(signal2, fs)

print("Signal 1:")
print(f"RMS: {rms1:.4f}")
print(f"Shape Factor: {sf1:.4f}")
print(f"Skewness: {skew1:.4f}")
print(f"Mean: {mean1:.4f}")
print("Entropy signal1:", entropy1)
print("Impulse1:", impulse1)
print("Margin1:", margin1)
print("Crest1:", crest1)


print()

print("Signal 2:")
print(f"RMS: {rms2:.4f}")
print(f"Shape Factor: {sf2:.4f}")
print(f"Skewness: {skew2:.4f}")
print(f"Mean: {mean2:.4f}")
print("Entropy signal2:", entropy2)
print("Impulse2:", impulse2)
print("Margin2:", margin2)
print("Crest2:", crest2)


plt.figure(figsize=(10,6))
# --- Time signal ---
plt.subplot(2,1,1)
plt.plot(time, signal1)
plt.plot(time, signal2)
plt.legend(('signal1', 'signal2'))
plt.grid(True)
plt.xlabel('time (sec)')
plt.ylabel('amplitude')
plt.title('Time Signals')

# --- FFT ---
plt.subplot(2,1,2)
plt.plot(freqs, amp1)
# plt.plot(freqs, amp2)
plt.legend(('FFT signal1', 'FFT signal2'))
plt.grid(True)
plt.xlabel('frequency (Hz)')
plt.ylabel('amplitude')
plt.title('FFT Spectrum')

plt.tight_layout()
plt.show()