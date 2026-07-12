"""Compare time-domain EMG-like features for two synthetic signals.

This script computes five commonly used time-domain features to support
condition monitoring and fault detection workflows:

- Mean Absolute Value (MAV)
- Simple Sign Integral (SSI)
- Waveform Length (WL)
- Slope Sign Change (SSC)
- Zero Crossing (ZC)

Based on Nayana and Geethanjali (2017), MAV and SSI are primarily
amplitude-oriented descriptors, while WL, SSC, and ZC provide
frequency-related information.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ----------------------------
# Feature functions
# ----------------------------


def MAV(x):
    """Compute the mean absolute value (MAV) of a signal segment.

    MAV is the absolute average value of a data segment and is commonly used
    as an amplitude descriptor.

    Parameters
    ----------
    x : array_like
        Input signal segment.

    Returns
    -------
    float
        Mean of ``abs(x)``.
    """
    return np.mean(np.abs(x))


def SSI(x):
    """Compute the simple sign integral (SSI) of a signal segment.

    SSI summarizes the energy contained in the segment and is commonly used
    as an amplitude-oriented feature.

    Parameters
    ----------
    x : array_like
        Input signal segment.

    Returns
    -------
    float
        Sum of squared samples, ``sum(x**2)``.
    """
    return np.sum(x**2)


def WL(x):
    """Compute waveform length (WL) of a signal segment.

    WL is the cumulative absolute difference between adjacent samples and
    represents the total waveform length over the time segment. It provides
    frequency-related information.

    Parameters
    ----------
    x : array_like
        Input signal segment.

    Returns
    -------
    float
        Waveform length, ``sum(abs(diff(x)))``.
    """
    return np.sum(np.abs(np.diff(x)))


def SSC(x, threshold=1e-5):
    """Count slope sign changes (SSC) in a signal segment.

    SSC estimates frequency content by counting how often the local slope
    changes sign, subject to a minimum magnitude criterion.

    Parameters
    ----------
    x : array_like
        Input signal segment.
    threshold : float, default=1e-5
        Minimum product threshold used to suppress small fluctuations.

    Returns
    -------
    int
        Number of detected slope sign changes.
    """
    count = 0
    for i in range(1, len(x)-1):
        if ((x[i] - x[i-1]) * (x[i] - x[i+1]) > threshold):
            count += 1
    return count


def ZC(x, threshold=1e-5):
    """Count zero crossings (ZC) in a signal segment.

    ZC provides frequency-related information by counting how many times the
    signal crosses zero, with an optional threshold to reduce sensitivity to
    low-amplitude noise.

    Parameters
    ----------
    x : array_like
        Input signal segment.
    threshold : float, default=1e-5
        Minimum absolute difference between adjacent samples required for a
        crossing to be counted.

    Returns
    -------
    int
        Number of detected zero crossings.
    """
    count = 0
    for i in range(len(x)-1):
        if (x[i] * x[i+1] < 0) and (abs(x[i] - x[i+1]) >= threshold):
            count += 1
    return count


# ----------------------------
# Generate signals
# ----------------------------
fs = 1000  # sampling frequency
t = np.linspace(0, 1, fs)

# Signal 1: low frequency sine
sig1 = np.sin(2 * np.pi * 40 * t)

# Signal 2: higher frequency + larger amplitude
sig2 = np.sin(2 * np.pi * 80 * t)

# ----------------------------
# Compute features
# ----------------------------
features = ['MAV', 'SSI', 'WL', 'SSC', 'ZC']

sig1_values = [
    MAV(sig1),
    SSI(sig1),
    WL(sig1),
    SSC(sig1),
    ZC(sig1)
]

sig2_values = [
    MAV(sig2),
    SSI(sig2),
    WL(sig2),
    SSC(sig2),
    ZC(sig2)
]

# ----------------------------
# Create comparison table
# ----------------------------
df = pd.DataFrame({'Feature': features,
                   'Signal 1 (5 Hz)': sig1_values,
                   'Signal 2 (20 Hz, 2x amp)': sig2_values})

print("\nFeature Comparison Table:\n")
print(df)
path = r"D:\7006575\VBL-VA002\normal\misalg_z_092-3_Ch08_100g_PE_Acceleration.csv"
data = np.genfromtxt(path, delimiter=',')
sample_rate = int (1 / (data[1, 0] - data[0, 0]))
signal = data[:, 3]
signal = signal - np.mean(signal)  # Remove DC offset
N = len(signal)
window = np.hanning(N)
windowed_signal = window * signal
window_weight = np.sum(window) / N
windowed_signal = windowed_signal / window_weight
fft_amp = np.fft.rfft(windowed_signal)
fft_amp = (np.sqrt(2) / N) * np.abs(fft_amp)
fft_amp[0] = fft_amp[0] / np.sqrt(2)
freqs = np.fft.rfftfreq(N, 1 / sample_rate)

plt.figure()
# plt.plot(freqs, fft_amp)
plt.plot(data[:, 0], data[:, 3])
plt.grid('major')
plt.show()