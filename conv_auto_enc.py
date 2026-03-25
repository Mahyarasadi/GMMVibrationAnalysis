import os
import glob
import numpy as np
import pandas as pd
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
print(tf.__version__)

# ─────────────────────────────────────────────
# 1. CONFIG
# ─────────────────────────────────────────────
FOLDER      = "TimeSignals"       # folder containing signal_*.csv files
N_SEGMENTS  = 8                   # later the FFT segmants should be dited per the folllowing document
# https://literature.rockwellautomation.com/idc/groups/literature/documents/at/1444-at001_-en-p.pdf
MODEL_PATH  = "autoencoder_model" # folder to save trained model

# ─────────────────────────────────────────────
# 2. FFT FUNCTION (Hanning windowed, RMS scaled)
# ─────────────────────────────────────────────
def calc_fft(signal, fs):
    N = len(signal)
    window = np.hanning(N)
    windowed_signal = window * signal
    window_weight = np.sum(window) / N
    windowed_signal = windowed_signal / window_weight
    fft_amp = np.fft.rfft(windowed_signal)
    fft_amp = (np.sqrt(2) / N) * np.abs(fft_amp)
    fft_amp[0] = fft_amp[0] / np.sqrt(2)
    freqs = np.fft.rfftfreq(N, 1 / fs)
    return freqs, fft_amp


# ─────────────────────────────────────────────
# 3. READ ALL CSV FILES
# ─────────────────────────────────────────────
def load_signals(folder):
    # this folder should contain all the csv files of the training period signals
    csv_files = sorted(glob.glob(os.path.join(folder, "*.csv")))
    print(f"\nFound {len(csv_files)} CSV file(s) in '{folder}'")
    amplitude_list = []
    fs = None
    for fpath in csv_files:
        df = pd.read_csv(fpath)

        # --- build time vector from actual timestamps ---
        time_col = df.iloc[:, 0].values        # column 1: time
        amp_col  = df.iloc[:, 1].values        # column 2: amplitude

        delta_t = time_col[4] - time_col[3]    # dt from index 4 - index 3
        fs = int(1.0 / delta_t)                # sampling rate
        n_samples = len(time_col)
        time_vector = np.linspace(
                time_col[0],
                time_col[0] + (n_samples - 1) * delta_t,
                n_samples)
        amplitude_list.append(amp_col)
        print(f"  Loaded: {os.path.basename(fpath):20s}  "
              f"samples={len(amp_col)}  fs={fs:.1f} Hz")

    # stack into 2D array: shape (n_signals, n_samples)
    amplitudes_2d = np.vstack(amplitude_list)
    print(f"\nAmplitude array shape : {amplitudes_2d.shape}  "
          f"(signals x samples)")
    print(f"Sampling rate         : {fs:.1f} Hz")
    print(f"Signal duration       : {n_samples / fs:.3f} s")
    return amplitudes_2d, fs, time_vector


# ─────────────────────────────────────────────
# 4. COMPUTE FFT FOR ALL SIGNALS → 2D FFT ARRAY
# ─────────────────────────────────────────────
def compute_fft_array(amplitudes_2d, fs):
    fft_list = []
    freqs = None
    for i, signal in enumerate(amplitudes_2d):
        f, fft_amp = calc_fft(signal, fs)
        fft_list.append(fft_amp)
        if freqs is None:
            freqs = f
    fft_2d = np.vstack(fft_list)   # shape: (n_signals, n_freq_bins)
    print(f"\nFFT array shape       : {fft_2d.shape}  "
          f"(signals × freq bins)")
    print(f"Frequency resolution  : {freqs[1] - freqs[0]:.3f} Hz")
    print(f"Max frequency         : {freqs[-1]:.1f} Hz")
    print('len(amplitude_list', len(fft_list))
    print('len(amplitudes_2d', fft_2d.shape)
    return freqs, fft_2d


# ─────────────────────────────────────────────
# 5. SPLIT FFT INTO N SEGMENTS
# ─────────────────────────────────────────────
def split_segments(fft_2d, freqs, n_segments):
    n_bins = fft_2d.shape[1]
    seg_len = n_bins // n_segments          # length of each segment
    usable  = seg_len * n_segments          # drop any remainder bins
    fft_trimmed = fft_2d[:, :usable]       # shape: (n_signals, usable_bins)
    segments = []
    freq_segments = []
    for s in range(n_segments):
        start = s * seg_len
        end   = start + seg_len
        segments.append(fft_trimmed[:, start:end])
        freq_segments.append(freqs[start:end])

    print(f"\nN_SEGMENTS            : {n_segments}")
    print(f"Bins per segment      : {seg_len}")
    print(f"Freq range per segment: {freq_segments[0][0]:.1f} – "
          f"{freq_segments[0][-1]:.1f} Hz  (first segment)")
    print('segments',segments[0].shape          )

    return segments, freq_segments, seg_len


# ─────────────────────────────────────────────
# 6. BUILD 1-D CAE MODEL
#    Encoder: Conv1D(64,8,relu) → Conv1D(32,8,relu)
#    Decoder: Conv1DTranspose(64,8,relu) → Conv1DTranspose(1,8,linear)
# ─────────────────────────────────────────────
def build_autoencoder(input_length):
    inp = keras.Input(shape=(input_length, 1), name="input")

    # --- encoder ---
    x = layers.Conv1D(filters=64, kernel_size=8,
                      activation="relu", padding="same",
                      name="enc_conv1")(inp)
    x = layers.Conv1D(filters=32, kernel_size=8,
                      activation="relu", padding="same",
                      name="enc_conv2")(x)

    # --- decoder ---
    x = layers.Conv1DTranspose(filters=64, kernel_size=8,
                                activation="relu", padding="same",
                                name="dec_tconv1")(x)
    x = layers.Conv1DTranspose(filters=1, kernel_size=8,
                                activation=None, padding="same",
                                name="dec_tconv2")(x)

    model = keras.Model(inputs=inp, outputs=x, name="1D_CAE")
    model.compile(optimizer="adam", loss="mae")
    return model



# ─────────────────────────────────────────────
# 7. TRAIN ONE AUTOENCODER PER SEGMENT
# ─────────────────────────────────────────────
def train_all_segments(segments, n_segments, model_path):
    EPOCHS      = 50                  # training epochs
    BATCH_SIZE  = 8                   # batch size
    os.makedirs(model_path, exist_ok=True)
    trained_models = []

    for s_idx, seg_data in enumerate(segments):
        print(f"\n{'─'*55}")
        print(f"  Training autoencoder for segment {s_idx + 1} / {n_segments}")
        print(f"  Segment data shape : {seg_data.shape}")

        # add channel dimension → (n_signals, seg_len, 1)
        X = seg_data[:, :, np.newaxis].astype(np.float32)

        # build model for this segment
        model = build_autoencoder(input_length=seg_data.shape[1])

        if s_idx == 0:
            model.summary()          # print architecture once

        # train: input == target (autoencoder)
        history = model.fit(
            X, X,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            shuffle=True,
            validation_split=0.1,
            verbose=1
        )

        final_loss = history.history["val_loss"][-1]
        print(f"  Final val MAE      : {final_loss:.6f}")

        # save model
        save_dir = os.path.join(model_path, f"segment_{s_idx + 1}.keras")
        model.save(save_dir)
        print(f"  Model saved to     : {save_dir}")

        trained_models.append({
            "model"    : model,
            "seg_index": s_idx
        })

    return trained_models


# ─────────────────────────────────────────────
# 8. QUICK SANITY CHECK — reconstruction MAE
# ─────────────────────────────────────────────
def sanity_check(trained_models, segments):
    print(f"\n{'─'*55}")
    print("  Reconstruction MAE per segment (training data):")
    for info in trained_models:
        s_idx    = info["seg_index"]
        model    = info["model"]
        seg_data = segments[s_idx]

        X = seg_data[:, :, np.newaxis].astype(np.float32)

        X_recon = model.predict(X, verbose=0)
        mae     = np.mean(np.abs(X - X_recon))
        print(f"  Segment {s_idx + 1}: MAE = {mae:.6f}")


# ─────────────────────────────────────────────
# 9. MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  1D Convolutional Autoencoder — Training Script")
    print("=" * 55)

    # step 1 — load signals
    amplitudes_2d, fs, time_vector = load_signals(FOLDER)

    # step 2 — compute FFT
    freqs, fft_2d = compute_fft_array(amplitudes_2d, fs)

    # step 3 — split into segments
    segments, freq_segs, seg_len = split_segments(fft_2d, freqs, N_SEGMENTS)

    # step 4 — train one autoencoder per segment
    trained_models = train_all_segments(segments, N_SEGMENTS, MODEL_PATH)

    # step 5 — sanity check
    sanity_check(trained_models, segments)

    print(f"\n{'='*55}")
    print(f"  All done.")
    print(f"  Trained models saved in '{MODEL_PATH}/'")
    print(f"  To load a model later:")
    print(f"    model = tf.keras.models.load_model('{MODEL_PATH}/segment_1')")
    print("=" * 55)
