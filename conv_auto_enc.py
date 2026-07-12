# fmt: off
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import pandas as pd
import numpy as np
from signal_to_bands import SignalToBands, BANDS_LIST
# fmt: on


"""
Convolutional Autoencoder for Anomaly Detection in Vibration Signals.

Requires:
tensorflow 2.10.0
numpy 1.26.4
pandas 1.5.3

This module implements a system for training and evaluating 1D convolutional
autoencoders on FFT-bands extracted from vibration signals for anomaly detection.

The training process involves:
- Loading training signals from CSV files
- Computing FFT with Hanning windowing
- Breaking FFT into frequency bands
- Training separate autoencoders for each frequency band using MAE loss
- Calculating reconstruction error thresholds for anomaly detection

The evaluation process assesses new signals by comparing reconstruction errors
against the trained thresholds.

Notes
-----
The system uses MAE loss for autoencoder training to reduce sensitivity to
noise compared to MSE/RMSE, following recommendations in the literature.

References
----------
.. [1] Kichang Park, and Yongkwan Lee
An Experiment on Anomaly Detection for Fault Vibration Signals
Using Autoencoder-Based N-Segmentation Algorithm
.. [2] Literature reference for Rockwell Automation vibration analysis.

Classes
-------
AutoEncoderTrainer
    Builds training datasets from raw signals, trains one model per band,
    and exports per-band thresholds.
AutoEncoderEvaluator
    Loads trained models and thresholds, then scores one signal.
"""
EPOCHS = 50  # The number of times the model trains over the entire dataset
BATCH_SIZE = 10  # N samples processed together in one weight-update step
THRESHOLD_EPS = 1e-9  # numeric tolerance for threshold comparisons


class AutoEncoderTrainer:
    """
    Trainer class for convolutional autoencoders on vibration signals.

    This class handles the complete training pipeline: signal loading, FFT computation,
    frequency band extraction, model training, and threshold calculation for anomaly detection.

    Parameters
    ----------
    train_signals_folder : str, default "train_signals_folder"
        Path to folder containing training signal CSV files.
    model_path : str, default "autoencoder_model"
        Path to save trained models and thresholds.
    demo : bool, default False
        If True, generate synthetic demo signals instead of loading from folder.
    speed_range : list[float]
        Expected machine speed range in RPM as ``[min_speed, max_speed]``.

    Attributes
    ----------
    train_signals_folder : str
        Training signals folder path.
    model_path : str
        Model save path.
    demo : bool
        Demo mode flag.
    speed_range : list[float]
        Expected machine speed range in RPM as ``[min_speed, max_speed]``.

    Notes
    -----
    The training tensor is organized as ``(n_bands, n_signals, n_points)`` so
    each model is trained on one band across all training signals.
    """

    def __init__(self, train_signals_folder="train_signals_folder",
                 model_path="autoencoder_model", demo=False,
                 speed_range: list = [500, 3000],):
        self.train_signals_folder = train_signals_folder
        self.model_path = model_path
        self.demo = demo
        self.speed_range = speed_range
        if self.demo:
            self.generate_demo_signals(20)

    def generate_demo_signals(self, N):
        """
        Generate synthetic demo signals for testing.

        Creates N CSV files with synthetic vibration signals composed of
        sine waves at 10, 20, and 30 Hz with random amplitudes.

        Parameters
        ----------
        N : int
            Number of demo signals to generate.

        Returns
        -------
        None

        Notes
        -----
        Signals are 1 second long at 1024 Hz sampling rate.
        """
        fs = 1024          # sampling frequency (Hz)
        duration = 1.0     # signal duration (seconds)
        n_samples = int(fs * duration)
        freqs = [10, 20, 30]   # Hz — peaks to include in every signal
        # --- create output folder ---
        folder = self.train_signals_folder
        os.makedirs(folder, exist_ok=True)
        for i in range(1, N + 1):
            # random amplitude for each frequency component (0.7 to 1.3)
            amplitudes = [np.random.uniform(0.7, 1.3) for _ in freqs]
            filename = os.path.join(folder, f"signal_{i}.csv")
            data = []
            for s in range(n_samples):
                t = s / fs
                # sum of sine waves at 10, 20, 30 Hz with random amplitudes
                amplitude = sum(
                    amplitudes[j] * np.sin(2 * np.pi * freqs[j] * t)
                    for j in range(len(freqs)))
                data.append([round(t, 6), round(amplitude, 6)])
            df = pd.DataFrame(data, columns=["time", "amplitude"])
            df.to_csv(filename, index=False)
        print(f"\nDone. {N} signal(s) saved to '{folder}/'.")

    def load_signals(self):
        """
        Load training signals from CSV files.

        Reads all CSV files in the training folder, extracts time and amplitude
        columns, and stacks them into a 2D array.

        Returns
        -------
        amplitudes_2d : ndarray
            2D array of shape (n_signals, n_samples).
        fs : float
            Sampling frequency in Hz.

        Notes
        -----
        Assumes CSV format with columns: time, amplitude.
        Sampling rate is calculated from time differences.
        """
        # this folder should contain all the csv files of the training period signals
        csv_files = [os.path.join(self.train_signals_folder, f) for f in os.listdir(
            self.train_signals_folder) if f.endswith(".csv")]
        print(
            f"\nFound {len(csv_files)} CSV file(s) in '{self.train_signals_folder}'")
        amplitude_list = []
        for fpath in csv_files:
            df = pd.read_csv(fpath)

            # --- build time vector from actual timestamps ---
            time_col = df.iloc[:, 0].values        # column 1: time
            amp_col = df.iloc[:, 1].values        # column 2: amplitude
            # sampling rate from first interval
            fs = 1.0 / (time_col[1] - time_col[0])
            n_samples = len(time_col)
            amplitude_list.append(amp_col)
            print(f"  Loaded: {os.path.basename(fpath):20s}")
        # stack into 2D array: shape (n_signals, n_samples)
        amplitudes_2d = np.vstack(amplitude_list)
        print(f"\nAmplitude array shape : {amplitudes_2d.shape}  "
              f"(signals x samples)")
        print(f"Sampling rate         : {fs:.1f} Hz")
        print(f"Signal duration       : {n_samples / fs:.3f} s")
        return amplitudes_2d, fs

    def build_autoencoder(self, input_length):
        """
        Build a 1D convolutional autoencoder model.

        Creates a symmetric convolutional autoencoder with encoder-decoder
        architecture using MAE loss for anomaly detection.

        Parameters
        ----------
        input_length : int
            Length of the input sequence (number of frequency bins in band).

        Returns
        -------
        keras.Model
            Compiled 1D convolutional autoencoder model.

        Notes
        -----
        Architecture:
        Encoder: Conv1D(64, 8, relu) -> Conv1D(32, 8, relu)
        Decoder: Conv1DTranspose(64, 8, relu) -> Conv1DTranspose(1, 8, linear)

        Uses MAE loss instead of MSE/RMSE to reduce sensitivity to noise,
        which can lead to false positives in anomaly detection [1]_.

        References
        ----------
        .. [1] Kang, Kim, Kang, & Gwak (2021).
        """
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

    def train_all_bands(self, segments):
        """Train one autoencoder per diagnostic band.

        Parameters
        ----------
        segments : ndarray
            Training tensor with shape ``(n_bands, n_signals, n_points)``.

        Returns
        -------
        list[dict]
            Metadata for each trained band model. Every item contains keys
            ``"model"`` and ``"band_index"``.
        """
        # TRAIN ONE AUTOENCODER PER BAND
        os.makedirs(self.model_path, exist_ok=True)
        trained_models = []

        for s_idx, seg_data in enumerate(segments):
            print(f"\n{'─'*55}")
            print(
                f"  Training autoencoder for band {s_idx}")
            print(f"  Band data shape : {seg_data.shape}")
            # add channel dimension → (n_signals, seg_len, 1)
            X = seg_data[:, :, np.newaxis].astype(np.float32)
            # build model for this band
            model = self.build_autoencoder(input_length=seg_data.shape[1])
            if s_idx == 0:
                model.summary()          # print architecture once
            # train: input == target (autoencoder)
            history = model.fit(
                X, X,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                shuffle=True,
                validation_split=0.1,
                verbose=1)
            final_loss = history.history["val_loss"][-1]
            print(f"  Final val MAE      : {final_loss:.6f}")
            # save model
            save_dir = os.path.join(
                self.model_path, f"band_{s_idx}.keras")
            model.save(save_dir)
            print(f"  Model saved to     : {save_dir}")
            trained_models.append({
                "model": model,
                "band_index": s_idx})
        return trained_models

    def export_threshold(self, trained_models, segments):
        """
        Compute and save anomaly thresholds derived from training errors.

        Parameters
        ----------
        trained_models : list[dict]
            Output from :meth:`train_all_bands`, including models and indices.
        segments : ndarray
            Training tensor with shape ``(n_bands, n_signals, n_points)``.

        Returns
        -------
        None

        Notes
        -----
        Threshold levels are derived from per-band maximum reconstruction error:

        ``observe = 1.25 * acceptable``
        ``alert = 1.25 * observe``
        ``critical = 1.25 * alert``

        The output CSV is written to ``<model_path>/thresholds.csv``.
        """
        print(f"\n{'─'*55}")
        print("  Reconstruction Maximum Error per band (training data):")
        max_errors = []
        for info in trained_models:
            s_idx = info["band_index"]
            model = info["model"]
            seg_data = segments[s_idx]
            X = seg_data[:, :, np.newaxis].astype(np.float32)
            X_recon = model(X, training=False).numpy()
            mae = np.max(np.abs(X - X_recon))
            print(f" Band {s_idx}: Maximum Error = {mae:.6f}")
            max_errors.append(mae)
        weight_accel = 1.5  # 50% distance observe/ alert to alert/ critical
        acceptable = np.array(max_errors)
        observe = weight_accel * acceptable
        alert = weight_accel * observe
        critical = weight_accel * alert
        n_bands = len(acceptable)
        df = pd.DataFrame(
            [acceptable, observe, alert, critical],
            index=["acceptable", "observe", "alert", "critical"],
            columns=[f"band{i}" for i in range(n_bands)])
        threshold_path = os.path.join(self.model_path, "thresholds.csv")
        # Preserve full floating-point precision to avoid tiny truncation drift.
        df.to_csv(threshold_path, float_format="%.18e")
        print(f"  Thresholds saved to: {threshold_path}")

    def train_system(self):
        """Run the full training pipeline.

        Returns
        -------
        None

        Notes
        -----
        Signals are converted into order-domain bands via ``SignalToBands`` and
        transposed to ``(n_bands, n_signals, n_points)`` before training.
        """
        total_signals, fs = self.load_signals()
        bands = []
        for signal in total_signals:
            signal_to_bands = SignalToBands(signal, self.speed_range, fs)
            bands.append(signal_to_bands.band_vectors)
        bands = np.array(bands)
        bands = np.transpose(bands, (1, 0, 2))
        trained_models = self.train_all_bands(bands)
        self.export_threshold(trained_models, bands)


class AutoEncoderEvaluator:
    """
    Evaluator class for anomaly detection using trained autoencoders.

    This class loads trained models and thresholds to evaluate new signals
    for anomalies by comparing reconstruction errors against thresholds.

    Parameters
    ----------
    model_path : str, default "autoencoder_model"
        Path to folder containing trained models and thresholds CSV.
    speed_range : list[float] or None, default None
        Expected machine speed range in RPM passed to ``SignalToBands``.

    Attributes
    ----------
    model_path : str
        Path to model files and thresholds.
    speed_range : list[float] or None
        Speed range used during feature extraction.
    """

    def __init__(self, model_path="autoencoder_model", speed_range=None):
        self.model_path = model_path
        self.speed_range = speed_range

    def load_single_signal(self, test_signal_path):
        """
        Load a single test signal from CSV file.

        Parameters
        ----------
        test_signal_path : str
            Path to the CSV file containing the test signal.

        Returns
        -------
        amp_col : ndarray
            Amplitude values of the signal.
        fs : int
            Sampling frequency in Hz.

        Notes
        -----
        Sampling frequency is estimated from the timestamp difference between
        samples at indices 1 and 0.
        """
        df = pd.read_csv(test_signal_path)
        time_col = df.iloc[:, 0].values       # column 1: time
        amp_col = df.iloc[:, 1].values       # column 2: amplitude

        # Keep fs computation consistent with training (no rounding to int),
        # otherwise order-domain interpolation shifts and inflates errors.
        fs = 1.0 / (time_col[1] - time_col[0])
        return amp_col, fs

    def predict_errors(self, bands):
        """
        Predict reconstruction errors for each frequency band.

        Loads trained models and computes maximum absolute reconstruction errors
        for each frequency band of the input signal.

        Parameters
        ----------
        bands : list of ndarray
            List of band vectors from ``SignalToBands.band_vectors``.

        Returns
        -------
        errors : list of float
            Maximum reconstruction errors for each band.
        """
        # print(f"\n{'─'*55}")
        # print("  Reconstruction error per band:")
        errors = []
        for s_idx in range(len(BANDS_LIST)):
            model_file = os.path.join(
                self.model_path, f"band_{s_idx}.keras")
            model = tf.keras.models.load_model(model_file)
            X = bands[s_idx][np.newaxis, :, np.newaxis].astype(np.float32)
            X_recon = model(X, training=False).numpy()
            max_err = float(np.max(np.abs(X - X_recon)))
            mae = float(np.mean(np.abs(X - X_recon)))
            errors.append(max_err)
            # print(
            #     f"  Band {s_idx:2d}:  Max Error = {max_err:.6f}   MAE = {mae:.6f}")
        return errors

    def assess_state(self, errors):
        """
        Compare reconstruction errors against saved thresholds.

        Parameters
        ----------
        errors : list of float
            Reconstruction errors for each band.

        Returns
        -------
        dict or None
            Dictionary with the most critical status as key and band index
            (1-based) as value. Returns ``None`` when thresholds are missing.

        Notes
        -----
        Thresholds are loaded from ``thresholds.csv`` in ``model_path``.
        Criticality order: CRITICAL > ALERT > OBSERVE > ACCEPTABLE.
        """
        threshold_path = os.path.join(self.model_path, "thresholds.csv")
        if not os.path.exists(threshold_path):
            print("\n  (no thresholds.csv found — skipping comparison)")
            return
        thresholds = pd.read_csv(threshold_path, index_col=0)
        # print(f"\n{'─'*55}")
        # print(f"  {'Band':<10} {'Error':>12} {'Status'}")
        band_scores = []
        for s_idx, err in enumerate(errors):
            col = f"band{s_idx}"
            acceptable = thresholds.loc["acceptable", col]
            observe = thresholds.loc["observe", col]
            alert = thresholds.loc["alert", col]
            critical = thresholds.loc["critical", col]
            # Numerical tolerance prevents false category jumps at the boundary.
            if err <= observe + THRESHOLD_EPS:
                status = "ACCEPTABLE"
            elif err <= alert + THRESHOLD_EPS:
                status = "OBSERVE"
            elif err <= critical + THRESHOLD_EPS:
                status = "ALERT"
            else:
                status = "CRITICAL"
            band_scores.append(status)
            # print(f"  Band {s_idx:<2}  {err:>12.6f}   {status}")

        # Define criticality order (lower number = more critical)
        criticality_order = {"CRITICAL": 3,
                             "ALERT": 2, "OBSERVE": 1, "ACCEPTABLE": 0}

        # Find the most critical band
        worst_idx = max(range(len(band_scores)),
                        key=lambda i: criticality_order.get(band_scores[i], 999))
        worst_status = band_scores[worst_idx]
        reconstruction_error = errors[worst_idx]
        worst_band_num = worst_idx
        # print(f"  Worst Band: Band {worst_band_num}: {worst_status}")
        return {worst_status: [worst_band_num, reconstruction_error]}

    def evaluate_signal(self, test_signal_path):
        """
        Evaluate a single signal for anomalies.

        Complete pipeline: load signal, compute FFT, extract bands, predict errors,
        assess against thresholds, and return the overall assessment.

        Parameters
        ----------
        test_signal_path : str
            Path to the CSV file containing the test signal.

        Returns
        -------
        dict or None
            Assessment result with status and band number, or ``None`` if
            threshold data is unavailable.
        """
        signal, fs = self.load_single_signal(test_signal_path)
        signal_to_bands = SignalToBands(signal, self.speed_range, fs)
        bands = signal_to_bands.band_vectors
        errors = self.predict_errors(bands)
        results = self.assess_state(errors)
        return results


if __name__ == "__main__":
    demo = True
    train_signals_folder = "train_signals_folder"
    model_save_path = "autoencoder_model"
    speed_range = [500, 700]
    sample_train = AutoEncoderTrainer(train_signals_folder,
                                      model_save_path, demo, speed_range)
    sample_train.train_system()
    test_signal_path = "signal_1.csv"
    for test_signal_path in os.listdir(train_signals_folder):
        # if test_signal_path.endswith(".csv"):
        test_signal_path = os.path.join(train_signals_folder, test_signal_path)
        sample_evaluate = AutoEncoderEvaluator(model_save_path, speed_range)
        results = sample_evaluate.evaluate_signal(test_signal_path)
        print(test_signal_path, ':', results)
