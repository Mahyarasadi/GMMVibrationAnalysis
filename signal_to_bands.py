"""Convert time-domain vibration signals into fixed-length order bands.

The module computes a windowed single-sided FFT, estimates running speed from
dominant spectral content in a configured speed range, maps frequency to order
space, and interpolates predefined diagnostic bands onto a fixed-length grid.
"""

import numpy as np

N_POINTS = 64  # fixed size per band for speed-invariant order domain

# Bands based on:
# https://literature.rockwellautomation.com/idc/groups/literature/documents/at/1444-at001_-en-p.pdf
BANDS_LIST = [
    ["Band 0", 0.2, 0.8, "Bearing Cage Anomalies"],
    ["Band 1", 0.8, 1.2, "Unbalance, Misalignment"],
    ["Band 2", 1.2, 2.2, "Misalignment, Looseness"],
    ["Band 3", 2.2, 3.2, "Misalignment, Looseness"],
    ["Band 4", 3.2, 4.2, "Misalignment, Looseness"],
    ["Band 5", 4.2, 12.2, "Bearing Fundamental Frequencies"],
    ["Band 6", 12.2, 1000, "Bearing Lower Harmonic Frequencies"],
    ["Band 7", 1000, 2000, "Bearing/ Gear Higher Harmonics and Frequencies"]]


class SignalToBands:
    """
    Convert a time signal into order-domain diagnostic band vectors.

    Parameters
    ----------
    signal : array_like
        Input time-domain signal samples.
    speed_range : list[float]
        Expected machine speed range in RPM as ``[min_speed, max_speed]``.
    fs : float
        Sampling frequency in Hz.

    Attributes
    ----------
    freqs : ndarray
        FFT frequency bins in Hz.
    fft_amps : ndarray
        Single-sided FFT amplitude spectrum.
    machine_speed : int
        Detected machine speed in RPM.
    machine_speed_hz : float
        Detected machine speed in Hz.
    order_axis : ndarray
        FFT frequency axis normalized by machine speed.
    band_vectors : list[ndarray]
        Interpolated amplitudes for each predefined diagnostic band.

    Notes
    -----
    The band boundaries are defined by ``BANDS_LIST``. Bands are resampled to a
    uniform number of points to make downstream comparison and model ingestion
    independent of the original machine speed.

    References
    ----------
    .. [1] Rockwell Automation, 1444-AT001A-EN-P.
    """

    def __init__(self, signal, speed_range, fs):
        self.signal = signal
        self.speed_range = speed_range
        self.fs = fs
        self.freqs, self.fft_amps = self.calc_fft()
        self.machine_speed = self.speed_finder()
        self.machine_speed_hz = (self.machine_speed / 60)
        self.order_axis = self.freqs / self.machine_speed_hz
        self.band_vectors = self.get_all_bands()

    def calc_fft(self):
        """
        Compute the single-sided FFT spectrum.

        Applies a Hann window, computes the real FFT, and scales amplitudes
        to approximate single-sided amplitude preservation.

        Returns
        -------
        freqs : ndarray
            Frequency bins in Hz.
        fft_amp : ndarray
            Scaled single-sided FFT amplitudes.
        """
        N = len(self.signal)
        window = np.hanning(N)
        windowed_signal = window * self.signal
        window_weight = np.sum(window) / N
        windowed_signal = windowed_signal / window_weight
        fft_amp = np.fft.rfft(windowed_signal)
        fft_amp = (np.sqrt(2) / N) * np.abs(fft_amp)
        fft_amp[0] = fft_amp[0] / np.sqrt(2)
        freqs = np.fft.rfftfreq(N, 1 / self.fs)
        return freqs, fft_amp

    def speed_finder(self):
        """
        Estimate the machine running speed from the FFT spectrum.

        The search is constrained to the user-provided speed range. The method
        identifies candidate peaks in that interval and refines the strongest
        peak estimate through parabolic interpolation.

        Returns
        -------
        int
            Detected machine speed in RPM.

        Notes
        -----
        The returned speed is clipped to at least 1 Hz equivalent
        (60 RPM) before conversion to integer RPM.
        """

        # speed range bandwidth in hz
        x_data = self.fft_amps
        min_bw, max_bw = [i / 60 for i in self.speed_range]
        frq_array = np.asarray(self.freqs)
        min_freq_range_idx = (np.abs(frq_array - min_bw)).argmin()
        max_freq_range_idx = (np.abs(frq_array - max_bw)).argmin()
        min_max_indexes = [min_freq_range_idx - 1, max_freq_range_idx + 1]
        min_max_indexes.sort()
        speed_search_segment = x_data[min_max_indexes[0]: min_max_indexes[1]]
        no_of_tries = 5
        peaks_indices, peaks_amp = self._peak_finder_indices(
            speed_search_segment, no_of_tries
        )
        peaks_indices.sort()
        peaks_indices = [min_max_indexes[0] + k for k in peaks_indices]
        no_of_tries = min(no_of_tries, len(peaks_indices))
        speed_idx = peaks_indices[0]
        for i in range(no_of_tries):
            if self.is_true_speed(peaks_indices[i]):
                speed_idx = peaks_indices[i]
                break
        max_amplitude_index = (np.asarray(speed_search_segment)).argmax()
        max_amplitude_index += min_max_indexes[0]
        # actual_freq, = self.freqs[max_amplitude_index]
        actual_freq = self.parabolic_interpolation(
            max_amplitude_index, self.freqs, self.fft_amps
        )[0]
        if actual_freq < 1:
            actual_freq = 1
        return (60 * actual_freq)

    @staticmethod
    def parabolic_interpolation(max_index, fft_frequencies, fft_amplitudes):
        """
        Estimate a peak location with parabolic interpolation.

        The interpolation uses the samples around a local maximum to obtain a
        sub-bin estimate of the peak frequency and amplitude.

        Parameters
        ----------
        max_index : int
            Index of the FFT local peak.
        fft_frequencies : array_like of float
            FFT frequency bins.
        fft_amplitudes : array_like of float
            FFT amplitudes.

        Returns
        -------
        peak_frequency : float
            Peak frequency in Hz.
        peak_amplitude : float
            Peak amplitude.

        Notes
        -----
        For background on the method, see
        https://mgasior.web.cern.ch/pap/FFT_resol_note.pdf.
        """

        if max_index == 0 or max_index == len(fft_amplitudes) - 1:
            return fft_frequencies[max_index], fft_amplitudes[max_index]
        x_prev = fft_frequencies[max_index - 1]
        x_max = fft_frequencies[max_index]
        x_next = fft_frequencies[max_index + 1]
        y_prev = fft_amplitudes[max_index - 1]
        y_max = fft_amplitudes[max_index]
        y_next = fft_amplitudes[max_index + 1]
        numerator = y_prev - y_next
        denominator = 2 * (y_prev - 2 * y_max + y_next)
        if denominator == 0:
            return x_max, y_max
        offset = numerator / denominator
        peak_frequency = x_max + offset
        peak_amplitude = y_max - (offset**2) * denominator / 4
        return peak_frequency, peak_amplitude

    @staticmethod
    def _peak_finder_indices(fft_amplitude, nbpeaks):
        """
        Return the indices and amplitudes of the strongest local peaks.

        Parameters
        ----------
        fft_amplitude : array_like
            FFT amplitude vector.
        nbpeaks : int
            Number of peaks to return.

        Returns
        -------
        peaks_indices : list[int]
            Indices of the detected peaks in descending amplitude order.
        peaks_amp : list[float]
            Peak amplitudes corresponding to ``peaks_indices``.

        Notes
        -----
        If the input is monotonic and no local maxima are found, the global
        maximum is returned as a fallback.
        """

        lastval = fft_amplitude[0]
        lastindex = 0
        dirpos = True
        peaks_amp = []
        for i in range(len(fft_amplitude)):
            currentval = fft_amplitude[i]
            if dirpos and currentval < lastval:
                peaks_amp.append(fft_amplitude[lastindex])
                dirpos = False
            if currentval > lastval:
                dirpos = True
            lastval = currentval
            lastindex = i
        peaks_amp = sorted(peaks_amp, reverse=True)[0:nbpeaks]
        if not peaks_amp:
            # if array is monotonic
            peaks_amp = [np.max(fft_amplitude)]
        peaks_indices = [list(fft_amplitude).index(value)
                         for value in peaks_amp]
        return peaks_indices, peaks_amp

    def is_true_speed(self, index):
        """
        Check whether a candidate speed peak is supported by harmonics.

        The method verifies whether the candidate frequency has detectable
        harmonic content near the second and third harmonics within predefined
        tolerance bands.

        Parameters
        ----------
        index : int
            Index of the candidate peak in the FFT spectrum.

        Returns
        -------
        bool
            ``True`` if the candidate peak is consistent with the harmonic
            checks, otherwise ``False``.

        Notes
        -----
        The harmonic validation approach is based on the discussion in
        US Patent US5115671A.

        References
        ----------
        .. [1] US Patent US5115671A.
        """
        freq_vector = self.freqs
        x_data = self.fft_amps
        checks = []
        harmonics_search = [2, 3]
        delta_frq = [1, 2]
        rage_frq = [2, 3]
        guess_speed = freq_vector[index]
        # First we check if there is a peak at 2X+-1Hz
        for idx, value in enumerate(harmonics_search):
            min_bw, max_bw = (
                value * guess_speed - rage_frq[idx],
                value * guess_speed + rage_frq[idx],
            )
            min_freq_range_idx = (np.abs(freq_vector - min_bw)).argmin()
            max_freq_range_idx = (np.abs(freq_vector - max_bw)).argmin()
            min_max_indexes = [min_freq_range_idx - 1, max_freq_range_idx - 1]
            min_max_indexes.sort()
            speed_search_segment = x_data[min_max_indexes[0]
                : min_max_indexes[1]]
            peaks_indices, peaks_amp = self._peak_finder_indices(
                speed_search_segment, 1
            )
            peaks_indices = [min_max_indexes[0] + k for k in peaks_indices]
            if peaks_indices:
                checks.append(
                    (freq_vector[peaks_indices[0]] >= min_bw + delta_frq[idx])
                    and ((freq_vector[peaks_indices[0]] <= max_bw - delta_frq[idx]))
                )
            else:
                checks.append(False)
        if all(checks):
            return True
        else:
            return False

    def extract_band(self, fmin_order, fmax_order):
        """
        Interpolate one order-domain band onto a fixed grid.

        Parameters
        ----------
        fmin_order : float
            Lower band bound in order units.
        fmax_order : float
            Upper band bound in order units.

        Returns
        -------
        ndarray
            Interpolated band amplitudes with shape ``(N_POINTS,)``.
        """
        # 1. Create a fixed uniform grid in order domain for this band
        fixed_order_grid = np.linspace(fmin_order, fmax_order, N_POINTS)
        # 2. Interpolate the amplitude onto this fixed grid
        band_amps = np.interp(fixed_order_grid, self.order_axis, self.fft_amps)
        return band_amps

    def get_all_bands(self):
        """
        Extract all predefined diagnostic bands.

        Returns
        -------
        list[ndarray]
            Band amplitude vectors ordered according to ``BANDS_LIST``.
            Each element has shape ``(N_POINTS,)``.
        """
        band_vector = []
        for idx, band in enumerate(BANDS_LIST):
            fmin_order = band[1]
            fmax_order = band[2]
            if idx == 6:
                fmax_order = band[2] / self.machine_speed_hz
            elif idx == 7:
                fmin_order = band[1] / self.machine_speed_hz
                fmax_order = band[2] / self.machine_speed_hz
            amps = self.extract_band(
                fmin_order, fmax_order)
            band_vector.append(amps)
        return band_vector


if __name__ == "__main__":
    file_path = "train_signals_folder\\signal_1.csv"
    x_data = np.genfromtxt(file_path, delimiter=",", skip_header=1)
    print(x_data.shape)
    fs = (1/x_data[1, 0])
    x_data = x_data[:, 1].tolist()
    band_amplitudes = SignalToBands(x_data, [500, 700], fs)
    for band_vector in band_amplitudes.band_vectors:
        print(band_vector)
