import os
import csv
import random
import math

def generate_signals(N):
    fs = 1024          # sampling frequency (Hz)
    duration = 1.0     # signal duration (seconds)
    n_samples = int(fs * duration)
    freqs = [10, 20, 30]   # Hz — peaks to include in every signal

    # --- create output folder ---
    folder = "TimeSignals"
    os.makedirs(folder, exist_ok=True)

    print(f"\nGenerating {N} signal(s) in folder '{folder}' ...")

    for i in range(1, N + 1):
        # random amplitude for each frequency component (0.7 to 1.3)
        amplitudes = [random.uniform(0.7, 1.3) for _ in freqs]

        filename = os.path.join(folder, f"signal_{i}.csv")

        with open(filename, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["time", "amplitude"])   # header

            for s in range(n_samples):
                t = s / fs
                # sum of sine waves at 10, 20, 30 Hz with random amplitudes
                amplitude = sum(
                    amplitudes[j] * math.sin(2 * math.pi * freqs[j] * t)
                    for j in range(len(freqs))
                )

                writer.writerow([round(t, 6), round(amplitude, 6)])

        print(f"  Saved: {filename}  "
              f"(A10={amplitudes[0]:.3f}, A20={amplitudes[1]:.3f}, A30={amplitudes[2]:.3f})")

    print(f"\nDone. {N} signal(s) saved to '{folder}/'.")
    print(f"Each file has {n_samples} rows (columns: time, amplitude).")
    print(f"Sampling rate: {fs} Hz | Duration: {duration} s")

if __name__ == "__main__":
    generate_signals(20)