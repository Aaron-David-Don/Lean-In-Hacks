import serial
import numpy as np
from scipy import signal
import pandas as pd
import time
import pickle
import asyncio

from collections import deque 

# Suppress sklearn warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def setup_filters(sampling_rate):
    b_notch, a_notch = signal.iirnotch(50.0 / (0.5 * sampling_rate), 30.0)
    b_bandpass, a_bandpass = signal.butter(
        4,
        [0.5 / (0.5 * sampling_rate), 30.0 / (0.5 * sampling_rate)],
        'band'
    )
    return b_notch, a_notch, b_bandpass, a_bandpass

def process_eeg_data(data, b_notch, a_notch, b_bandpass, a_bandpass):
    data = signal.filtfilt(b_notch, a_notch, data)
    data = signal.filtfilt(b_bandpass, a_bandpass, data)
    return data

def calculate_psd_features(segment, sampling_rate):
    f, psd_values = signal.welch(segment, fs=sampling_rate, nperseg=len(segment))
    bands = {
        'alpha': (8, 13),
        'beta': (14, 30),
        'theta': (4, 7),
        'delta': (0.5, 3)
    }
    features = {}
    for band, (low, high) in bands.items():
        idx = np.where((f >= low) & (f <= high))
        features[f'E_{band}'] = np.sum(psd_values[idx])
    features['alpha_beta_ratio'] = (
        features['E_alpha'] / features['E_beta']
        if features['E_beta'] > 0 else 0
    )
    return features

def calculate_additional_features(segment, sampling_rate):
    f, psd = signal.welch(segment, fs=sampling_rate, nperseg=len(segment))
    peak_frequency = f[np.argmax(psd)]
    spectral_centroid = np.sum(f * psd) / np.sum(psd)
    log_f = np.log(f[1:])
    log_psd = np.log(psd[1:])
    spectral_slope = np.polyfit(log_f, log_psd, 1)[0]
    return {
        'peak_frequency': peak_frequency,
        'spectral_centroid': spectral_centroid,
        'spectral_slope': spectral_slope
    }

def load_model_and_scaler():
    with open('modelhack.pkl', 'rb') as f:
        clf = pickle.load(f)
    with open('scalerhack.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return clf, scaler

async def main():
    ser = serial.Serial('COM3', 115200, timeout=1)
    clf, scaler = load_model_and_scaler()

    b_notch, a_notch, b_bandpass, a_bandpass = setup_filters(512)
    buffer = deque(maxlen=512)

    while True:
        try:
            raw_data = ser.readline().decode('latin-1').strip()

            # ✅ Ignore motor command echoes if any
            if not raw_data or raw_data.startswith('M'):
                continue

            eeg_value = float(raw_data)
            buffer.append(eeg_value)

            if len(buffer) == 512:
                buffer_array = np.array(buffer)

                processed_data = process_eeg_data(
                    buffer_array,
                    b_notch, a_notch,
                    b_bandpass, a_bandpass
                )

                psd_features = calculate_psd_features(processed_data, 512)
                additional_features = calculate_additional_features(processed_data, 512)
                features = {**psd_features, **additional_features}

                df = pd.DataFrame([features])
                X_scaled = scaler.transform(df)
                prediction = clf.predict(X_scaled)[0]

                if prediction != 2:
                    print("Calm State → Wheelchair STOP")
                    ser.write(b'M0')
                else:
                    print("Active State → Wheelchair FORWARD")
                    ser.write(b'M1')

                buffer.clear()

        except Exception as e:
            print(f'Error: {e}')
            continue

if __name__ == '__main__':
    asyncio.run(main())