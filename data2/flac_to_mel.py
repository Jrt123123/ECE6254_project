import os
import numpy as np
import librosa
import soundfile as sf

# === HYPERPARAMETERS (from hparams.py) ===
sample_rate = 22050
n_fft = 2048
num_mels = 80
hop_length = 275
win_length = 1100
fmin = 40
min_level_db = -100
ref_level_db = 20

# === INPUT DIRECTORY (CHANGE THIS) ===
flac_dir = 'E:/Georgia_Tech/ECE6254_Statistical_Machine_Learning/project/models/waveRNN_2/WaveRNN_2/WaveRNN-master/data2/wavs'

# === PROCESSING ===

def load_wav(path, sr):
    wav, file_sr = sf.read(path)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)  # Convert stereo to mono
    if file_sr != sr:
        wav = librosa.resample(wav, orig_sr=file_sr, target_sr=sr)
    return wav

def wav_to_mel(wav):
    mel = librosa.feature.melspectrogram(
        y=wav,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=num_mels,
        fmin=fmin
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_normalized = np.clip((mel_db - ref_level_db - min_level_db) / -min_level_db, 0, 1)
    mel_normalized = mel_normalized * 8 - 4  # Map to [-4, 4]
    return mel_normalized.T  # Shape: (T, 80)

def convert_all_flacs(flac_dir):
    for filename in os.listdir(flac_dir):
        if filename.endswith('.flac'):
            in_path = os.path.join(flac_dir, filename)
            out_path = os.path.splitext(in_path)[0] + '.npy'
            print(f'Processing {filename}...')

            wav = load_wav(in_path, sample_rate)
            mel = wav_to_mel(wav)
            np.save(out_path, mel.astype(np.float32))

            print(f'Saved mel to {out_path} | shape: {mel.shape}')

# Run it
convert_all_flacs(flac_dir)
