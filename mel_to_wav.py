# import numpy as np
# import torch
# from models.fatchord_version import WaveRNN
# from utils import hparams as hp
# import os

# # === CONFIG ===
# hp.configure('hparams.py')  # Load hparams
# mel_file = 'data2/mel_spectro/mel/121-121726-0010.npy'  # Change this to your mel file
# output_wav = 'data2/wavs/reconstructed/121-121726-0010_reconstructed.wav'
# weights_path = 'quick_start/voc_weights/latest_weights.pyt'  # Or your path

# # === DEVICE ===
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # === LOAD MEL ===
# mel = np.load(mel_file)  # Shape: (T, 80)
# # if mel.shape[1] != hp.num_mels:
# #     raise ValueError(f'Expected mel shape (_, {hp.num_mels}) but got {mel.shape}')

# # Normalize mel to [0, 1] if not already
# if mel.min() < 0 or mel.max() > 1:
#     mel = (mel + 4) / 8  # assuming it's in [-4, 4]

# mel = torch.tensor(mel).unsqueeze(0).to(device)  # Shape: (1, T, 80)

# # === INIT WAVERNN ===
# voc_model = WaveRNN(rnn_dims=hp.voc_rnn_dims,
#                     fc_dims=hp.voc_fc_dims,
#                     bits=hp.bits,
#                     pad=hp.voc_pad,
#                     upsample_factors=hp.voc_upsample_factors,
#                     feat_dims=hp.num_mels,
#                     compute_dims=hp.voc_compute_dims,
#                     res_out_dims=hp.voc_res_out_dims,
#                     res_blocks=hp.voc_res_blocks,
#                     hop_length=hp.hop_length,
#                     sample_rate=hp.sample_rate,
#                     mode=hp.voc_mode).to(device)

# voc_model.load(weights_path)

# # === GENERATE AUDIO ===
# voc_model.generate(mel, output_wav, batched=True, target=hp.voc_target, overlap=hp.voc_overlap, mu_law=hp.mu_law)

# print(f'✅ Saved audio to {output_wav}')


import os
import numpy as np
import torch
from models.fatchord_version import WaveRNN
from utils import hparams as hp

# === CONFIG ===
hp.configure('hparams.py')
mel_folder = 'data2/mel_spectro/mel'
output_folder = 'data2/wavs/reconstructed'
weights_path = 'quick_start/voc_weights/latest_weights.pyt'

# === DEVICE ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === INIT WAVERNN ===
voc_model = WaveRNN(rnn_dims=hp.voc_rnn_dims,
                    fc_dims=hp.voc_fc_dims,
                    bits=hp.bits,
                    pad=hp.voc_pad,
                    upsample_factors=hp.voc_upsample_factors,
                    feat_dims=hp.num_mels,
                    compute_dims=hp.voc_compute_dims,
                    res_out_dims=hp.voc_res_out_dims,
                    res_blocks=hp.voc_res_blocks,
                    hop_length=hp.hop_length,
                    sample_rate=hp.sample_rate,
                    mode=hp.voc_mode).to(device)

voc_model.load(weights_path)

# === CREATE OUTPUT DIR IF NEEDED ===
os.makedirs(output_folder, exist_ok=True)

# === PROCESS EACH MEL FILE ===
for file_name in os.listdir(mel_folder):
    if not file_name.endswith('.npy'):
        continue

    mel_path = os.path.join(mel_folder, file_name)
    wav_path = os.path.join(output_folder, file_name.replace('.npy', '_reconstructed.wav'))

    print(f'🔄 Processing: {file_name}')

    mel = np.load(mel_path)
    if mel.min() < 0 or mel.max() > 1:
        mel = (mel + 4) / 8  # Normalize [-4, 4] → [0, 1]

    mel_tensor = torch.tensor(mel).unsqueeze(0).to(device)

    voc_model.generate(mel_tensor, wav_path, batched=True, target=hp.voc_target, overlap=hp.voc_overlap, mu_law=hp.mu_law)

    print(f'✅ Saved: {wav_path}')
