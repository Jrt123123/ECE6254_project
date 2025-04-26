import os
import soundfile as sf

# === CONFIG ===
input_dir = 'data2/flacs'  # Change to your folder with .flac files

def convert_all_flac_to_wav(input_dir):
    for filename in os.listdir(input_dir):
        if filename.lower().endswith('.flac'):
            flac_path = os.path.join(input_dir, filename)
            wav_path = os.path.splitext(flac_path)[0] + '.wav'

            print(f'🔄 Converting: {filename} -> {os.path.basename(wav_path)}')
            data, samplerate = sf.read(flac_path)
            sf.write(wav_path, data, samplerate)

    print('✅ All FLAC files converted to WAV.')

# Run
convert_all_flac_to_wav(input_dir)
