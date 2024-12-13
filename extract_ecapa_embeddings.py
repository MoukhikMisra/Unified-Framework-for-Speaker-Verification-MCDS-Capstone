import os
import pandas as pd
import torchaudio
from speechbrain.pretrained import EncoderClassifier
import numpy as np


classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

base_dir = "../vox1/vox1_dev_wav/wav" 
model_name = "speechbrain/spkrec-ecapa-voxceleb"

output_file = "ecapa_embeddings.csv"


def process_wav_file(speaker_id, voicefile_id, wav_file_path):
    """Process a single WAV file and return results."""
    try:
        signal, fs = torchaudio.load(wav_file_path)
        embeddings = classifier.encode_batch(signal).squeeze().numpy()
        wav_file_name = os.path.basename(wav_file_path)
        print(f"Processing: {speaker_id}-{voicefile_id}-{wav_file_name}")
        return {
            "speaker_id": speaker_id,
            "voicefile_id": voicefile_id,
            "wav_file_name": wav_file_name,
            "embedding": embeddings.tolist(), 
            "model_name": model_name
        }
    except Exception as e:
        print(f"Error processing {wav_file_path}: {e}")
        return None


def main():
    rows = []

    file_paths = []
    for speaker_id in os.listdir(base_dir):
        speaker_dir = os.path.join(base_dir, speaker_id)
        if os.path.isdir(speaker_dir):
            for voicefile_id in os.listdir(speaker_dir):
                voicefile_dir = os.path.join(speaker_dir, voicefile_id)
                if os.path.isdir(voicefile_dir):
                    for wav_file in os.listdir(voicefile_dir):
                        if wav_file.endswith(".wav"):
                            wav_file_path = os.path.join(voicefile_dir, wav_file)
                            file_paths.append((speaker_id, voicefile_id, wav_file_path))

    for speaker_id, voicefile_id, wav_file_path in file_paths:
        result = process_wav_file(speaker_id, voicefile_id, wav_file_path)
        if result:
            rows.append(result)
    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False)
    print(f"Saved {len(rows)} entries to {output_file}")


if __name__ == "__main__":
    main()
