import os
import pandas as pd
import torchaudio
from speechbrain.pretrained import EncoderClassifier

# Define the base directory of the VoxCeleb1 dataset
base_dir = '../vox1/vox1_dev_wav/wav'

classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-xvect-voxceleb",
    savedir="pretrained_models/spkrec-xvect-voxceleb"
)

def process_wav_file(speaker_id, voicefile_id, wav_file_path):
    signal, fs = torchaudio.load(wav_file_path)
    if signal.shape[0] > 1:
        signal = signal.mean(dim=0, keepdim=True)
    embeddings = classifier.encode_batch(signal)
    embedding_list = embeddings.squeeze().tolist()
    return {
        'speaker_id': speaker_id,
        'voicefile_id': voicefile_id,
        'wav_file': wav_file_path,
        'embedding': embedding_list
    }


def main():
    rows = []

    for speaker_id in os.listdir(base_dir):
        speaker_dir = os.path.join(base_dir, speaker_id)
        if os.path.isdir(speaker_dir):
            for voicefile_id in os.listdir(speaker_dir):
                voicefile_dir = os.path.join(speaker_dir, voicefile_id)
                if os.path.isdir(voicefile_dir):
                    for wav_file in os.listdir(voicefile_dir):
                        if wav_file.endswith(".wav"):
                            wav_file_path = os.path.join(voicefile_dir, wav_file)
                            result = process_wav_file(speaker_id, voicefile_id, wav_file_path)
                            if result:
                                rows.append(result)

    df = pd.DataFrame(rows)
    df.to_csv('xvector_embeddings.csv', index=False)

if __name__ == "__main__":
    main()

