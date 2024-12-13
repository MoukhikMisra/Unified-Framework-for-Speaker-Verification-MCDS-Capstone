import os
import numpy as np
import pandas as pd
from espnet2.bin.spk_inference import Speech2Embedding

def extract_embedding(model, wav_path):
    waveform, _ = torchaudio.load(wav_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    embedding = model(waveform.numpy()[0])
    return embedding.squeeze()

def main(base_dir, output_csv):
    model = Speech2Embedding.from_pretrained(model_tag="espnet/voxcelebs12_rawnet3")

    rows = []
    for speaker_id in os.listdir(base_dir):
        speaker_dir = os.path.join(base_dir, speaker_id)
        if os.path.isdir(speaker_dir):
            for wav_file in os.listdir(speaker_dir):
                if wav_file.endswith(".wav"):
                    wav_path = os.path.join(speaker_dir, wav_file)
                    embedding = extract_embedding(model, wav_path)
                    row = [speaker_id, wav_file] + embedding.tolist()
                    rows.append(row)

    columns = ['speaker_id', 'file_name'] + [f'embedding_{i}' for i in range(len(rows[0]) - 2)]
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(output_csv, index=False)

if __name__ == "__main__":
    base_dir = "../vox1/vox1_dev_wav/wav"  
    output_csv = "rawnet_embeddings.csv"
    main(base_dir, output_csv)
