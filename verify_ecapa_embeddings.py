import os
import pandas as pd
import numpy as np
import faiss
import torchaudio
from speechbrain.pretrained import EncoderClassifier

# Paths
index_path = "ecapa_ivf_index.faiss"       
metadata_csv_path = "ecapa_embeddings.csv" 
query_wav_path = "test.wav"               

index = faiss.read_index(index_path)
df = pd.read_csv(metadata_csv_path)
classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

def get_embedding_for_wav(wav_path):
    signal, fs = torchaudio.load(wav_path)
    embedding = classifier.encode_batch(signal).squeeze().numpy().reshape(1, -1).astype('float32')
    faiss.normalize_L2(embedding)
    return embedding

query_embedding = get_embedding_for_wav(query_wav_path)

k = 5
D, I = index.search(query_embedding, k)
nearest_neighbors = df.iloc[I[0]].reset_index(drop=True)

threshold = 0.7  
top_score = D[0][0]
if top_score < threshold:
    print("Speaker does not exist (no sufficiently close match).")
else:
    # If we have a good match, print out the top match info
    # Retrieve speaker_id or any other identifying metadata.
    matched_speaker_id = nearest_neighbors.loc[0, 'speaker_id']
    matched_voicefile_id = nearest_neighbors.loc[0, 'voicefile_id']
    matched_wav_file = nearest_neighbors.loc[0, 'wav_file_name'] if 'wav_file_name' in nearest_neighbors.columns else nearest_neighbors.loc[0, 'wav_file']
    print(f"Identified Speaker: {matched_speaker_id}")
    print(f"Voicefile ID: {matched_voicefile_id}")
    print(f"Nearest neighbor WAV file: {matched_wav_file}")
    print(f"Similarity Score: {top_score}")
