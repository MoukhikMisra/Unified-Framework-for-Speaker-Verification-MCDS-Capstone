import os
import numpy as np
import pandas as pd
import faiss
import torchaudio
from espnet2.bin.spk_inference import Speech2Embedding
from speechbrain.pretrained import EncoderClassifier
from nemo.collections.asr.models import EncDecSpeakerLabelModel
from nemo.collections.asr.parts.utils.audio_utils import load_audio

# Paths
index_path = "titanet_ivf_index.faiss"     
metadata_csv_path = "titanet_embeddings.csv" 
query_wav_path = "test.wav"              

print("Loading FAISS index...")
index = faiss.read_index(index_path)

print("Loading metadata...")
df = pd.read_csv(metadata_csv_path)

print("Loading Titanet model...")
model =  EncDecSpeakerLabelModel.from_pretrained(model_name="nvidia/speakerverification_en_titanet_large")

def extract_rawnet_embedding(wav_path, model):
    waveform, _ = torchaudio.load(wav_path)
    
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    embedding = model(waveform.numpy()[0])
    embedding = embedding.squeeze().astype('float32').reshape(1, -1)
    
    faiss.normalize_L2(embedding)
    
    return embedding

print(f"Processing query WAV file: {query_wav_path}")
query_embedding = extract_rawnet_embedding(query_wav_path, model)


k = 5 
D, I = index.search(query_embedding, k)

nearest_neighbors = df.iloc[I[0]].reset_index(drop=True)

threshold = 0.7  
top_score = D[0][0]

if top_score < threshold:
    print("Speaker does not exist (no sufficiently close match).")
else:
    matched_speaker_id = nearest_neighbors.loc[0, 'speaker_id']
    matched_file_name = nearest_neighbors.loc[0, 'file_name']
    
    print(f"Identified Speaker: {matched_speaker_id}")
    print(f"Nearest neighbor file: {matched_file_name}")
    print(f"Similarity Score: {top_score}")