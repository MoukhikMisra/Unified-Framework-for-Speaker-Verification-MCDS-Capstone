import numpy as np
import pandas as pd
import faiss
import time 

def create_faiss_index(csv_path, model_name, output_index_path, nlist=100, nprobe=10, k=5):
    print(f"\nProcessing {model_name} embeddings...")
    
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")
    
    metadata = df[['speaker_id', 'voicefile_id', 'wav_file']]
    embeddings = np.ascontiguousarray(
        df['embedding'].apply(eval).tolist(), dtype='float32'
    )  
    
    dimension = embeddings.shape[1]  
    
    faiss.normalize_L2(embeddings)

    quantizer = faiss.IndexFlatIP(dimension)
    ivf_index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
    
    print("Training IVF index...")
    start_time = time.time()
    ivf_index.train(embeddings)
    train_time = time.time() - start_time
    print(f"Training time: {train_time:.6f} seconds")
    
    print("Adding embeddings to IVF index...")
    start_time = time.time()
    ivf_index.add(embeddings)
    add_time = time.time() - start_time
    print(f"Adding time: {add_time:.6f} seconds")
    
    ivf_index.nprobe = nprobe
    
    print(f"Saving IVF index to {output_index_path}...")
    faiss.write_index(ivf_index, output_index_path)
    
    print("\nTesting IVF index...")
    query_vector = embeddings[0].reshape(1, -1)  # Use the first embedding as query
    start_time = time.time()
    D_ivf, I_ivf = ivf_index.search(query_vector, k)
    search_time = time.time() - start_time
    

    nearest_neighbors = metadata.iloc[I_ivf[0]].reset_index(drop=True)
    print(f"Search latency: {search_time:.6f} seconds")
    print("Nearest neighbors metadata:")
    print(nearest_neighbors)
    
    return {
        "metadata": nearest_neighbors,
        "timing": {
            "train_time": train_time,
            "add_time": add_time,
            "search_time": search_time,
        },
    }


model_csv_paths = {
    "RawNet": "rawnet_embeddings.csv",
    "Titanet": "titanet_embeddings.csv",
    "Vector": "xvector_embeddings.csv",
    "ECAPA": "ecapa_embeddings.csv",
}

output_indices = {}
for model, csv_path in model_csv_paths.items():
    output_path = f"{model.lower()}_ivf_index.faiss" 
    output_indices[model] = create_faiss_index(csv_path, model, output_path)


print("\nTiming Summary:")
for model, results in output_indices.items():
    print(f"{model} - Train: {results['timing']['train_time']:.6f}s, Add: {results['timing']['add_time']:.6f}s, Search: {results['timing']['search_time']:.6f}s")
