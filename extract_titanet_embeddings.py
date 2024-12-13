import os
import csv
import torch
from nemo.collections.asr.models import EncDecSpeakerLabelModel
from nemo.collections.asr.parts.utils.audio_utils import load_audio

model = EncDecSpeakerLabelModel.from_pretrained(model_name="nvidia/speakerverification_en_titanet_large")

base_dir = "../vox1/vox1_dev_wav/wav"

output_csv = "titanet_embeddings.csv"

def process_wav_file(speaker_id, voicefile_id, wav_file_path):
    try:
        signal = load_audio(wav_file_path, target_sr=model.sample_rate)
        signal = torch.tensor(signal).unsqueeze(0).to(model.device)

        embedding = model.forward(input_signal=signal, input_signal_length=torch.tensor([signal.shape[1]])).squeeze().cpu().detach().numpy()

        result = {
            "speaker_id": speaker_id,
            "voicefile_id": voicefile_id,
            "wav_file_name": os.path.basename(wav_file_path),
            "embedding": embedding.tolist()
        }
        return result
    except Exception as e:
        print(f"Error processing {wav_file_path}: {e}")
        return None

def main():
    rows = []

    # Collect all file paths to process
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

    with open(output_csv, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["speaker_id", "voicefile_id", "wav_file_name", "embedding"])
        writer.writeheader()
        for row in rows:
            row["embedding"] = " ".join(map(str, row["embedding"]))
            writer.writerow(row)

    print(f"Embeddings saved to {output_csv}")

if __name__ == "__main__":
    main()
