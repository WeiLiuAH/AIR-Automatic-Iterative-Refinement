import json
import argparse
import os
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import torch

# Constants
BATCH_SIZE = 32
MODEL_NAME = 'all-MiniLM-L6-v2'

def preprocess_text(text: str):
    text = text.strip()
    assert len(text) != 0
    return text

def calculate_embeddings(samples, model, fout):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    for start_idx in tqdm(range(0, len(samples), BATCH_SIZE)):
        sample_chunk = samples[start_idx: start_idx + BATCH_SIZE]
        texts = list(map(preprocess_text, sample_chunk))
        embeddings_chunk = model.encode(texts, convert_to_tensor=True)
        embeddings_chunk = embeddings_chunk.cpu().numpy().tolist()
        
        for embedding in embeddings_chunk:
            fout.write(json.dumps(embedding)+"\n")

def main(args):
    model = SentenceTransformer(MODEL_NAME)

    with open(args.input_path, "r") as f:
        data1 = [json.loads(line) for line in f]
        samples = [i["document"] for i in data1]
    print(len(samples))
    
    with open(args.output_path, "w", encoding="utf-8") as fout:
        calculate_embeddings(samples, model, fout)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    main(vars(args))
