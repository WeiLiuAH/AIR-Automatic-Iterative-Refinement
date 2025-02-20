import json
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import euclidean_distances
import argparse
np.random.seed(42)

SAMPLE_SIZE = 60000  # Number of samples to be selected

def load_data(input_path, embedding_path):
    with open(input_path, "r") as f:
        data = [json.loads(line) for line in f]

    with open(embedding_path, 'r', encoding='utf-8') as f:
        embeddings = [json.loads(line.strip()) for line in f]
    embeddings = np.array(embeddings)
    assert len(data) == len(embeddings)
    print(len(data))

    return data, embeddings

def select_diverse_samples(embeddings, n_samples=1000):
    """Select diverse samples using maximum-minimum distance strategy"""
    indices = np.arange(len(embeddings))
    
    # Select the first sample (can be random or first one)
    selected_indices = [0]
    remained_indices = indices[1:]
    
    # Initialize minimum distances
    minimumm_lengths = None
    
    # Iteratively select samples
    for _ in tqdm(range(n_samples - 1)):
        # Calculate distances between current selected sample and remaining samples
        current_distances = euclidean_distances(
            embeddings[selected_indices[-1:]], 
            embeddings[remained_indices]
        ).squeeze()
        
        # Update minimum distances
        if minimumm_lengths is None:
            minimumm_lengths = current_distances
        else:
            minimumm_lengths = np.minimum(minimumm_lengths, current_distances)
        
        # Select sample with maximum minimum distance
        selected_idx = minimumm_lengths.argmax()
        new_idx = remained_indices[selected_idx]
        
        # Update indices
        selected_indices.append(new_idx)
        remained_indices = np.delete(remained_indices, selected_idx)
        minimumm_lengths = np.delete(minimumm_lengths, selected_idx)
    
    return selected_indices

def save_selected_data(selected_indices, data, output_path):
    """Save the selected data"""
    with open(output_path, 'w', encoding='utf-8') as f:
        for idx in selected_indices:
            f.write(json.dumps(data[idx], ensure_ascii=False) + '\n')

def main():
    # Load data
    print("Loading data...")
    data, embeddings = load_data(args.input_path, args.embedding_path)
    
    # Select diverse samples
    print("Selecting diverse samples...")
    selected_indices = select_diverse_samples(embeddings, SAMPLE_SIZE)
    
    # Save results
    print("Saving results...")
    save_selected_data(selected_indices, data, args.output_path)
    
    print(f"Successfully selected and saved {SAMPLE_SIZE} samples!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--embedding_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()
    main(args)


