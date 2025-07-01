import safetensors.torch
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import os

def main():
    scores_path = "./influence_results/bias_pythia_410m/scores_ekfac_half/pairwise_scores.safetensors"

    if not os.path.exists(scores_path):
        print(f"File not found: {scores_path}")
        return

    print(f"Loading influence scores from {scores_path} ...")
    scores = safetensors.torch.load_file(scores_path)

    print(f"Loaded keys: {list(scores.keys())}")
    # If there's only one tensor, take it
    if len(scores) == 1:
        influence_matrix = list(scores.values())[0]
    else:
        # If multiple, adjust key name accordingly
        influence_matrix = scores.get('influence_scores') or list(scores.values())[0]

    print(f"Influence matrix shape: {influence_matrix.shape}")
    print("Sample data (first 5x5 block):")
    print(influence_matrix[:5, :5])

    # Plot heatmap
    print("Plotting heatmap ...")
    sns.heatmap(influence_matrix.cpu().to(torch.float32).numpy(), cmap="viridis")
    plt.title("Pairwise Influence Matrix Heatmap")
    plt.xlabel("Query sample index")
    plt.ylabel("Training sample index")
    plt.show()


    plt.figure(figsize=(10, 8))
    sns.heatmap(influence_matrix.cpu().to(torch.float32).numpy(), cmap="viridis")
    plt.title("Influence Scores Heatmap")
    plt.savefig("influence_heatmap.png")  # Save plot to file
    plt.close()  # Close the figure to free memory
    
if __name__ == "__main__":
    main()
