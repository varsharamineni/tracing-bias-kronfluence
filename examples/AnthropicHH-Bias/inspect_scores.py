import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer

from examples.openwebtext.pipeline import (
    MODEL_NAME,
    get_custom_dataset,
    get_openwebtext_dataset,
)
from kronfluence.analyzer import Analyzer

from utils import get_anthropic_dataset, get_bias_agreement_dataset, get_bias_datasets

from task import (
    BiasTask,
    model,
    tokenizer
)


MODEL_NAME = "ncgc/pythia_410m_hh_full_sft_trainer"  # or path to your fine-tuned model



def main():
    # scores = Analyzer.load_file("influence_results/openwebtext/scores_raw/pairwise_scores.safetensors")[
    #     "all_modules"
    # ].float()
    scores = Analyzer.load_file("influence_results/ncgc/pythia_410m_hh_full_sft_trainer/scores_ekfac_half/pairwise_scores.safetensors"
)[
#    scores = Analyzer.load_file("influence_results/bias_pythia_410m/scores_ekfac_half/pairwise_scores.safetensors"
#)[
        "all_modules"
    ].float()

   
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True, trust_remote_code=True)
    train_dataset = get_anthropic_dataset(tokenizer)
    #eval_dataset = get_bias_agreement_dataset(tokenizer)
    eval_dataset, eval1_dataset = get_bias_datasets(tokenizer)

    
    eval_idx = 0
    sorted_scores = torch.sort(scores[eval_idx], descending=True)
    top_indices = sorted_scores.indices

    plt.plot(sorted_scores.values)
    plt.grid()
    plt.ylabel("IF Score")
    plt.show()

    print("Query Sequence:")
    print("Prompt:", eval_dataset.iloc[eval_idx]["prompt"])

    print("Top Influential Sequences:")
    for i in range(100):
        print("=" * 80)
        print(f"Rank = {i}; Score = {scores[eval_idx][int(top_indices[i])].item()}")
        print(tokenizer.decode(train_dataset[int(top_indices[i])]["input_ids"]))


if __name__ == "__main__":
    main()