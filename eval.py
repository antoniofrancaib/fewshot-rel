"""
eval.py

This script evaluates a trained model (either CAVIA or baseline) on few-shot episodes.
For CAVIA, it adapts the context vector on the support set of each episode.
For the baseline, if --finetune_support is provided, it fine-tunes the model on the support set.
Usage examples:
    python eval.py --model_type cavia --model_checkpoint runs/cavia_experiment/best_model.pt --N 5 --K 5 --num_episodes 100
    python eval.py --model_type baseline --model_checkpoint runs/baseline_experiment/model.pt --N 5 --K 5 --finetune_support
"""

import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
from data.fewrel import FewRelDataset
from models.relation_model import RelationClassifierCAVIA, RelationClassifierBaseline
from utils import compute_accuracy, set_seed
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, choices=["cavia", "baseline"], required=True, help="Model type to evaluate.")
    parser.add_argument("--model_checkpoint", type=str, required=True, help="Path to the saved model checkpoint.")
    parser.add_argument("--N", type=int, default=5, help="N-way.")
    parser.add_argument("--K", type=int, default=5, help="K-shot.")
    parser.add_argument("--num_episodes", type=int, default=100, help="Number of episodes for evaluation.")
    parser.add_argument("--finetune_support", action="store_true", help="For baseline: fine-tune on support set.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()

def evaluate_episode(model, fewrel, N, K, query_size, device, inner_steps=5, inner_lr=0.001, model_type="cavia"):
    # Sample an episode.
    support_set, query_set = fewrel.sample_episode(N_way=N, K_shot=K, query_size=query_size)
    support_tokens = fewrel.tokenize_batch(support_set)
    query_tokens = fewrel.tokenize_batch(query_set)
    for key in support_tokens:
        support_tokens[key] = support_tokens[key].to(device)
    for key in query_tokens:
        query_tokens[key] = query_tokens[key].to(device)
    
    if model_type == "cavia":
        # For CAVIA, perform inner-loop adaptation on the support set.
        criterion = nn.CrossEntropyLoss()
        # Clone the context and perform adaptation.
        adapted_context = model.context.clone().detach().requires_grad_(True)
        optimizer_inner = optim.SGD([adapted_context], lr=inner_lr)
        for _ in range(inner_steps):
            optimizer_inner.zero_grad()
            logits = model(support_tokens["input_ids"], support_tokens["attention_mask"], context_override=adapted_context)
            loss = criterion(logits, torch.tensor(support_tokens["labels"]).to(device))
            loss.backward()
            optimizer_inner.step()
        # Evaluate on query set using adapted context.
        logits = model(query_tokens["input_ids"], query_tokens["attention_mask"], context_override=adapted_context)
    else:
        # For baseline evaluation.
        if getattr(args, "finetune_support", False):
            # Fine-tune on support set.
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=1e-5)
            for _ in range(100):  # Fine-tune for 100 iterations.
                optimizer.zero_grad()
                logits = model(support_tokens["input_ids"], support_tokens["attention_mask"])
                loss = criterion(logits, torch.tensor(support_tokens["labels"]).to(device))
                loss.backward()
                optimizer.step()
        logits = model(query_tokens["input_ids"], query_tokens["attention_mask"])
    query_labels = torch.tensor(query_tokens["labels"]).to(device)
    acc = compute_accuracy(logits, query_labels)
    return acc

if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    fewrel = FewRelDataset(split="validation")
    
    if args.model_type == "cavia":
        # Load CAVIA model.
        # Note: N_way must match the model's classifier output dimension.
        model = RelationClassifierCAVIA(context_dim=100, num_classes=args.N)
        model.load_state_dict(torch.load(args.model_checkpoint, map_location=device))
    else:
        # Load baseline model.
        model = RelationClassifierBaseline(num_classes=args.N, fine_tune_encoder=False)
        model.load_state_dict(torch.load(args.model_checkpoint, map_location=device))
    model.to(device)
    model.eval()
    
    total_acc = 0.0
    for _ in tqdm(range(args.num_episodes), desc="Evaluating"):
        acc = evaluate_episode(model, fewrel, args.N, args.K, query_size=15, device=device, model_type=args.model_type)
        total_acc += acc
    avg_acc = total_acc / args.num_episodes
    print(f"Average accuracy over {args.num_episodes} episodes: {avg_acc:.2f}")
