"""
eval.py

Evaluates a trained model (CAVIA or baseline) on few-shot episodes. For CAVIA,
it adapts the context vector on the support set of each episode. For the baseline,
if --finetune_support is provided, it fine-tunes on the support set.

Usage examples:
    python eval.py --model_type cavia \
                   --model_checkpoint runs/cavia_experiment/best_model.pt \
                   --N 5 \
                   --K 5 \
                   --num_episodes 100 \
                   --config configs/cavia_config.json

    python eval.py --model_type baseline \
                   --model_checkpoint runs/baseline_experiment/model.pt \
                   --N 5 \
                   --K 5 \
                   --finetune_support
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
    parser.add_argument("--model_type", type=str, choices=["cavia", "baseline"], required=True,
                        help="Model type to evaluate.")
    parser.add_argument("--model_checkpoint", type=str, required=True,
                        help="Path to the saved model checkpoint.")
    parser.add_argument("--N", type=int, default=5, help="N-way.")
    parser.add_argument("--K", type=int, default=5, help="K-shot.")
    parser.add_argument("--num_episodes", type=int, default=100,
                        help="Number of episodes for evaluation.")
    parser.add_argument("--finetune_support", action="store_true",
                        help="For baseline: fine-tune on support set.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    # Add this argument to load the same config used in training
    parser.add_argument("--config", type=str, default="configs/cavia_config.json",
                        help="Path to CAVIA config JSON (for hyperparams like inner_steps, etc.).")

    return parser.parse_args()

def evaluate_episode(model,
                     fewrel,
                     N,
                     K,
                     query_size,
                     device,
                     inner_steps=5,
                     inner_lr=0.001,
                     model_type="cavia",
                     finetune_support=False):
    """
    Evaluates one few-shot episode (support + query).
    For CAVIA: does inner-loop updates on the context vector.
    For baseline: optionally does fine-tuning if finetune_support=True.
    """
    # Sample an episode
    support_set, query_set = fewrel.sample_episode(N_way=N, K_shot=K, query_size=query_size)
    if not support_set or not query_set:
        # If we fail to sample an episode, skip it (returns 0.0 accuracy)
        return 0.0

    support_tokens = fewrel.tokenize_batch(support_set)
    query_tokens = fewrel.tokenize_batch(query_set)
    for key in support_tokens:
        support_tokens[key] = support_tokens[key].to(device)
    for key in query_tokens:
        query_tokens[key] = query_tokens[key].to(device)
    
    if model_type == "cavia":
        # Inner-loop adaptation on the support set
        criterion = nn.CrossEntropyLoss()
        adapted_context = model.context.clone().detach().requires_grad_(True)
        optimizer_inner = optim.SGD([adapted_context], lr=inner_lr)

        for _ in range(inner_steps):
            optimizer_inner.zero_grad()
            logits = model(support_tokens["input_ids"],
                           support_tokens["attention_mask"],
                           context_override=adapted_context)
            loss = criterion(logits, support_tokens["labels"].to(device))
            loss.backward()
            optimizer_inner.step()

        # Evaluate on query set using adapted context
        logits = model(query_tokens["input_ids"],
                       query_tokens["attention_mask"],
                       context_override=adapted_context)
    else:
        # Baseline evaluation
        if finetune_support:
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=1e-5)
            for _ in range(100):  # Fine-tune for 100 iterations
                optimizer.zero_grad()
                logits = model(support_tokens["input_ids"], support_tokens["attention_mask"])
                loss = criterion(logits, support_tokens["labels"].to(device))
                loss.backward()
                optimizer.step()
        logits = model(query_tokens["input_ids"], query_tokens["attention_mask"])

    query_labels = query_tokens["labels"].to(device)
    acc = compute_accuracy(logits, query_labels)
    return acc

if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load config for hyperparameters (if using CAVIA)
    with open(args.config, "r") as f:
        config = json.load(f)

    # Initialize dataset
    fewrel = FewRelDataset(split="validation", tokenizer_name="bert-base-uncased")
    
    # Load model
    if args.model_type == "cavia":
        # Use config-based unfreeze_layers, context_dim, etc.
        model = RelationClassifierCAVIA(
            context_dim=config["context_dim"],
            num_classes=args.N,
            pretrained_model="bert-base-uncased",
            unfreeze_layers=config["unfreeze_layers"]
        )
        model.load_state_dict(torch.load(args.model_checkpoint, map_location=device))
    else:
        # Baseline model
        model = RelationClassifierBaseline(num_classes=64, fine_tune_encoder=False)
        model.load_state_dict(torch.load(args.model_checkpoint, map_location=device))

    model.to(device)
    model.eval()
    
    total_acc = 0.0
    for _ in tqdm(range(args.num_episodes), desc="Evaluating"):
        acc = evaluate_episode(
            model,
            fewrel,
            args.N,
            args.K,
            query_size=15,
            device=device,
            inner_steps=config.get("inner_steps", 5),  # from config, default 5
            inner_lr=config.get("inner_lr", 0.001),    # from config, default 0.001
            model_type=args.model_type,
            finetune_support=args.finetune_support
        )
        total_acc += acc
    
    avg_acc = total_acc / args.num_episodes
    print(f"Average accuracy over {args.num_episodes} episodes: {avg_acc:.2f}")
