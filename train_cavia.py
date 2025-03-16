"""
train_cavia.py

This script performs meta-training using the CAVIA-based relation classifier on the FewRel dataset.
It follows an episodic training loop:
  - For each episode, sample an N-way K-shot task.
  - Adapt the context vector φ on the support set (inner loop).
  - Evaluate on the query set and update shared parameters (outer loop).

Usage:
    python train_cavia.py --config configs/cavia_config.json --output_dir runs/cavia_experiment
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from data.fewrel import FewRelDataset
from models.relation_model import RelationClassifierCAVIA
from utils import compute_accuracy, set_seed

import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/cavia_config.json", help="Path to config JSON file.")
    parser.add_argument("--output_dir", type=str, default="runs/cavia_experiment", help="Directory to save model checkpoints and logs.")
    return parser.parse_args()

def adapt_context(model, support_tokens, inner_steps, inner_lr, criterion):
    """
    Performs inner-loop adaptation on the support set by updating the context vector.
    
    Args:
        model: The CAVIA model.
        support_tokens: Tokenized support set (dict with input_ids, attention_mask, labels).
        inner_steps (int): Number of adaptation steps.
        inner_lr (float): Learning rate for inner-loop.
        criterion: Loss function.
    
    Returns:
        adapted_context: The updated context vector.
    """
    # Clone the initial context.
    adapted_context = model.context.clone().detach().requires_grad_(True)
    optimizer_inner = optim.SGD([adapted_context], lr=inner_lr)
    
    for _ in range(inner_steps):
        optimizer_inner.zero_grad()
        logits = model(support_tokens["input_ids"], support_tokens["attention_mask"], context_override=adapted_context)
        loss = criterion(logits, torch.tensor(support_tokens["labels"]).to(logits.device))
        loss.backward()
        optimizer_inner.step()
    return adapted_context.detach()

def main():
    args = parse_args()
    # Load configuration.
    with open(args.config, "r") as f:
        config = json.load(f)
    
    set_seed(config.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create output directory.
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize dataset.
    fewrel = FewRelDataset(split="train")
    
    N_way = config["N_way"]
    K_shot = config["K_shot"]
    query_size = config.get("query_size", 15)
    
    # Initialize model.
    model = RelationClassifierCAVIA(context_dim=config["context_dim"], num_classes=N_way)
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["outer_lr"])
    
    num_episodes = config["num_episodes"]
    inner_steps = config["inner_steps"]
    inner_lr = config["inner_lr"]
    meta_batch_size = config.get("meta_batch_size", 1)
    
    best_val_acc = 0.0
    
    # Meta-training loop.
    for episode in tqdm(range(num_episodes), desc="Meta-training"):
        optimizer.zero_grad()
        meta_loss = 0.0
        meta_acc = 0.0
        
        # For each meta-batch (here we use meta_batch_size=1 for simplicity).
        for _ in range(meta_batch_size):
            support_set, query_set = fewrel.sample_episode(N_way=N_way, K_shot=K_shot, query_size=query_size)
            support_tokens = fewrel.tokenize_batch(support_set)
            query_tokens = fewrel.tokenize_batch(query_set)
            
            # Move tensors to device.
            for key in support_tokens:
                support_tokens[key] = support_tokens[key].to(device)
            for key in query_tokens:
                query_tokens[key] = query_tokens[key].to(device)
            
            # Inner loop: adapt context on support set.
            adapted_context = adapt_context(model, support_tokens, inner_steps, inner_lr, criterion)
            
            # Outer loop: compute loss on query set using the adapted context.
            logits = model(query_tokens["input_ids"], query_tokens["attention_mask"], context_override=adapted_context)
            query_labels = torch.tensor(query_tokens["labels"]).to(device)
            loss = criterion(logits, query_labels)
            meta_loss += loss
            
            # Compute accuracy for reporting.
            acc = compute_accuracy(logits, query_labels)
            meta_acc += acc
        
        meta_loss = meta_loss / meta_batch_size
        meta_loss.backward()
        optimizer.step()
        
        if (episode + 1) % config.get("log_interval", 50) == 0:
            print(f"Episode {episode+1}/{num_episodes} - Meta Loss: {meta_loss.item():.4f}, Query Acc: {meta_acc/meta_batch_size:.2f}")
            # (Optional) Save a checkpoint if improved.
            if meta_acc/meta_batch_size > best_val_acc:
                best_val_acc = meta_acc/meta_batch_size
                torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pt"))
    
    # Save final model.
    torch.save(model.state_dict(), os.path.join(args.output_dir, "final_model.pt"))
    print("Meta-training complete.")

if __name__ == "__main__":
    main()
