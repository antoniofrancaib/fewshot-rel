"""
train_baseline.py

This script trains a baseline relation classifier by fine-tuning DistilBERT on the FewRel dataset.
It performs standard supervised training (on all training classes) and then adapts on few-shot episodes.
Usage:
    python train_baseline.py --config configs/baseline_config.json --output_dir runs/baseline_experiment
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from data.fewrel import FewRelDataset
from models.relation_model import RelationClassifierBaseline
from utils import compute_accuracy, set_seed

import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/baseline_config.json", help="Path to config JSON file.")
    parser.add_argument("--output_dir", type=str, default="runs/baseline_experiment", help="Directory to save model checkpoints.")
    return parser.parse_args()

def main():
    args = parse_args()
    with open(args.config, "r") as f:
        config = json.load(f)
    
    set_seed(config.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load full training data.
    fewrel = FewRelDataset(split="train")
    
    # For baseline training, we assume a fixed number of classes equal to N_way in the config.
    N_way = config["N_way"]
    model = RelationClassifierBaseline(num_classes=N_way, fine_tune_encoder=config.get("fine_tune_encoder", False))
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    
    num_epochs = config["num_epochs"]
    batch_size = config.get("batch_size", 16)
    
    # For simplicity, we sample episodes and treat all support examples as training data.
    for epoch in range(num_epochs):
        support_set, _ = fewrel.sample_episode(N_way=N_way, K_shot=config["K_shot"], query_size=0)
        tokens = fewrel.tokenize_batch(support_set)
        for key in tokens:
            tokens[key] = tokens[key].to(device)
        optimizer.zero_grad()
        logits = model(tokens["input_ids"], tokens["attention_mask"])
        labels = torch.tensor(tokens["labels"]).to(device)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        acc = compute_accuracy(logits, labels)
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {loss.item():.4f}, Acc: {acc:.2f}")
    
    torch.save(model.state_dict(), os.path.join(args.output_dir, "model.pt"))
    print("Baseline training complete.")

if __name__ == "__main__":
    main()
