"""
train_baseline.py

This script trains a baseline relation classifier by fine-tuning DistilBERT on the FewRel dataset.
It performs standard supervised training on all 64 training relation classes.
Usage:
    python train_baseline.py --config configs/baseline_config.json --output_dir runs/baseline_experiment
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
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

def create_dataset(fewrel):
    """Create a dataset of all examples across all 64 relation classes."""
    all_input_ids = []
    all_attention_masks = []
    all_labels = []
    
    # Convert each relation to a numeric label (0-63) and collect all examples
    for label_id, relation in enumerate(fewrel.data_by_relation):
        sentences = fewrel.data_by_relation[relation]  # These are already formatted sentences
        # Create a list of dicts with sentence and label for tokenization
        examples = [{"sentence": sent, "label": label_id} for sent in sentences]
        tokens = fewrel.tokenize_batch(examples)
        
        all_input_ids.append(tokens["input_ids"])
        all_attention_masks.append(tokens["attention_mask"])
        all_labels.extend([label_id] * len(sentences))
    
    # Convert to tensors
    all_input_ids = torch.cat(all_input_ids)
    all_attention_masks = torch.cat(all_attention_masks)
    all_labels = torch.tensor(all_labels)
    
    return TensorDataset(all_input_ids, all_attention_masks, all_labels)

def main():
    args = parse_args()
    with open(args.config, "r") as f:
        config = json.load(f)
    
    set_seed(config.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load full training data
    fewrel = FewRelDataset(split="train")
    
    # Create dataset with all 64 classes
    dataset = create_dataset(fewrel)
    dataloader = DataLoader(
        dataset, 
        batch_size=config.get("batch_size", 16),
        shuffle=True,
        num_workers=0
    )
    
    # Initialize model for 64-way classification
    model = RelationClassifierBaseline(
        num_classes=64,
        fine_tune_encoder=config.get("fine_tune_encoder", False)
    )
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    
    num_epochs = config["num_epochs"]
    
    # Standard epoch-based training
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_acc = 0
        num_batches = 0
        
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            acc = compute_accuracy(logits, labels)
            total_loss += loss.item()
            total_acc += acc
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_acc = total_acc / num_batches
        print(f"Epoch {epoch+1}/{num_epochs} - Avg Loss: {avg_loss:.4f}, Avg Acc: {avg_acc:.2f}")
    
    torch.save(model.state_dict(), os.path.join(args.output_dir, "model.pt"))
    print("Baseline training complete.")

if __name__ == "__main__":
    main()
