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
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # hides TF info & warnings, only shows errors

import warnings
warnings.filterwarnings("ignore")  # optional: hides *all* Python warnings

import json
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from data.fewrel import FewRelDataset
from models.relation_model import RelationClassifierCAVIA
from utils import compute_accuracy, set_seed
from sklearn.metrics import f1_score
import numpy as np
from datetime import datetime
import sys
import matplotlib.pyplot as plt
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/cavia_config.json",
                        help="Path to config JSON file.")
    parser.add_argument("--output_dir", type=str, default="runs/cavia_experiment",
                        help="Directory to save model checkpoints and logs.")
    return parser.parse_args()

def adapt_context(model, support_tokens, inner_steps, inner_lr, criterion, grad_clip=1.0):
    """
    Performs inner-loop adaptation on the support set by updating the context vector.
    
    Args:
        model: The CAVIA model.
        support_tokens: Tokenized support set (dict with input_ids, attention_mask, labels).
        inner_steps (int): Number of adaptation steps.
        inner_lr (float): Learning rate for inner-loop.
        criterion: Loss function.
        grad_clip (float): Gradient clipping value.
    
    Returns:
        adapted_context: The updated context vector.
    """
    # Clone the initial context
    adapted_context = model.context.clone().detach().requires_grad_(True)
    
    # Use Adam optimizer for inner loop
    optimizer_inner = optim.Adam([adapted_context], lr=inner_lr)
    
    for _ in range(inner_steps):
        optimizer_inner.zero_grad()
        logits = model(
            support_tokens["input_ids"],
            support_tokens["attention_mask"],
            context_override=adapted_context
        )
        loss = criterion(logits, support_tokens["labels"].to(logits.device))
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(adapted_context, grad_clip)
        
        optimizer_inner.step()
    
    return adapted_context.detach()

def setup_logging(output_dir):
    """Setup logging to both file and stdout"""
    logs_dir = os.path.join(output_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(logs_dir, f"training_log_{timestamp}.txt")
    
    class Logger:
        def __init__(self, filename):
            self.terminal = sys.stdout
            self.log = open(filename, "w")
            
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
            self.log.flush()
            
        def flush(self):
            self.terminal.flush()
            self.log.flush()
    
    sys.stdout = Logger(log_file)
    return log_file

def compute_f1_score(logits, labels, n_classes):
    """Compute macro F1-score"""
    predictions = torch.argmax(logits, dim=1).cpu().numpy()
    labels = labels.cpu().numpy()
    return f1_score(labels, predictions, average='macro', labels=range(n_classes))

def save_training_plots(losses, accuracies, output_dir, log_interval):
    """
    Create and save training plots.
    
    Args:
        losses (list): List of meta-training losses
        accuracies (list): List of query accuracies
        output_dir (str): Directory to save plots
        log_interval (int): Interval between logged episodes
    """
    plots_dir = os.path.join(output_dir, "logs")
    os.makedirs(plots_dir, exist_ok=True)
    
    episodes = [(i + 1) * log_interval for i in range(len(losses))]
    
    # Plot loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, losses, 'b-', label='Meta Loss')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.title('Meta-Training Loss over Episodes')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(plots_dir, 'loss_curve.png'))
    plt.close()
    
    # Plot accuracy curve
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, accuracies, 'g-', label='Query Accuracy')
    plt.xlabel('Episode')
    plt.ylabel('Accuracy')
    plt.title('Average Query Accuracy over Episodes')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(plots_dir, 'accuracy_curve.png'))
    plt.close()

def evaluate_model(model, dataset, N_way, K_shot, query_size, num_episodes,
                   device, config, criterion):
    """
    Evaluates the model on validation episodes.
    
    Args:
        model: The CAVIA model.
        dataset: FewRelDataset instance.
        N_way (int): Number of classes per episode.
        K_shot (int): Number of support examples per class.
        query_size (int): Number of query examples per class.
        num_episodes (int): Number of evaluation episodes.
        device: PyTorch device.
        config (dict): The loaded config with inner_steps, inner_lr, grad_clip, etc.
        criterion: Loss function.
    
    Returns:
        avg_acc: Average accuracy across episodes.
        avg_f1: Average F1 score across episodes.
    """
    model.eval()
    total_acc = 0.0
    total_f1 = 0.0
    
    with torch.no_grad():
        for _ in range(num_episodes):
            support_set, query_set = dataset.sample_episode(N_way=N_way, K_shot=K_shot, query_size=query_size)
            if not support_set or not query_set:
                continue
                
            support_tokens = dataset.tokenize_batch(support_set)
            query_tokens = dataset.tokenize_batch(query_set)
            
            # Move tensors to device
            for key in support_tokens:
                support_tokens[key] = support_tokens[key].to(device)
            for key in query_tokens:
                query_tokens[key] = query_tokens[key].to(device)
            
            # Adapt context on support set
            adapted_context = adapt_context(
                model,
                support_tokens,
                config["inner_steps"],
                config["inner_lr"],
                criterion,
                config["grad_clip"]
            )
            
            # Evaluate on query set
            logits = model(
                query_tokens["input_ids"],
                query_tokens["attention_mask"],
                context_override=adapted_context
            )
            query_labels = query_tokens["labels"].to(device)
            
            acc = compute_accuracy(logits, query_labels)
            f1 = compute_f1_score(logits, query_labels, N_way)
            
            total_acc += acc
            total_f1 += f1
    
    model.train()
    return total_acc / num_episodes, total_f1 / num_episodes

def main():
    args = parse_args()
    with open(args.config, "r") as f:
        config = json.load(f)
    
    set_seed(config.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    os.makedirs(args.output_dir, exist_ok=True)
    log_file = setup_logging(args.output_dir)
    print(f"Logging to: {log_file}")
    print(f"Configuration: {json.dumps(config, indent=2)}")
    print(f"Device: {device}")
    
    # Initialize datasets
    train_dataset = FewRelDataset(split="train")
    val_dataset = FewRelDataset(split="validation")
    
    N_way = config["N_way"]
    K_shot = config["K_shot"]
    query_size = config.get("query_size", 15)
    
    # Initialize model
    model = RelationClassifierCAVIA(
        context_dim=config["context_dim"],
        num_classes=N_way,
        pretrained_model="bert-base-uncased",
        unfreeze_layers=config["unfreeze_layers"]
    )
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["outer_lr"])
    
    num_episodes = config["num_episodes"]
    inner_steps = config["inner_steps"]
    inner_lr = config["inner_lr"]
    meta_batch_size = config.get("meta_batch_size", 1)
    log_interval = config.get("log_interval", 50)
    
    best_val_acc = 0.0
    best_val_f1 = 0.0
    patience_counter = 0
    
    # Lists to store metrics for plotting
    losses = []
    accuracies = []
    
    # Meta-training loop
    for episode in tqdm(range(num_episodes), desc="Meta-training"):
        optimizer.zero_grad()
        meta_loss = 0.0
        meta_acc = 0.0
        meta_f1 = 0.0
        
        # For each meta-batch
        for _ in range(meta_batch_size):
            support_set, query_set = train_dataset.sample_episode(
                N_way=N_way, K_shot=K_shot, query_size=query_size
            )
            if not support_set or not query_set:
                continue
                
            support_tokens = train_dataset.tokenize_batch(support_set)
            query_tokens = train_dataset.tokenize_batch(query_set)
            
            # Move tensors to device
            for key in support_tokens:
                support_tokens[key] = support_tokens[key].to(device)
            for key in query_tokens:
                query_tokens[key] = query_tokens[key].to(device)
            
            # Inner loop: adapt context on support set
            adapted_context = adapt_context(
                model,
                support_tokens,
                inner_steps,
                inner_lr,
                criterion,
                config["grad_clip"]
            )
            
            # Outer loop: compute loss on query set using the adapted context
            logits = model(
                query_tokens["input_ids"],
                query_tokens["attention_mask"],
                context_override=adapted_context
            )
            query_labels = query_tokens["labels"].to(device)
            loss = criterion(logits, query_labels)
            meta_loss += loss
            
            # Compute metrics
            acc = compute_accuracy(logits, query_labels)
            f1 = compute_f1_score(logits, query_labels, N_way)
            meta_acc += acc
            meta_f1 += f1
        
        if meta_batch_size > 0:
            meta_loss = meta_loss / meta_batch_size
        
        meta_loss.backward()
        
        # Gradient clipping for outer loop
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
        
        optimizer.step()
        
        # Logging
        if (episode + 1) % log_interval == 0:
            avg_acc = meta_acc / meta_batch_size if meta_batch_size > 0 else 0
            avg_f1 = meta_f1 / meta_batch_size if meta_batch_size > 0 else 0
            log_msg = (
                f"Episode {episode+1}/{num_episodes} - "
                f"Meta Loss: {meta_loss.item():.4f}, "
                f"Query Acc: {avg_acc:.2f}, "
                f"Macro F1: {avg_f1:.2f}"
            )
            print(log_msg)
            
            # Store metrics for plotting
            losses.append(meta_loss.item())
            accuracies.append(avg_acc)
            
            # Evaluate on validation set
            val_acc, val_f1 = evaluate_model(
                model,
                val_dataset,
                N_way,
                K_shot,
                query_size,
                config["validation_episodes"],
                device,
                config,    # pass config
                criterion  # pass criterion
            )
            print(f"Validation - Acc: {val_acc:.2f}, F1: {val_f1:.2f}")
            
            # Early stopping check
            if val_acc > best_val_acc + config["min_delta"]:
                best_val_acc = val_acc
                best_val_f1 = val_f1
                print(f"New best model! Acc: {best_val_acc:.2f}, F1: {best_val_f1:.2f}")
                torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pt"))
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= config["patience"]:
                    print(f"Early stopping triggered after {episode + 1} episodes")
                    break
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(args.output_dir, "final_model.pt"))
    print("\nTraining completed!")
    print(f"Best validation accuracy: {best_val_acc:.2f}")
    print(f"Best validation macro F1-score: {best_val_f1:.2f}")
    print(f"Log file saved at: {log_file}")
    
    # Generate and save plots
    save_training_plots(losses, accuracies, args.output_dir, log_interval)
    print("Training plots saved in the logs directory.")

if __name__ == "__main__":
    main()
