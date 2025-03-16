# Few-Shot Relation Classification with DistilBERT and CAVIA

This project implements few-shot learning for relation classification using CAVIA (Context Adaptation Via Meta-Learning) and DistilBERT. The implementation focuses on the FewRel dataset, which contains 100 relation types with 700 instances each.

## Overview

### What is Few-Shot Relation Classification?
Few-shot relation classification is the task of classifying relations between entities in text when only a few examples of each relation type are available for training. For example, given just 5 examples of the "capital-of" relation, the model should learn to identify this relation in new sentences.

### Our Approach
We combine two powerful techniques:
1. **DistilBERT**: A lightweight, distilled version of BERT that serves as our sentence encoder
2. **CAVIA**: A meta-learning approach that introduces small, task-specific context parameters for fast adaptation

### Key Features
- Fast adaptation to new relations with minimal data (1-5 examples per relation)
- Efficient training by freezing DistilBERT and only updating context parameters
- Support for various N-way K-shot configurations
- Comprehensive evaluation and comparison with baseline approaches

## Installation

1. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset

The FewRel dataset is organized as follows:
- 64 relations for training
- 16 relations for validation
- 20 relations for testing
- Each relation has 700 labeled instances

The data files (`train_wiki.json` and `val_wiki.json`) should be placed in the `data/` directory.

## Running Experiments

### Training CAVIA Model

```bash
python train_cavia.py --config configs/cavia_config.json --output_dir runs/cavia_experiment1
```

Key configuration parameters in `cavia_config.json`:
- `N_way`: Number of relations per episode (e.g., 5)
- `K_shot`: Number of examples per relation (e.g., 5)
- `context_dim`: Dimension of context vector (e.g., 100)
- `num_episodes`: Total training episodes (e.g., 1000)
- `inner_lr`: Learning rate for context adaptation (e.g., 0.01)
- `outer_lr`: Learning rate for meta-learning (e.g., 0.001)

### Training Baseline Model

```bash
python train_baseline.py --config configs/baseline_config.json --output_dir runs/baseline_experiment1
```

The baseline model fine-tunes DistilBERT on the support set of each episode.

### Evaluation

Evaluate a trained model on few-shot episodes:

```bash
python eval.py --model_type cavia --model_checkpoint runs/cavia_experiment1/best_model.pt --N 5 --K 5 --num_episodes 100
```

For baseline model evaluation:
```bash
python eval.py --model_type baseline --model_checkpoint runs/baseline_experiment1/model.pt --N 5 --K 5 --finetune_support
```

## Understanding the Results

### Metrics
- **Accuracy**: Percentage of correctly classified query examples
- **Query Acc**: Performance on unseen examples after adaptation
- **Meta Loss**: Overall loss during meta-training

### Typical Results
- 5-way 1-shot: Expected accuracy ~40-50%
- 5-way 5-shot: Expected accuracy ~60-70%
- 10-way 5-shot: Expected accuracy ~50-60%

### Interpreting Output
During training, you'll see:
```
Episode [50/1000] - Meta Loss: 2.345 - Query Acc: 0.48
```
This shows:
- Current episode number
- Meta-learning loss
- Accuracy on query set after adaptation

## Project Structure
```
fewshot-rel/
├── data/            # Dataset and data loading utilities
├── models/          # Model implementations (CAVIA and baseline)
├── configs/         # Configuration files
├── train_cavia.py   # CAVIA training script
├── train_baseline.py# Baseline model training
├── eval.py         # Evaluation script
└── utils.py        # Utility functions
```

## Tips for Best Results

1. **GPU Usage**: Training on GPU is recommended. For CPU-only systems:
   - Reduce `num_episodes` to 100-200 for testing
   - Use smaller `context_dim` (e.g., 50)
   - Increase `log_interval` to see progress more frequently

2. **Hyperparameter Tuning**:
   - Start with small `N_way` and `K_shot` (e.g., 5-way 1-shot)
   - Adjust `inner_lr` if adaptation is unstable
   - Modify `context_dim` based on task complexity

3. **Common Issues**:
   - If accuracy is low, try increasing `K_shot`
   - If training is unstable, reduce `outer_lr`
   - If memory issues occur, reduce `meta_batch_size`

## License

MIT License 