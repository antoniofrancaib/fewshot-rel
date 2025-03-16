"""
fewrel.py

This module loads the FewRel dataset from local JSON files and provides an episode sampler for few-shot relation classification.
"""

import random
import json
import os
from transformers import DistilBertTokenizer
import torch

class FewRelDataset:
    def __init__(self, split="train", tokenizer_name="distilbert-base-uncased", seed=42):
        """
        Initializes the FewRel dataset.
        
        Args:
            split (str): One of "train", "validation", "test".
            tokenizer_name (str): Name of the tokenizer to use.
            seed (int): Random seed for reproducibility.
        """
        # Load from local JSON files
        file_path = os.path.join(os.path.dirname(__file__), 
                               "train_wiki.json" if split == "train" else "val_wiki.json")
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        self.tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_name)
        self.seed = seed
        random.seed(seed)
        
        # Organize data by relation label
        self.data_by_relation = {}
        for rel, examples in data.items():
            self.data_by_relation[rel] = []
            for example in examples:
                # Format the sentence with entity markers
                tokens = example['tokens'].copy()  # Create a copy to avoid modifying original
                
                # Extract head and tail entity positions
                h_pos = example['h'][2]  # List of position lists for head entity
                t_pos = example['t'][2]  # List of position lists for tail entity
                
                # Get start and end positions for both entities
                h_start = min(pos[0] for pos in h_pos)
                h_end = max(pos[-1] for pos in h_pos)
                t_start = min(pos[0] for pos in t_pos)
                t_end = max(pos[-1] for pos in t_pos)
                
                # Insert entity markers
                if h_start < t_start:
                    tokens.insert(t_end + 1, '</e2>')
                    tokens.insert(t_start, '<e2>')
                    tokens.insert(h_end + 1, '</e1>')
                    tokens.insert(h_start, '<e1>')
                else:
                    tokens.insert(h_end + 1, '</e1>')
                    tokens.insert(h_start, '<e1>')
                    tokens.insert(t_end + 1, '</e2>')
                    tokens.insert(t_start, '<e2>')
                
                sentence = ' '.join(tokens)
                self.data_by_relation[rel].append(sentence)
        
        self.relations = list(self.data_by_relation.keys())
    
    def sample_episode(self, N_way=5, K_shot=5, query_size=15):
        """
        Samples an episode for N-way K-shot learning.
        
        Returns:
            support_set: List of dicts with keys 'sentence' and 'label'
            query_set: List of dicts with keys 'sentence' and 'label'
        """
        selected_relations = random.sample(self.relations, N_way)
        support_set = []
        query_set = []
        for idx, rel in enumerate(selected_relations):
            sentences = self.data_by_relation[rel]
            # Randomly sample without replacement
            samples = random.sample(sentences, K_shot + query_size)
            support_sentences = samples[:K_shot]
            query_sentences = samples[K_shot:]
            for s in support_sentences:
                support_set.append({"sentence": s, "label": idx})
            for s in query_sentences:
                query_set.append({"sentence": s, "label": idx})
        return support_set, query_set

    def tokenize_batch(self, examples, max_length=128):
        """
        Tokenizes a list of examples.
        
        Args:
            examples: List of dicts with key 'sentence'
            max_length (int): Maximum sequence length
        
        Returns:
            A dict with 'input_ids', 'attention_mask', and 'labels' as PyTorch tensors.
        """
        sentences = [ex["sentence"] for ex in examples]
        tokens = self.tokenizer(sentences, truncation=True, padding="max_length", max_length=max_length, return_tensors="pt")
        labels = torch.tensor([ex["label"] for ex in examples], dtype=torch.long)
        tokens["labels"] = labels
        return tokens
