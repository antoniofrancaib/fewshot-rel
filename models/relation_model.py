"""
relation_model.py

This module defines two models:
1. RelationClassifierCAVIA: Uses a frozen DistilBERT encoder and a CAVIA-style classification head with context adaptation.
2. RelationClassifierBaseline: A baseline classifier that fine-tunes DistilBERT (or only its classifier) on few-shot data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DistilBertModel, BertModel
class RelationClassifierCAVIA(nn.Module):
    def __init__(self, context_dim, num_classes, pretrained_model="bert-base-uncased", unfreeze_layers=0):
        """
        Initializes the CAVIA-based relation classifier with a two-layer MLP
        and layer normalization in the classification head.
        
        Args:
            context_dim (int): Dimension of the context vector φ.
            num_classes (int): Number of classes in the episode.
            pretrained_model (str): Pretrained transformer model name (e.g., "bert-base-uncased").
            unfreeze_layers (int): Number of top transformer layers to unfreeze (0 means all frozen).
        """
        super().__init__()
        # Load BERT encoder
        self.encoder = BertModel.from_pretrained(pretrained_model)
        
        # Freeze all parameters initially
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # Optionally unfreeze the last N layers of BERT
        if unfreeze_layers > 0:
            total_layers = len(self.encoder.encoder.layer)  # BERT has encoder.layer
            for layer_idx in range(total_layers - unfreeze_layers, total_layers):
                for param in self.encoder.encoder.layer[layer_idx].parameters():
                    param.requires_grad = True
        
        self.hidden_size = self.encoder.config.hidden_size  # typically 768
        self.context_dim = context_dim
        
        # Context parameter (task-specific adaptation)
        self.context = nn.Parameter(torch.zeros(context_dim), requires_grad=True)
        
        # A two-layer MLP with layer norm and dropout
        mlp_dim = 512  # can be tuned (e.g. 256, 512, 768, etc.)
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size + context_dim, mlp_dim),
            nn.ReLU(),
            nn.LayerNorm(mlp_dim),
            nn.Dropout(p=0.1),  # optional dropout, can tune p
            nn.Linear(mlp_dim, num_classes)
        )
    
    def forward(self, input_ids, attention_mask, context_override=None):
        """
        Forward pass of the model.
        
        Args:
            input_ids (torch.Tensor): Token IDs of shape [batch_size, seq_length].
            attention_mask (torch.Tensor): Attention mask of shape [batch_size, seq_length].
            context_override (torch.Tensor, optional): If provided, overrides self.context.
            
        Returns:
            logits (torch.Tensor): Classification logits of shape [batch_size, num_classes].
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_embed = outputs.last_hidden_state[:, 0, :]  # [CLS] embedding
        
        # Use provided context_override if available, else self.context.
        context_vec = self.context if context_override is None else context_override
        
        # Expand context to match batch size and concatenate
        batch_size = cls_embed.size(0)
        context_expand = context_vec.expand(batch_size, self.context_dim)
        features = torch.cat([cls_embed, context_expand], dim=1)
        
        # Pass through the two-layer MLP with layer norm
        logits = self.classifier(features)
        return logits

class RelationClassifierBaseline(nn.Module):
    def __init__(self, num_classes, pretrained_model="distilbert-base-uncased", fine_tune_encoder=False):
        """
        Initializes the baseline relation classifier.
        
        Args:
            num_classes (int): Number of classes.
            pretrained_model (str): Pretrained transformer model name.
            fine_tune_encoder (bool): Whether to fine-tune DistilBERT encoder.
        """
        super().__init__()
        self.encoder = DistilBertModel.from_pretrained(pretrained_model)
        if not fine_tune_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        self.hidden_size = self.encoder.config.hidden_size
        
        self.classifier = nn.Linear(self.hidden_size, num_classes)
    
    def forward(self, input_ids, attention_mask):
        """
        Forward pass for the baseline.
        
        Args:
            input_ids: Tensor of input token ids.
            attention_mask: Attention mask tensor.
            
        Returns:
            logits: Classification logits.
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_embed = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_embed)
        return logits
