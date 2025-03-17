"""
relation_model.py

This module defines two models:
1. RelationClassifierCAVIA: Uses a frozen DistilBERT encoder and a CAVIA-style classification head with context adaptation.
2. RelationClassifierBaseline: A baseline classifier that fine-tunes DistilBERT (or only its classifier) on few-shot data.
"""

import torch
import torch.nn as nn
from transformers import DistilBertModel

class RelationClassifierCAVIA(nn.Module):
    def __init__(self, context_dim, num_classes, pretrained_model="distilbert-base-uncased"):
        """
        Initializes the CAVIA-based relation classifier.
        
        Args:
            context_dim (int): Dimension of the context vector φ.
            num_classes (int): Number of classes in the episode.
            pretrained_model (str): Pretrained transformer model name.
        """
        super().__init__()
        # Load DistilBERT encoder and freeze parameters.
        self.encoder = DistilBertModel.from_pretrained(pretrained_model)
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.hidden_size = self.encoder.config.hidden_size  # typically 768
        
        self.context_dim = context_dim
        # Classification head: takes concatenated [CLS] embedding and context vector.
        self.classifier = nn.Linear(self.hidden_size + context_dim, num_classes)
        
        # Initialize context parameter (will be re-initialized for each episode)
        self.context = nn.Parameter(torch.zeros(context_dim), requires_grad=True)
    
    def forward(self, input_ids, attention_mask, context_override=None):
        """
        Forward pass of the model.
        
        Args:
            input_ids: Tensor of input token ids.
            attention_mask: Attention mask tensor.
            context_override: Optional tensor to override self.context.
            
        Returns:
            logits: Classification logits.
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_embed = outputs.last_hidden_state[:, 0, :]  # [CLS] embedding
        
        # Use provided context_override if available, else self.context.
        if context_override is None:
            context = self.context
        else:
            context = context_override
        
        # Expand context to batch size and concatenate.
        batch_size = cls_embed.size(0)
        context_expand = context.expand(batch_size, self.context_dim)
        features = torch.cat([cls_embed, context_expand], dim=1)
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
