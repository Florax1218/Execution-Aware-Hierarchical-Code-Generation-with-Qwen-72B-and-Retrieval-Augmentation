"""
Dual Encoder Architecture for CodeForge-Quantum
Processes natural language descriptions and pseudocode through separate pathways
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from dataclasses import dataclass


@dataclass
class DualEncoderConfig:
    """Configuration for dual encoder"""
    hidden_dim: int = 8192
    nl_hidden_dim: int = 8192
    pc_hidden_dim: int = 8192
    num_heads: int = 64
    num_layers: int = 40  # Split between NL and PC
    dropout: float = 0.1
    max_seq_length: int = 2048
    vocab_size: int = 65536
    fusion_method: str = "adaptive"  # adaptive, concat, average, attention


class NaturalLanguageEncoder(nn.Module):
    """Encoder for natural language problem descriptions"""
    
    def __init__(self, config: DualEncoderConfig):
        super().__init__()
        self.config = config
        
        # Embedding layer
        self.embedding = nn.Embedding(
            config.vocab_size,
            config.nl_hidden_dim
        )
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(
            config.nl_hidden_dim,
            config.max_seq_length
        )
        
        # Transformer layers
        self.layers = nn.ModuleList([
            NLTransformerLayer(config)
            for _ in range(config.num_layers // 2)
        ])
        
        # Layer-wise attention weights (learnable)
        self.layer_weights = nn.Parameter(
            torch.ones(config.num_layers // 2)
        )
        
        # Problem-specific components
        self.problem_type_classifier = nn.Linear(
            config.nl_hidden_dim, 10
        )
        
        self.difficulty_estimator = nn.Linear(
            config.nl_hidden_dim, 3
        )
        
        # Output projection
        self.output_projection = nn.Linear(
            config.nl_hidden_dim,
            config.hidden_dim
        )
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Encode natural language description"""
        
        # Embed and add positional encoding
        x = self.embedding(input_ids)
        x = self.positional_encoding(x)
        
        # Store layer outputs for weighted aggregation
        layer_outputs = []
        
        # Apply transformer layers
        for i, layer in enumerate(self.layers):
            x = layer(x, attention_mask)
            # Weight and store layer output
            weighted_output = self.layer_weights[i] * x
            layer_outputs.append(weighted_output)
        
        # Aggregate layer outputs
        aggregated = torch.stack(layer_outputs).sum(dim=0)
        
        # Normalize weights
        aggregated = aggregated / self.layer_weights.sum()
        
        # Pool for classification tasks
        pooled = aggregated.mean(dim=1)
        
        # Classify problem type
        problem_type = F.softmax(
            self.problem_type_classifier(pooled), dim=-1
        )
        
        # Estimate difficulty
        difficulty = F.softmax(
            self.difficulty_estimator(pooled), dim=-1
        )
        
        # Project to common dimension
        nl_encoding = self.output_projection(aggregated)
        
        return {
            'nl_encoding': nl_encoding,
            'nl_hidden': aggregated,
            'problem_type': problem_type,
            'difficulty': difficulty,
            'layer_outputs': layer_outputs
        }


class PseudocodeEncoder(nn.Module):
    """Encoder for pseudocode with syntax-aware processing"""
    
    def __init__(self, config: DualEncoderConfig):
        super().__init__()
        self.config = config
        
        # Embedding layer
        self.embedding = nn.Embedding(
            config.vocab_size,
            config.pc_hidden_dim
        )
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(
            config.pc_hidden_dim,
            config.max_seq_length
        )
        
        # Syntax-aware mask generator
        self.syntax_mask_generator = SyntaxMaskGenerator(config)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            PCTransformerLayer(config)
            for _ in range(config.num_layers // 2)
        ])
        
        # Code structure analyzer
        self.structure_analyzer = nn.Sequential(
            nn.Linear(config.pc_hidden_dim, config.pc_hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.pc_hidden_dim // 2, 20)  # Structure types
        )
        
        # Indentation encoder
        self.indent_encoder = nn.Embedding(10, config.pc_hidden_dim)
        
        # Output projection
        self.output_projection = nn.Linear(
            config.pc_hidden_dim,
            config.hidden_dim
        )
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        indentation_levels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Encode pseudocode with syntax awareness"""
        
        # Embed
        x = self.embedding(input_ids)
        x = self.positional_encoding(x)
        
        # Add indentation information if available
        if indentation_levels is not None:
            indent_embed = self.indent_encoder(indentation_levels)
            x = x + indent_embed
        
        # Generate syntax-aware mask
        syntax_mask = self.syntax_mask_generator(x, attention_mask)
        
        # Apply transformer layers with syntax mask
        for layer in self.layers:
            x = layer(x, syntax_mask)
        
        # Analyze code structure
        pooled = x.mean(dim=1)
        structure = self.structure_analyzer(pooled)
        structure_probs = F.softmax(structure, dim=-1)
        
        # Project to common dimension
        pc_encoding = self.output_projection(x)
        
        return {
            'pc_encoding': pc_encoding,
            'pc_hidden': x,
            'structure_types': structure_probs,
            'syntax_mask': syntax_mask
        }


class SyntaxMaskGenerator(nn.Module):
    """Generate syntax-aware attention masks"""
    
    def __init__(self, config: DualEncoderConfig):
        super().__init__()
        
        # Mask predictor
        self.mask_predictor = nn.Sequential(
            nn.Linear(config.pc_hidden_dim, config.pc_hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.pc_hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Block boundary detector
        self.boundary_detector = nn.Sequential(
            nn.Linear(config.pc_hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  # Start/End of block
        )
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        base_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Generate syntax-aware mask"""
        
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Predict mask values
        mask_scores = self.mask_predictor(hidden_states).squeeze(-1)
        
        # Detect block boundaries
        boundaries = self.boundary_detector(hidden_states)
        boundary_probs = F.softmax(boundaries, dim=-1)
        
        # Create syntax mask
        syntax_mask = torch.ones(batch_size, seq_len, seq_len)
        
        # Apply block-wise masking
        for b in range(batch_size):
            block_starts = torch.where(boundary_probs[b, :, 0] > 0.5)[0]
            block_ends = torch.where(boundary_probs[b, :, 1] > 0.5)[0]
            
            # Simple block masking (can be more sophisticated)
            for start, end in zip(block_starts, block_ends):
                if start < end:
                    # Mask within block
                    syntax_mask[b, start:end, start:end] = 2.0
        
        # Combine with base mask if provided
        if base_mask is not None:
            syntax_mask = syntax_mask * base_mask.unsqueeze(1)
        
        return syntax_mask


class AdaptiveFusion(nn.Module):
    """Adaptive fusion of NL and PC encodings"""
    
    def __init__(self, config: DualEncoderConfig):
        super().__init__()
        
        # Fusion gate
        self.fusion_gate = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 2),
            nn.Softmax(dim=-1)
        )
        
        # Cross-attention for fusion
        self.cross_attention = nn.MultiheadAttention(
            config.hidden_dim,
            config.num_heads // 2,
            dropout=config.dropout,
            batch_first=True
        )
        
        # Fusion projection
        self.fusion_projection = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )
        
    def forward(
        self,
        nl_encoding: torch.Tensor,
        pc_encoding: torch.Tensor
    ) -> torch.Tensor:
        """Adaptively fuse NL and PC encodings"""
        
        # Ensure same sequence length
        min_len = min(nl_encoding.size(1), pc_encoding.size(1))
        nl_encoding = nl_encoding[:, :min_len, :]
        pc_encoding = pc_encoding[:, :min_len, :]
        
        # Compute fusion weights
        concat_encodings = torch.cat([nl_encoding, pc_encoding], dim=-1)
        gate_weights = self.fusion_gate(concat_encodings.mean(dim=1))
        
        # Apply cross-attention
        nl_attended, _ = self.cross_attention(
            nl_encoding, pc_encoding, pc_encoding
        )
        pc_attended, _ = self.cross_attention(
            pc_encoding, nl_encoding, nl_encoding
        )
        
        # Weighted combination
        nl_weight = gate_weights[:, 0:1].unsqueeze(1)
        pc_weight = gate_weights[:, 1:2].unsqueeze(1)
        
        weighted_fusion = nl_weight * nl_attended + pc_weight * pc_attended
        
        # Project fused representation
        concat_fusion = torch.cat([weighted_fusion, concat_encodings], dim=-1)
        fused = self.fusion_projection(concat_fusion)
        
        return fused


class DualEncoder(nn.Module):
    """Complete dual encoder architecture"""
    
    def __init__(self, config: DualEncoderConfig):
        super().__init__()
        self.config = config
        
        # Natural language encoder
        self.nl_encoder = NaturalLanguageEncoder(config)
        
        # Pseudocode encoder
        self.pc_encoder = PseudocodeEncoder(config)
        
        # Fusion module
        if config.fusion_method == "adaptive":
            self.fusion = AdaptiveFusion(config)
        elif config.fusion_method == "attention":
            self.fusion = AttentionFusion(config)
        elif config.fusion_method == "concat":
            self.fusion = ConcatFusion(config)
        else:  # average
            self.fusion = AverageFusion(config)
        
        # Joint encoder (processes fused representation)
        self.joint_encoder = nn.ModuleList([
            JointTransformerLayer(config)
            for _ in range(4)  # Fewer layers for joint processing
        ])
        
        # Output heads
        self.semantic_head = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.syntactic_head = nn.Linear(config.hidden_dim, config.hidden_dim)
        
    def forward(
        self,
        nl_input_ids: torch.Tensor,
        nl_attention_mask: Optional[torch.Tensor] = None,
        pc_input_ids: Optional[torch.Tensor] = None,
        pc_attention_mask: Optional[torch.Tensor] = None,
        pc_indentation: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """Process through dual encoder"""
        
        # Encode natural language
        nl_output = self.nl_encoder(nl_input_ids, nl_attention_mask)
        
        # Encode pseudocode if provided
        if pc_input_ids is not None:
            pc_output = self.pc_encoder(
                pc_input_ids, pc_attention_mask, pc_indentation
            )
            
            # Fuse encodings
            fused = self.fusion(
                nl_output['nl_encoding'],
                pc_output['pc_encoding']
            )
        else:
            # Only NL encoding
            fused = nl_output['nl_encoding']
            pc_output = None
        
        # Joint encoding
        joint_hidden = fused
        for layer in self.joint_encoder:
            joint_hidden = layer(joint_hidden)
        
        # Generate semantic and syntactic representations
        semantic_repr = self.semantic_head(joint_hidden)
        syntactic_repr = self.syntactic_head(joint_hidden)
        
        return {
            'fused_encoding': fused,
            'joint_encoding': joint_hidden,
            'semantic_representation': semantic_repr,
            'syntactic_representation': syntactic_repr,
            'nl_output': nl_output,
            'pc_output': pc_output
        }


class NLTransformerLayer(nn.Module):
    """Transformer layer for natural language encoder"""
    
    def __init__(self, config: DualEncoderConfig):
        super().__init__()
        
        self.self_attention = nn.MultiheadAttention(
            config.nl_hidden_dim,
            config.num_heads,
            dropout=config.dropout,
            batch_first=True
        )
        
        self.ffn = nn.Sequential(
            nn.Linear(config.nl_hidden_dim, config.nl_hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.nl_hidden_dim * 4, config.nl_hidden_dim)
        )
        
        self.norm1 = nn.LayerNorm(config.nl_hidden_dim)
        self.norm2 = nn.LayerNorm(config.nl_hidden_dim)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Self-attention
        attn_out, _ = self.self_attention(x, x, x, key_padding_mask=attention_mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # FFN
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        
        return x


class PCTransformerLayer(nn.Module):
    """Transformer layer for pseudocode encoder"""
    
    def __init__(self, config: DualEncoderConfig):
        super().__init__()
        
        self.self_attention = nn.MultiheadAttention(
            config.pc_hidden_dim,
            config.num_heads,
            dropout=config.dropout,
            batch_first=True
        )
        
        self.ffn = nn.Sequential(
            nn.Linear(config.pc_hidden_dim, config.pc_hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.pc_hidden_dim * 4, config.pc_hidden_dim)
        )
        
        self.norm1 = nn.LayerNorm(config.pc_hidden_dim)
        self.norm2 = nn.LayerNorm(config.pc_hidden_dim)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Self-attention with possible syntax mask
        attn_out, _ = self.self_attention(x, x, x, attn_mask=attention_mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # FFN
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        
        return x


class JointTransformerLayer(nn.Module):
    """Transformer layer for joint encoder"""
    
    def __init__(self, config: DualEncoderConfig):
        super().__init__()
        
        self.self_attention = nn.MultiheadAttention(
            config.hidden_dim,
            config.num_heads,
            dropout=config.dropout,
            batch_first=True
        )
        
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim * 4, config.hidden_dim)
        )
        
        self.norm1 = nn.LayerNorm(config.hidden_dim)
        self.norm2 = nn.LayerNorm(config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention
        attn_out, _ = self.self_attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        
        # FFN
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        
        return x


class PositionalEncoding(nn.Module):
    """Positional encoding for transformers"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            -(np.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class AttentionFusion(nn.Module):
    """Attention-based fusion"""
    
    def __init__(self, config: DualEncoderConfig):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            config.hidden_dim,
            config.num_heads // 2,
            batch_first=True
        )
        
    def forward(self, nl_enc: torch.Tensor, pc_enc: torch.Tensor) -> torch.Tensor:
        fused, _ = self.attention(nl_enc, pc_enc, pc_enc)
        return fused


class ConcatFusion(nn.Module):
    """Concatenation-based fusion"""
    
    def __init__(self, config: DualEncoderConfig):
        super().__init__()
        self.projection = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        
    def forward(self, nl_enc: torch.Tensor, pc_enc: torch.Tensor) -> torch.Tensor:
        # Ensure same length
        min_len = min(nl_enc.size(1), pc_enc.size(1))
        concat = torch.cat([nl_enc[:, :min_len], pc_enc[:, :min_len]], dim=-1)
        return self.projection(concat)


class AverageFusion(nn.Module):
    """Simple average fusion"""
    
    def forward(self, nl_enc: torch.Tensor, pc_enc: torch.Tensor) -> torch.Tensor:
        min_len = min(nl_enc.size(1), pc_enc.size(1))
        return (nl_enc[:, :min_len] + pc_enc[:, :min_len]) / 2