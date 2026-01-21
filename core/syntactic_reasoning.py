"""
Syntactic Reasoning Module for CodeForge-Quantum
Second stage of the triple-stage architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import ast
import numpy as np
from dataclasses import dataclass


@dataclass
class SyntacticConfig:
    """Configuration for syntactic reasoning module"""
    hidden_dim: int = 8192
    num_heads: int = 64
    num_layers: int = 20
    dropout: float = 0.1
    max_depth: int = 20
    vocab_size: int = 65536
    ast_node_types: int = 100


class ASTEncoder(nn.Module):
    """Encode Abstract Syntax Tree structures"""
    
    def __init__(self, config: SyntacticConfig):
        super().__init__()
        
        # AST node type embeddings
        self.node_embeddings = nn.Embedding(
            config.ast_node_types,
            config.hidden_dim
        )
        
        # Tree-structured LSTM
        self.tree_lstm = TreeLSTM(
            config.hidden_dim,
            config.hidden_dim
        )
        
        # AST position encoding
        self.depth_embedding = nn.Embedding(
            config.max_depth,
            config.hidden_dim
        )
        
        self.breadth_embedding = nn.Embedding(
            100,  # Max breadth position
            config.hidden_dim
        )
        
        # AST attention
        self.ast_attention = nn.MultiheadAttention(
            config.hidden_dim,
            config.num_heads,
            dropout=config.dropout,
            batch_first=True
        )
        
    def forward(
        self,
        ast_nodes: List[Dict],
        hidden_states: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Encode AST structure"""
        
        if not ast_nodes:
            # Return zero tensors if no AST
            batch_size = hidden_states.size(0)
            return {
                'ast_encoding': torch.zeros(batch_size, 1, self.node_embeddings.embedding_dim),
                'tree_hidden': torch.zeros(batch_size, self.node_embeddings.embedding_dim),
                'node_representations': []
            }
        
        # Convert AST to tensor representation
        node_tensors = []
        for node in ast_nodes:
            node_type = node.get('type', 0)
            depth = min(node.get('depth', 0), self.depth_embedding.num_embeddings - 1)
            breadth = min(node.get('breadth', 0), self.breadth_embedding.num_embeddings - 1)
            
            # Embed node
            node_embed = self.node_embeddings(torch.tensor(node_type))
            depth_embed = self.depth_embedding(torch.tensor(depth))
            breadth_embed = self.breadth_embedding(torch.tensor(breadth))
            
            # Combine embeddings
            node_repr = node_embed + depth_embed + breadth_embed
            node_tensors.append(node_repr)
        
        # Stack node representations
        node_tensor = torch.stack(node_tensors).unsqueeze(0)  # Add batch dimension
        
        # Apply tree LSTM
        tree_hidden = self.tree_lstm(node_tensor)
        
        # Apply attention with hidden states
        attended, attn_weights = self.ast_attention(
            hidden_states,
            node_tensor,
            node_tensor
        )
        
        return {
            'ast_encoding': attended,
            'tree_hidden': tree_hidden,
            'node_representations': node_tensors,
            'attention_weights': attn_weights
        }


class TreeLSTM(nn.Module):
    """Child-sum Tree-LSTM for AST encoding"""
    
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Input gate
        self.i_x = nn.Linear(input_dim, hidden_dim)
        self.i_h = nn.Linear(hidden_dim, hidden_dim)
        
        # Forget gate (one per child)
        self.f_x = nn.Linear(input_dim, hidden_dim)
        self.f_h = nn.Linear(hidden_dim, hidden_dim)
        
        # Output gate
        self.o_x = nn.Linear(input_dim, hidden_dim)
        self.o_h = nn.Linear(hidden_dim, hidden_dim)
        
        # Cell update
        self.u_x = nn.Linear(input_dim, hidden_dim)
        self.u_h = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through Tree-LSTM"""
        # Simplified version - treats as sequence
        batch_size, seq_len, input_dim = x.shape
        
        h = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        c = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        
        for t in range(seq_len):
            x_t = x[:, t, :]
            
            # Gates
            i = torch.sigmoid(self.i_x(x_t) + self.i_h(h))
            f = torch.sigmoid(self.f_x(x_t) + self.f_h(h))
            o = torch.sigmoid(self.o_x(x_t) + self.o_h(h))
            u = torch.tanh(self.u_x(x_t) + self.u_h(h))
            
            # Update cell and hidden
            c = f * c + i * u
            h = o * torch.tanh(c)
        
        return h


class ControlFlowAnalyzer(nn.Module):
    """Analyze control flow structures in code"""
    
    def __init__(self, config: SyntacticConfig):
        super().__init__()
        
        # Control flow patterns
        self.flow_patterns = {
            'sequential': 0,
            'conditional': 1,
            'loop': 2,
            'nested_loop': 3,
            'recursion': 4,
            'exception': 5
        }
        
        # Pattern detector
        self.pattern_detector = nn.Linear(
            config.hidden_dim,
            len(self.flow_patterns)
        )
        
        # Flow complexity estimator
        self.complexity_estimator = nn.Sequential(
            nn.Linear(config.hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)  # Cyclomatic complexity
        )
        
        # Nesting depth analyzer
        self.nesting_analyzer = nn.Sequential(
            nn.Linear(config.hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # Max nesting depth
        )
        
        # Control flow encoder
        self.flow_encoder = nn.GRU(
            config.hidden_dim,
            config.hidden_dim,
            num_layers=2,
            batch_first=True
        )
        
    def forward(self, hidden_states: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Analyze control flow"""
        
        # Encode flow
        flow_encoded, _ = self.flow_encoder(hidden_states)
        
        # Pool for classification
        pooled = flow_encoded.mean(dim=1)
        
        # Detect patterns
        pattern_logits = self.pattern_detector(pooled)
        pattern_probs = F.softmax(pattern_logits, dim=-1)
        
        # Estimate complexity
        cyclomatic = self.complexity_estimator(pooled)
        nesting = self.nesting_analyzer(pooled)
        
        return {
            'flow_encoding': flow_encoded,
            'flow_patterns': pattern_probs,
            'cyclomatic_complexity': cyclomatic,
            'max_nesting': nesting
        }


class DataStructureReasoner(nn.Module):
    """Reason about appropriate data structures"""
    
    def __init__(self, config: SyntacticConfig):
        super().__init__()
        
        # Common data structures
        self.data_structures = [
            'array', 'list', 'stack', 'queue', 'deque',
            'hashmap', 'set', 'tree', 'graph', 'heap',
            'trie', 'union_find', 'segment_tree'
        ]
        
        # Data structure selector
        self.ds_selector = nn.Linear(
            config.hidden_dim,
            len(self.data_structures)
        )
        
        # Operation analyzer
        self.operation_analyzer = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, 10)  # Common operations
        )
        
        # Efficiency scorer
        self.efficiency_scorer = nn.Sequential(
            nn.Linear(config.hidden_dim + len(self.data_structures), 256),
            nn.ReLU(),
            nn.Linear(256, len(self.data_structures))
        )
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        algorithm_hints: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Reason about data structures"""
        
        pooled = hidden_states.mean(dim=1)
        
        # Select data structures
        ds_logits = self.ds_selector(pooled)
        ds_probs = torch.sigmoid(ds_logits)  # Multi-label
        
        # Analyze required operations
        operations = self.operation_analyzer(pooled)
        operation_probs = torch.sigmoid(operations)
        
        # Score efficiency of each DS
        combined = torch.cat([pooled, ds_probs], dim=-1)
        efficiency = self.efficiency_scorer(combined)
        efficiency_scores = F.softmax(efficiency, dim=-1)
        
        return {
            'data_structures': ds_probs,
            'operations': operation_probs,
            'efficiency_scores': efficiency_scores,
            'recommended_ds': torch.argmax(efficiency_scores, dim=-1)
        }


class SyntaxPatternMatcher(nn.Module):
    """Match and generate syntax patterns"""
    
    def __init__(self, config: SyntacticConfig):
        super().__init__()
        
        # Pattern library
        self.pattern_library_size = 500
        
        # Pattern encoder
        self.pattern_encoder = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, self.pattern_library_size)
        )
        
        # Pattern memory
        self.pattern_memory = nn.Parameter(
            torch.randn(self.pattern_library_size, config.hidden_dim)
        )
        
        # Pattern combiner
        self.pattern_combiner = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=config.hidden_dim,
                nhead=8,
                dim_feedforward=config.hidden_dim * 2,
                dropout=config.dropout,
                batch_first=True
            ),
            num_layers=2
        )
        
    def forward(self, hidden_states: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Match syntax patterns"""
        
        # Encode to pattern space
        pattern_logits = self.pattern_encoder(hidden_states.mean(dim=1))
        pattern_weights = F.softmax(pattern_logits, dim=-1)
        
        # Retrieve patterns from memory
        retrieved_patterns = torch.matmul(pattern_weights, self.pattern_memory)
        
        # Combine patterns with context
        combined = self.pattern_combiner(
            hidden_states,
            retrieved_patterns.unsqueeze(1).expand(-1, hidden_states.size(1), -1)
        )
        
        return {
            'pattern_weights': pattern_weights,
            'retrieved_patterns': retrieved_patterns,
            'combined_syntax': combined
        }


class SyntacticReasoning(nn.Module):
    """Complete syntactic reasoning module"""
    
    def __init__(self, config: SyntacticConfig):
        super().__init__()
        self.config = config
        
        # Components
        self.ast_encoder = ASTEncoder(config)
        self.control_flow_analyzer = ControlFlowAnalyzer(config)
        self.ds_reasoner = DataStructureReasoner(config)
        self.pattern_matcher = SyntaxPatternMatcher(config)
        
        # Syntactic transformer layers
        self.syntactic_layers = nn.ModuleList([
            SyntacticTransformerLayer(config)
            for _ in range(config.num_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(
            config.hidden_dim * 4,  # Concatenated components
            config.hidden_dim
        )
        
        # Syntax validator
        self.syntax_validator = nn.Sequential(
            nn.Linear(config.hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 2)  # Valid/Invalid
        )
        
    def forward(
        self,
        semantic_repr: torch.Tensor,
        ast_nodes: Optional[List[Dict]] = None,
        algorithm_hints: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """Apply syntactic reasoning"""
        
        # Initial hidden states from semantic representation
        hidden_states = semantic_repr
        
        # Apply syntactic transformer layers
        for layer in self.syntactic_layers:
            hidden_states = layer(hidden_states)
        
        # Encode AST if available
        ast_output = self.ast_encoder(ast_nodes or [], hidden_states)
        
        # Analyze control flow
        control_flow = self.control_flow_analyzer(hidden_states)
        
        # Reason about data structures
        data_structures = self.ds_reasoner(hidden_states, algorithm_hints)
        
        # Match syntax patterns
        patterns = self.pattern_matcher(hidden_states)
        
        # Combine all syntactic components
        combined = torch.cat([
            hidden_states.mean(dim=1),
            ast_output['tree_hidden'] if ast_output['tree_hidden'].dim() > 1 else ast_output['tree_hidden'].unsqueeze(0),
            control_flow['flow_encoding'].mean(dim=1),
            patterns['retrieved_patterns']
        ], dim=-1)
        
        # Project to output dimension
        syntactic_repr = self.output_projection(combined)
        
        # Validate syntax
        validity = self.syntax_validator(syntactic_repr)
        validity_prob = F.softmax(validity, dim=-1)
        
        return {
            'syntactic_representation': syntactic_repr,
            'ast_encoding': ast_output,
            'control_flow': control_flow,
            'data_structures': data_structures,
            'syntax_patterns': patterns,
            'validity_score': validity_prob[:, 1],  # Probability of valid
            'hidden_states': hidden_states
        }


class SyntacticTransformerLayer(nn.Module):
    """Single transformer layer for syntactic reasoning"""
    
    def __init__(self, config: SyntacticConfig):
        super().__init__()
        
        # Self-attention
        self.self_attention = nn.MultiheadAttention(
            config.hidden_dim,
            config.num_heads,
            dropout=config.dropout,
            batch_first=True
        )
        
        # Feed-forward
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim * 4, config.hidden_dim)
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(config.hidden_dim)
        self.norm2 = nn.LayerNorm(config.hidden_dim)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through layer"""
        
        # Self-attention with residual
        attn_out, _ = self.self_attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        
        # FFN with residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        
        return x


class CodeStructureBuilder(nn.Module):
    """Build code structure from syntactic reasoning"""
    
    def __init__(self, config: SyntacticConfig):
        super().__init__()
        
        # Structure templates
        self.template_size = 100
        
        # Template selector
        self.template_selector = nn.Linear(
            config.hidden_dim,
            self.template_size
        )
        
        # Template memory
        self.template_memory = nn.Parameter(
            torch.randn(self.template_size, config.hidden_dim)
        )
        
        # Structure decoder
        self.structure_decoder = nn.LSTM(
            config.hidden_dim,
            config.hidden_dim,
            num_layers=2,
            batch_first=True
        )
        
        # Component placer
        self.component_placer = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 10)  # Component types
        )
        
    def forward(
        self,
        syntactic_repr: torch.Tensor,
        target_length: int = 100
    ) -> Dict[str, torch.Tensor]:
        """Build code structure"""
        
        # Select template
        template_logits = self.template_selector(syntactic_repr)
        template_weights = F.softmax(template_logits, dim=-1)
        
        # Retrieve template
        template = torch.matmul(template_weights, self.template_memory)
        
        # Decode structure
        # Initialize with template
        batch_size = syntactic_repr.size(0)
        hidden = template.unsqueeze(0).expand(2, batch_size, -1).contiguous()
        cell = torch.zeros_like(hidden)
        
        structures = []
        input_t = syntactic_repr.unsqueeze(1)
        
        for _ in range(target_length):
            output, (hidden, cell) = self.structure_decoder(
                input_t, (hidden, cell)
            )
            structures.append(output)
            input_t = output
        
        structure = torch.cat(structures, dim=1)
        
        # Place components
        combined = torch.cat([
            structure,
            syntactic_repr.unsqueeze(1).expand(-1, structure.size(1), -1)
        ], dim=-1)
        components = self.component_placer(combined)
        
        return {
            'structure': structure,
            'components': F.softmax(components, dim=-1),
            'template_weights': template_weights
        }