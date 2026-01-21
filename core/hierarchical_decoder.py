"""
Hierarchical Decoder for CodeForge-Quantum
Implements function-level, block-level, and statement-level generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from dataclasses import dataclass
from enum import Enum


class GenerationLevel(Enum):
    """Levels of code generation"""
    FUNCTION = "function"
    BLOCK = "block"
    STATEMENT = "statement"


@dataclass
class HierarchicalConfig:
    """Configuration for hierarchical decoder"""
    hidden_dim: int = 8192
    vocab_size: int = 65536
    num_heads: int = 64
    num_layers: int = 10
    dropout: float = 0.1
    max_functions: int = 10
    max_blocks_per_function: int = 20
    max_statements_per_block: int = 50
    beam_width: int = 5
    length_penalty: float = 0.6
    coverage_penalty: float = 0.4


class FunctionLevelDecoder(nn.Module):
    """Decoder for function-level generation"""
    
    def __init__(self, config: HierarchicalConfig):
        super().__init__()
        self.config = config
        
        # Function LSTM
        self.function_lstm = nn.LSTM(
            config.hidden_dim,
            config.hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=config.dropout,
            bidirectional=False
        )
        
        # Function type predictor
        self.function_type_predictor = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, 10)  # Function types
        )
        
        # Function signature generator
        self.signature_generator = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.vocab_size)
        )
        
        # Function context encoder
        self.context_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.hidden_dim,
                nhead=config.num_heads // 2,
                dim_feedforward=config.hidden_dim * 2,
                dropout=config.dropout,
                batch_first=True
            ),
            num_layers=2
        )
        
        # Output projection
        self.output_projection = nn.Linear(config.hidden_dim, config.vocab_size)
        
    def forward(
        self,
        encoder_output: torch.Tensor,
        prev_functions: Optional[torch.Tensor] = None,
        target: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Generate function-level code structure"""
        
        batch_size = encoder_output.size(0)
        device = encoder_output.device
        
        # Initialize hidden states
        h_0 = torch.zeros(2, batch_size, self.config.hidden_dim, device=device)
        c_0 = torch.zeros(2, batch_size, self.config.hidden_dim, device=device)
        
        # Process with LSTM
        func_output, (h_n, c_n) = self.function_lstm(encoder_output, (h_0, c_0))
        
        # Predict function types
        func_types = self.function_type_predictor(func_output.mean(dim=1))
        func_type_probs = F.softmax(func_types, dim=-1)
        
        # Generate function signatures
        signatures = self.signature_generator(func_output)
        
        # Encode function context
        if prev_functions is not None:
            # Incorporate previous functions for context
            context_input = torch.cat([func_output, prev_functions], dim=1)
        else:
            context_input = func_output
        
        func_context = self.context_encoder(context_input)
        
        # Generate function tokens
        func_logits = self.output_projection(func_context)
        
        return {
            'function_logits': func_logits,
            'function_hidden': h_n,
            'function_cell': c_n,
            'function_types': func_type_probs,
            'function_signatures': F.softmax(signatures, dim=-1),
            'function_context': func_context
        }


class BlockLevelDecoder(nn.Module):
    """Decoder for block-level generation"""
    
    def __init__(self, config: HierarchicalConfig):
        super().__init__()
        self.config = config
        
        # Block LSTM
        self.block_lstm = nn.LSTM(
            config.hidden_dim,
            config.hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=config.dropout
        )
        
        # Block type classifier
        self.block_type_classifier = nn.Sequential(
            nn.Linear(config.hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 8)  # Block types: if, for, while, try, etc.
        )
        
        # Condition generator for control blocks
        self.condition_generator = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.vocab_size)
        )
        
        # Block context fusion
        self.context_fusion = nn.MultiheadAttention(
            config.hidden_dim,
            config.num_heads // 2,
            dropout=config.dropout,
            batch_first=True
        )
        
        # Indentation predictor
        self.indent_predictor = nn.Linear(config.hidden_dim, 10)  # Max indent levels
        
        # Output projection
        self.output_projection = nn.Linear(config.hidden_dim, config.vocab_size)
        
    def forward(
        self,
        function_context: torch.Tensor,
        function_hidden: torch.Tensor,
        function_cell: torch.Tensor,
        prev_blocks: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Generate block-level code structure"""
        
        # Initialize with function context
        batch_size = function_context.size(0)
        
        # Process with LSTM
        block_output, (h_n, c_n) = self.block_lstm(
            function_context,
            (function_hidden, function_cell)
        )
        
        # Classify block types
        block_types = self.block_type_classifier(block_output.mean(dim=1))
        block_type_probs = F.softmax(block_types, dim=-1)
        
        # Generate conditions for control flow blocks
        condition_input = torch.cat([block_output, function_context], dim=-1)
        conditions = self.condition_generator(condition_input)
        
        # Fuse with function context
        block_fused, attn_weights = self.context_fusion(
            block_output,
            function_context,
            function_context
        )
        
        # Predict indentation levels
        indent_levels = self.indent_predictor(block_fused)
        indent_probs = F.softmax(indent_levels, dim=-1)
        
        # Generate block tokens
        block_logits = self.output_projection(block_fused)
        
        return {
            'block_logits': block_logits,
            'block_hidden': h_n,
            'block_cell': c_n,
            'block_types': block_type_probs,
            'block_conditions': F.softmax(conditions, dim=-1),
            'indentation_levels': indent_probs,
            'block_context': block_fused,
            'attention_weights': attn_weights
        }


class StatementLevelDecoder(nn.Module):
    """Decoder for statement-level generation"""
    
    def __init__(self, config: HierarchicalConfig):
        super().__init__()
        self.config = config
        
        # Statement LSTM
        self.statement_lstm = nn.LSTM(
            config.hidden_dim,
            config.hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=config.dropout
        )
        
        # Statement type predictor
        self.statement_type_predictor = nn.Sequential(
            nn.Linear(config.hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 15)  # Statement types
        )
        
        # Variable tracker
        self.variable_tracker = VariableTracker(config.hidden_dim)
        
        # Expression generator
        self.expression_generator = ExpressionGenerator(config)
        
        # Context fusion with block and function
        self.hierarchical_fusion = nn.Sequential(
            nn.Linear(config.hidden_dim * 3, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )
        
        # Fine-grained token generator
        self.token_generator = nn.Linear(config.hidden_dim, config.vocab_size)
        
    def forward(
        self,
        block_context: torch.Tensor,
        function_context: torch.Tensor,
        block_hidden: torch.Tensor,
        block_cell: torch.Tensor,
        variable_context: Optional[Dict] = None
    ) -> Dict[str, torch.Tensor]:
        """Generate statement-level code"""
        
        # Process with LSTM
        stmt_output, (h_n, c_n) = self.statement_lstm(
            block_context,
            (block_hidden, block_cell)
        )
        
        # Predict statement types
        stmt_types = self.statement_type_predictor(stmt_output.mean(dim=1))
        stmt_type_probs = F.softmax(stmt_types, dim=-1)
        
        # Track variables
        if variable_context is None:
            variable_context = {}
        
        variable_state = self.variable_tracker(stmt_output, variable_context)
        
        # Generate expressions
        expressions = self.expression_generator(
            stmt_output,
            variable_state['variable_embeddings']
        )
        
        # Hierarchical fusion
        combined_context = torch.cat([
            stmt_output,
            block_context,
            function_context
        ], dim=-1)
        
        fused_context = self.hierarchical_fusion(combined_context)
        
        # Generate fine-grained tokens
        stmt_logits = self.token_generator(fused_context)
        
        return {
            'statement_logits': stmt_logits,
            'statement_hidden': h_n,
            'statement_cell': c_n,
            'statement_types': stmt_type_probs,
            'variable_state': variable_state,
            'expressions': expressions,
            'statement_context': fused_context
        }


class VariableTracker(nn.Module):
    """Track and manage variables during generation"""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Variable encoder
        self.var_encoder = nn.LSTM(
            hidden_dim,
            hidden_dim // 2,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )
        
        # Variable type classifier
        self.type_classifier = nn.Linear(hidden_dim, 10)  # Variable types
        
        # Variable name generator
        self.name_generator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 100)  # Common variable names
        )
        
        # Variable scope tracker
        self.scope_tracker = nn.Linear(hidden_dim, 5)  # Scope levels
        
    def forward(
        self,
        context: torch.Tensor,
        variable_dict: Dict
    ) -> Dict[str, torch.Tensor]:
        """Track variables in current context"""
        
        # Encode current context
        var_encoded, _ = self.var_encoder(context)
        
        # Classify variable types
        var_types = self.type_classifier(var_encoded.mean(dim=1))
        var_type_probs = F.softmax(var_types, dim=-1)
        
        # Generate variable names
        var_names = self.name_generator(var_encoded.mean(dim=1))
        var_name_probs = F.softmax(var_names, dim=-1)
        
        # Track scope
        scope_levels = self.scope_tracker(var_encoded.mean(dim=1))
        scope_probs = F.softmax(scope_levels, dim=-1)
        
        return {
            'variable_embeddings': var_encoded,
            'variable_types': var_type_probs,
            'variable_names': var_name_probs,
            'scope_levels': scope_probs
        }


class ExpressionGenerator(nn.Module):
    """Generate code expressions"""
    
    def __init__(self, config: HierarchicalConfig):
        super().__init__()
        
        # Expression type predictor
        self.expr_type_predictor = nn.Linear(config.hidden_dim, 20)
        
        # Operator selector
        self.operator_selector = nn.Linear(config.hidden_dim, 30)
        
        # Literal generator
        self.literal_generator = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.vocab_size)
        )
        
        # Expression combiner
        self.expr_combiner = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=config.hidden_dim,
                nhead=8,
                dim_feedforward=config.hidden_dim * 2,
                dropout=config.dropout,
                batch_first=True
            ),
            num_layers=2
        )
        
    def forward(
        self,
        context: torch.Tensor,
        variable_embeddings: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Generate expressions"""
        
        # Predict expression types
        expr_types = self.expr_type_predictor(context.mean(dim=1))
        expr_type_probs = F.softmax(expr_types, dim=-1)
        
        # Select operators
        operators = self.operator_selector(context.mean(dim=1))
        operator_probs = F.softmax(operators, dim=-1)
        
        # Generate literals
        if variable_embeddings is not None:
            literal_input = torch.cat([context, variable_embeddings], dim=-1)
        else:
            literal_input = torch.cat([context, context], dim=-1)
        
        literals = self.literal_generator(literal_input)
        literal_probs = F.softmax(literals, dim=-1)
        
        # Combine into expressions
        if variable_embeddings is not None:
            expr_combined = self.expr_combiner(context, variable_embeddings)
        else:
            expr_combined = context
        
        return {
            'expression_types': expr_type_probs,
            'operators': operator_probs,
            'literals': literal_probs,
            'combined_expressions': expr_combined
        }


class HierarchicalDecoder(nn.Module):
    """Complete hierarchical decoder"""
    
    def __init__(self, config: HierarchicalConfig):
        super().__init__()
        self.config = config
        
        # Level decoders
        self.function_decoder = FunctionLevelDecoder(config)
        self.block_decoder = BlockLevelDecoder(config)
        self.statement_decoder = StatementLevelDecoder(config)
        
        # Level selector
        self.level_selector = nn.Sequential(
            nn.Linear(config.hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 3)  # Three levels
        )
        
        # Cross-level attention
        self.cross_level_attention = nn.ModuleDict({
            'func_to_block': nn.MultiheadAttention(
                config.hidden_dim, config.num_heads // 2,
                batch_first=True
            ),
            'block_to_stmt': nn.MultiheadAttention(
                config.hidden_dim, config.num_heads // 2,
                batch_first=True
            ),
            'stmt_to_func': nn.MultiheadAttention(
                config.hidden_dim, config.num_heads // 2,
                batch_first=True
            )
        })
        
        # Output combiner
        self.output_combiner = nn.Sequential(
            nn.Linear(config.hidden_dim * 3, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.vocab_size)
        )
        
    def forward(
        self,
        encoder_output: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        generation_mode: str = "hierarchical"
    ) -> Dict[str, Any]:
        """Hierarchical decoding process"""
        
        batch_size = encoder_output.size(0)
        device = encoder_output.device
        
        # Determine generation level
        level_logits = self.level_selector(encoder_output.mean(dim=1))
        level_probs = F.softmax(level_logits, dim=-1)
        
        # Function-level generation
        func_output = self.function_decoder(encoder_output)
        
        # Block-level generation (conditioned on functions)
        block_output = self.block_decoder(
            func_output['function_context'],
            func_output['function_hidden'],
            func_output['function_cell']
        )
        
        # Apply cross-level attention (function to block)
        block_attended, _ = self.cross_level_attention['func_to_block'](
            block_output['block_context'],
            func_output['function_context'],
            func_output['function_context']
        )
        
        # Statement-level generation (conditioned on blocks and functions)
        stmt_output = self.statement_decoder(
            block_attended,
            func_output['function_context'],
            block_output['block_hidden'],
            block_output['block_cell']
        )
        
        # Apply cross-level attention (block to statement)
        stmt_attended, _ = self.cross_level_attention['block_to_stmt'](
            stmt_output['statement_context'],
            block_attended,
            block_attended
        )
        
        # Optional: feedback from statement to function level
        if generation_mode == "hierarchical_feedback":
            func_refined, _ = self.cross_level_attention['stmt_to_func'](
                func_output['function_context'],
                stmt_attended,
                stmt_attended
            )
        else:
            func_refined = func_output['function_context']
        
        # Combine all levels
        combined = torch.cat([
            func_refined,
            block_attended,
            stmt_attended
        ], dim=-1)
        
        # Generate final output
        final_logits = self.output_combiner(combined)
        
        # Compute hierarchical probability
        if generation_mode == "hierarchical":
            # P(code) = P(func) * P(block|func) * P(stmt|block,func)
            hierarchical_prob = self._compute_hierarchical_probability(
                func_output['function_logits'],
                block_output['block_logits'],
                stmt_output['statement_logits']
            )
        else:
            hierarchical_prob = F.softmax(final_logits, dim=-1)
        
        return {
            'logits': final_logits,
            'hierarchical_probability': hierarchical_prob,
            'function_output': func_output,
            'block_output': block_output,
            'statement_output': stmt_output,
            'level_probabilities': level_probs,
            'combined_context': combined
        }
    
    def _compute_hierarchical_probability(
        self,
        func_logits: torch.Tensor,
        block_logits: torch.Tensor,
        stmt_logits: torch.Tensor
    ) -> torch.Tensor:
        """Compute hierarchical probability distribution"""
        
        # Convert to probabilities
        func_probs = F.softmax(func_logits, dim=-1)
        block_probs = F.softmax(block_logits, dim=-1)
        stmt_probs = F.softmax(stmt_logits, dim=-1)
        
        # Weighted combination based on level
        # This is simplified - actual implementation would be more sophisticated
        combined_probs = (
            0.3 * func_probs.mean(dim=1, keepdim=True).expand_as(stmt_probs) +
            0.3 * block_probs.mean(dim=1, keepdim=True).expand_as(stmt_probs) +
            0.4 * stmt_probs
        )
        
        return combined_probs
    
    def beam_search(
        self,
        encoder_output: torch.Tensor,
        max_length: int = 512,
        beam_width: Optional[int] = None
    ) -> List[torch.Tensor]:
        """Hierarchical beam search decoding"""
        
        beam_width = beam_width or self.config.beam_width
        batch_size = encoder_output.size(0)
        device = encoder_output.device
        
        # Initialize beams for each level
        function_beams = []
        block_beams = []
        statement_beams = []
        
        # Generate functions first
        func_output = self.function_decoder(encoder_output)
        func_candidates = self._get_top_k_tokens(
            func_output['function_logits'],
            beam_width
        )
        
        for func_seq, func_score in func_candidates:
            # For each function, generate blocks
            block_output = self.block_decoder(
                func_output['function_context'],
                func_output['function_hidden'],
                func_output['function_cell']
            )
            
            block_candidates = self._get_top_k_tokens(
                block_output['block_logits'],
                beam_width
            )
            
            for block_seq, block_score in block_candidates:
                # For each block, generate statements
                stmt_output = self.statement_decoder(
                    block_output['block_context'],
                    func_output['function_context'],
                    block_output['block_hidden'],
                    block_output['block_cell']
                )
                
                stmt_candidates = self._get_top_k_tokens(
                    stmt_output['statement_logits'],
                    beam_width
                )
                
                for stmt_seq, stmt_score in stmt_candidates:
                    # Combine hierarchical score
                    total_score = (
                        func_score * self.config.length_penalty +
                        block_score * self.config.length_penalty +
                        stmt_score
                    )
                    
                    # Combine sequences
                    combined_seq = self._combine_hierarchical_sequences(
                        func_seq, block_seq, stmt_seq
                    )
                    
                    statement_beams.append((combined_seq, total_score))
        
        # Sort by score and return top beam
        statement_beams.sort(key=lambda x: x[1], reverse=True)
        
        if statement_beams:
            return [statement_beams[0][0]]
        else:
            return [torch.zeros(1, max_length, dtype=torch.long, device=device)]
    
    def _get_top_k_tokens(
        self,
        logits: torch.Tensor,
        k: int
    ) -> List[Tuple[torch.Tensor, float]]:
        """Get top-k token sequences with scores"""
        
        probs = F.softmax(logits[:, -1, :], dim=-1)
        top_k_probs, top_k_indices = torch.topk(probs, k, dim=-1)
        
        candidates = []
        for i in range(k):
            seq = top_k_indices[:, i]
            score = top_k_probs[:, i].item()
            candidates.append((seq, score))
        
        return candidates
    
    def _combine_hierarchical_sequences(
        self,
        func_seq: torch.Tensor,
        block_seq: torch.Tensor,
        stmt_seq: torch.Tensor
    ) -> torch.Tensor:
        """Combine sequences from different hierarchy levels"""
        
        # Simple concatenation - actual implementation would be more sophisticated
        combined = torch.cat([func_seq, block_seq, stmt_seq], dim=0)
        return combined