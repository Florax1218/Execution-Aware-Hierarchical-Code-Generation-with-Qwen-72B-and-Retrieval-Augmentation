"""
Multi-Objective Loss Functions for CodeForge-Quantum
Includes cross-entropy, AST alignment, semantic preservation, trace alignment, and complexity regularization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import ast
import numpy as np
from dataclasses import dataclass
import edit_distance


@dataclass
class LossComponents:
    """Container for different loss components"""
    ce_loss: torch.Tensor
    ast_loss: torch.Tensor
    sem_loss: torch.Tensor
    trace_loss: torch.Tensor
    complex_loss: torch.Tensor
    total_loss: torch.Tensor
    
    def to_dict(self) -> Dict[str, torch.Tensor]:
        return {
            'ce_loss': self.ce_loss,
            'ast_loss': self.ast_loss,
            'sem_loss': self.sem_loss,
            'trace_loss': self.trace_loss,
            'complex_loss': self.complex_loss,
            'total_loss': self.total_loss
        }


class CrossEntropyWithLabelSmoothing(nn.Module):
    """Cross-entropy loss with label smoothing"""
    def __init__(
        self,
        vocab_size: int = 65536,
        smoothing: float = 0.1,
        ignore_index: int = -100
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.smoothing = smoothing
        self.ignore_index = ignore_index
        self.confidence = 1.0 - smoothing
        
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute cross-entropy loss with label smoothing"""
        batch_size, seq_len, vocab_size = logits.shape
        
        # Reshape for loss computation
        logits = logits.reshape(-1, vocab_size)
        targets = targets.reshape(-1)
        
        # Create smoothed target distribution
        smooth_targets = torch.zeros_like(logits)
        smooth_targets.fill_(self.smoothing / (vocab_size - 1))
        
        # Set confidence for true labels
        mask = targets != self.ignore_index
        if mask.any():
            smooth_targets[mask, targets[mask]] = self.confidence
        
        # Compute KL divergence
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -torch.sum(smooth_targets * log_probs, dim=-1)
        
        # Apply mask for padding
        loss = loss * mask.float()
        
        # Average over non-padding tokens
        return loss.sum() / mask.sum() if mask.any() else loss.mean()


class ASTAlignmentLoss(nn.Module):
    """Abstract Syntax Tree alignment loss for structural similarity"""
    def __init__(
        self,
        lambda_ast: float = 0.3,
        depth_penalty: float = 0.05
    ):
        super().__init__()
        self.lambda_ast = lambda_ast
        self.depth_penalty = depth_penalty
        
    def forward(
        self,
        generated_code: str,
        reference_code: str
    ) -> torch.Tensor:
        """Compute AST alignment loss"""
        try:
            # Parse ASTs
            gen_ast = ast.parse(generated_code)
            ref_ast = ast.parse(reference_code)
            
            # Compute structural distance
            distance = self._compute_tree_edit_distance(gen_ast, ref_ast)
            
            # Compute depth difference
            gen_depth = self._get_max_depth(gen_ast)
            ref_depth = self._get_max_depth(ref_ast)
            depth_diff = abs(gen_depth - ref_depth)
            
            # Combine losses
            loss = self.lambda_ast * distance + self.depth_penalty * depth_diff
            
            return torch.tensor(loss, dtype=torch.float32)
            
        except SyntaxError:
            # If parsing fails, return high loss
            return torch.tensor(10.0, dtype=torch.float32)
    
    def _compute_tree_edit_distance(
        self,
        tree1: ast.AST,
        tree2: ast.AST
    ) -> float:
        """Compute tree edit distance between two ASTs"""
        # Simplified tree edit distance
        # In practice, would use Zhang-Shasha algorithm
        
        nodes1 = self._get_all_nodes(tree1)
        nodes2 = self._get_all_nodes(tree2)
        
        # Compare node types
        types1 = [type(n).__name__ for n in nodes1]
        types2 = [type(n).__name__ for n in nodes2]
        
        # Use edit distance on node type sequences
        distance = edit_distance.SequenceMatcher(
            a=types1,
            b=types2
        ).distance()
        
        # Normalize by average length
        norm_distance = distance / max(len(types1), len(types2), 1)
        
        return norm_distance
    
    def _get_all_nodes(self, tree: ast.AST) -> List[ast.AST]:
        """Get all nodes from AST"""
        nodes = []
        
        def visit(node):
            nodes.append(node)
            for child in ast.iter_child_nodes(node):
                visit(child)
        
        visit(tree)
        return nodes
    
    def _get_max_depth(self, tree: ast.AST) -> int:
        """Get maximum depth of AST"""
        def depth(node, current_depth=0):
            if not list(ast.iter_child_nodes(node)):
                return current_depth
            
            max_child_depth = 0
            for child in ast.iter_child_nodes(node):
                child_depth = depth(child, current_depth + 1)
                max_child_depth = max(max_child_depth, child_depth)
            
            return max_child_depth
        
        return depth(tree)


class SemanticPreservationLoss(nn.Module):
    """Loss for ensuring functional equivalence"""
    def __init__(
        self,
        lambda_sem: float = 0.25,
        num_test_cases: int = 100
    ):
        super().__init__()
        self.lambda_sem = lambda_sem
        self.num_test_cases = num_test_cases
        
    def forward(
        self,
        generated_func: callable,
        reference_func: callable,
        test_inputs: List[Any]
    ) -> torch.Tensor:
        """Compute semantic preservation loss"""
        errors = []
        
        for test_input in test_inputs[:self.num_test_cases]:
            try:
                gen_output = generated_func(*test_input)
                ref_output = reference_func(*test_input)
                
                # Compute output difference
                if isinstance(gen_output, (int, float)):
                    error = abs(gen_output - ref_output)
                elif isinstance(gen_output, str):
                    error = 0 if gen_output == ref_output else 1
                elif isinstance(gen_output, (list, tuple)):
                    error = self._sequence_distance(gen_output, ref_output)
                else:
                    error = 0 if gen_output == ref_output else 1
                
                errors.append(error)
                
            except Exception:
                # If execution fails, add penalty
                errors.append(1.0)
        
        # Average error across test cases
        avg_error = sum(errors) / len(errors) if errors else 0.0
        
        return torch.tensor(self.lambda_sem * avg_error, dtype=torch.float32)
    
    def _sequence_distance(self, seq1, seq2) -> float:
        """Compute normalized distance between sequences"""
        if len(seq1) != len(seq2):
            return 1.0
        
        if not seq1:
            return 0.0
        
        differences = sum(1 for a, b in zip(seq1, seq2) if a != b)
        return differences / len(seq1)


class TraceAlignmentLoss(nn.Module):
    """Loss for aligning execution traces"""
    def __init__(self):
        super().__init__()
        
    def forward(
        self,
        generated_trace: List[Dict],
        reference_trace: List[Dict]
    ) -> torch.Tensor:
        """Compute trace alignment loss using KL divergence"""
        total_loss = 0.0
        
        # Align traces by time step
        min_len = min(len(generated_trace), len(reference_trace))
        
        for t in range(min_len):
            gen_state = generated_trace[t]
            ref_state = reference_trace[t]
            
            # Convert states to probability distributions
            gen_probs = self._state_to_distribution(gen_state)
            ref_probs = self._state_to_distribution(ref_state)
            
            # Compute KL divergence
            kl_div = F.kl_div(
                torch.log(gen_probs + 1e-10),
                ref_probs,
                reduction='sum'
            )
            
            total_loss += kl_div
        
        # Add penalty for length mismatch
        length_penalty = abs(len(generated_trace) - len(reference_trace)) * 0.1
        total_loss += length_penalty
        
        return total_loss / min_len if min_len > 0 else torch.tensor(0.0)
    
    def _state_to_distribution(self, state: Dict) -> torch.Tensor:
        """Convert execution state to probability distribution"""
        # Simplified - in practice would encode full state
        # Here we just create a dummy distribution
        
        num_features = 100
        distribution = torch.zeros(num_features)
        
        # Encode some state features
        if 'variables' in state:
            distribution[0] = len(state['variables'])
        if 'line_number' in state:
            distribution[1] = state['line_number']
        if 'loop_depth' in state:
            distribution[2] = state['loop_depth']
        
        # Normalize to probability distribution
        distribution = F.softmax(distribution, dim=0)
        
        return distribution


class ComplexityRegularization(nn.Module):
    """Regularization based on code complexity metrics"""
    def __init__(
        self,
        lambda_c: float = 0.1,
        max_cyclomatic: int = 10,
        max_nesting: int = 5,
        max_lines: int = 100
    ):
        super().__init__()
        self.lambda_c = lambda_c
        self.max_cyclomatic = max_cyclomatic
        self.max_nesting = max_nesting
        self.max_lines = max_lines
        
    def forward(self, code: str) -> torch.Tensor:
        """Compute complexity regularization loss"""
        cyclomatic = self._compute_cyclomatic_complexity(code)
        nesting = self._compute_nesting_depth(code)
        lines = len(code.split('\n'))
        
        # Normalize complexity metrics
        cyclo_penalty = max(0, cyclomatic - self.max_cyclomatic) / self.max_cyclomatic
        nest_penalty = max(0, nesting - self.max_nesting) / self.max_nesting
        lines_penalty = max(0, lines - self.max_lines) / self.max_lines
        
        # Combine penalties
        total_penalty = (cyclo_penalty + nest_penalty + lines_penalty) / 3
        
        return torch.tensor(self.lambda_c * total_penalty, dtype=torch.float32)
    
    def _compute_cyclomatic_complexity(self, code: str) -> int:
        """Compute cyclomatic complexity of code"""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return 0
        
        complexity = 1  # Base complexity
        
        class ComplexityVisitor(ast.NodeVisitor):
            def __init__(self):
                self.complexity = 0
                
            def visit_If(self, node):
                self.complexity += 1
                # Add 1 for elif branches
                self.complexity += len(node.orelse) if node.orelse else 0
                self.generic_visit(node)
                
            def visit_For(self, node):
                self.complexity += 1
                self.generic_visit(node)
                
            def visit_While(self, node):
                self.complexity += 1
                self.generic_visit(node)
                
            def visit_ExceptHandler(self, node):
                self.complexity += 1
                self.generic_visit(node)
                
            def visit_With(self, node):
                self.complexity += 1
                self.generic_visit(node)
                
            def visit_Assert(self, node):
                self.complexity += 1
                self.generic_visit(node)
                
            def visit_BoolOp(self, node):
                # Add complexity for boolean operators
                self.complexity += len(node.values) - 1
                self.generic_visit(node)
        
        visitor = ComplexityVisitor()
        visitor.visit(tree)
        
        return complexity + visitor.complexity
    
    def _compute_nesting_depth(self, code: str) -> int:
        """Compute maximum nesting depth of code"""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return 0
        
        class NestingVisitor(ast.NodeVisitor):
            def __init__(self):
                self.max_depth = 0
                self.current_depth = 0
                
            def visit_If(self, node):
                self._enter_block(node)
                
            def visit_For(self, node):
                self._enter_block(node)
                
            def visit_While(self, node):
                self._enter_block(node)
                
            def visit_With(self, node):
                self._enter_block(node)
                
            def visit_FunctionDef(self, node):
                self._enter_block(node)
                
            def visit_ClassDef(self, node):
                self._enter_block(node)
                
            def _enter_block(self, node):
                self.current_depth += 1
                self.max_depth = max(self.max_depth, self.current_depth)
                self.generic_visit(node)
                self.current_depth -= 1
        
        visitor = NestingVisitor()
        visitor.visit(tree)
        
        return visitor.max_depth


class MultiObjectiveLoss(nn.Module):
    """Combined multi-objective loss with dynamic weighting"""
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        # Initialize individual loss components
        self.ce_loss = CrossEntropyWithLabelSmoothing(
            vocab_size=config.get('vocab_size', 65536),
            smoothing=config.get('label_smoothing', 0.1)
        )
        
        self.ast_loss = ASTAlignmentLoss(
            lambda_ast=config.get('weight_ast', 0.3),
            depth_penalty=config.get('depth_penalty', 0.05)
        )
        
        self.sem_loss = SemanticPreservationLoss(
            lambda_sem=config.get('weight_sem', 0.25),
            num_test_cases=config.get('num_test_cases', 100)
        )
        
        self.trace_loss = TraceAlignmentLoss()
        
        self.complex_loss = ComplexityRegularization(
            lambda_c=config.get('weight_complex', 0.1)
        )
        
        # Dynamic weights (learnable)
        self.log_weights = nn.Parameter(torch.zeros(5))
        
        # Weight adaptation rate
        self.adaptation_rate = config.get('adaptation_rate', 0.01)
        
        # Loss history for adaptive weighting
        self.loss_history = {
            'ce': [],
            'ast': [],
            'sem': [],
            'trace': [],
            'complex': []
        }
        
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, Any]
    ) -> LossComponents:
        """Compute all loss components"""
        
        # Cross-entropy loss
        ce_loss = self.ce_loss(outputs['logits'], targets['token_ids'])
        
        # AST alignment loss (if code strings available)
        ast_loss = torch.tensor(0.0)
        if 'generated_code' in outputs and 'reference_code' in targets:
            ast_loss = self.ast_loss(
                outputs['generated_code'],
                targets['reference_code']
            )
        
        # Semantic preservation loss (if functions available)
        sem_loss = torch.tensor(0.0)
        if 'generated_func' in outputs and 'reference_func' in targets:
            sem_loss = self.sem_loss(
                outputs['generated_func'],
                targets['reference_func'],
                targets.get('test_inputs', [])
            )
        
        # Trace alignment loss (if traces available)
        trace_loss = torch.tensor(0.0)
        if 'generated_trace' in outputs and 'reference_trace' in targets:
            trace_loss = self.trace_loss(
                outputs['generated_trace'],
                targets['reference_trace']
            )
        
        # Complexity regularization
        complex_loss = torch.tensor(0.0)
        if 'generated_code' in outputs:
            complex_loss = self.complex_loss(outputs['generated_code'])
        
        # Get dynamic weights
        weights = F.softmax(self.log_weights, dim=0) * 5  # Scale to sum ~5
        
        # Compute total loss with dynamic weights
        total_loss = (
            weights[0] * ce_loss +
            weights[1] * ast_loss +
            weights[2] * sem_loss +
            weights[3] * trace_loss +
            weights[4] * complex_loss
        )
        
        # Update loss history
        self.loss_history['ce'].append(ce_loss.item())
        self.loss_history['ast'].append(ast_loss.item())
        self.loss_history['sem'].append(sem_loss.item())
        self.loss_history['trace'].append(trace_loss.item())
        self.loss_history['complex'].append(complex_loss.item())
        
        # Adaptive weight update
        if len(self.loss_history['ce']) > 10:
            self._update_weights()
        
        return LossComponents(
            ce_loss=ce_loss,
            ast_loss=ast_loss,
            sem_loss=sem_loss,
            trace_loss=trace_loss,
            complex_loss=complex_loss,
            total_loss=total_loss
        )
    
    def _update_weights(self):
        """Update weights based on loss gradients"""
        with torch.no_grad():
            # Compute rate of change for each loss
            for i, key in enumerate(['ce', 'ast', 'sem', 'trace', 'complex']):
                if len(self.loss_history[key]) > 1:
                    recent_losses = self.loss_history[key][-10:]
                    gradient = (recent_losses[-1] - recent_losses[0]) / len(recent_losses)
                    
                    # Update weight based on gradient
                    # Increase weight if loss is decreasing slowly
                    # Decrease weight if loss is decreasing quickly
                    self.log_weights[i] -= self.adaptation_rate * gradient
            
            # Keep only recent history
            for key in self.loss_history:
                self.loss_history[key] = self.loss_history[key][-100:]
    
    def get_weights(self) -> Dict[str, float]:
        """Get current loss weights"""
        weights = F.softmax(self.log_weights, dim=0) * 5
        return {
            'ce': weights[0].item(),
            'ast': weights[1].item(),
            'sem': weights[2].item(),
            'trace': weights[3].item(),
            'complex': weights[4].item()
        }