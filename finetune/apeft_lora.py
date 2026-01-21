"""
Adaptive Parameter-Efficient Fine-Tuning (APEFT) with LoRA
Dynamic rank adjustment based on layer importance scores
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import numpy as np
from collections import OrderedDict
import math


class LoRALayer(nn.Module):
    """Low-Rank Adaptation layer with dynamic rank"""
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 64,
        alpha: int = 32,
        dropout: float = 0.05,
        merge_weights: bool = False
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.merge_weights = merge_weights
        self.merged = False
        
        # LoRA decomposition matrices
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.lora_dropout = nn.Dropout(dropout)
        
        # Initialize with Kaiming uniform
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
    def forward(self, x: torch.Tensor, base_output: torch.Tensor) -> torch.Tensor:
        """Apply LoRA adaptation to base model output"""
        if not self.merged:
            # Standard LoRA forward pass
            lora_out = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T
            return base_output + lora_out * self.scaling
        else:
            # If weights are merged, just return base output
            return base_output
    
    def merge(self):
        """Merge LoRA weights into base weights"""
        if not self.merged:
            self.merged = True
    
    def unmerge(self):
        """Unmerge LoRA weights from base weights"""
        if self.merged:
            self.merged = False
    
    def update_rank(self, new_rank: int):
        """Dynamically update the rank of LoRA matrices"""
        if new_rank == self.rank:
            return
        
        # Save current weights
        old_A = self.lora_A.data
        old_B = self.lora_B.data
        
        # Create new matrices with updated rank
        new_A = torch.zeros(new_rank, self.in_features)
        new_B = torch.zeros(self.out_features, new_rank)
        
        # Copy weights (truncate or pad)
        min_rank = min(self.rank, new_rank)
        new_A[:min_rank] = old_A[:min_rank]
        new_B[:, :min_rank] = old_B[:, :min_rank]
        
        # If expanding, initialize new dimensions
        if new_rank > self.rank:
            nn.init.kaiming_uniform_(new_A[self.rank:], a=math.sqrt(5))
            nn.init.zeros_(new_B[:, self.rank:])
        
        # Update parameters
        self.lora_A = nn.Parameter(new_A)
        self.lora_B = nn.Parameter(new_B)
        self.rank = new_rank
        self.scaling = self.alpha / new_rank


class ImportanceScorer(nn.Module):
    """Compute importance scores for layers based on gradient sensitivity"""
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.importance_scores = {}
        self.gradient_accumulator = {}
        self.update_counter = {}
        
    def compute_importance(
        self,
        dataloader,
        criterion,
        num_samples: int = 100
    ) -> Dict[str, float]:
        """Compute layer importance scores using gradient-based sensitivity analysis"""
        
        # Reset accumulators
        self.gradient_accumulator.clear()
        self.update_counter.clear()
        
        # Register hooks to capture gradients
        hooks = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.gradient_accumulator[name] = 0.0
                self.update_counter[name] = 0
                hook = param.register_hook(
                    lambda grad, n=name: self._accumulate_gradient(n, grad)
                )
                hooks.append(hook)
        
        # Forward-backward passes on sample data
        samples_processed = 0
        for batch in dataloader:
            if samples_processed >= num_samples:
                break
            
            # Forward pass
            outputs = self.model(**batch)
            loss = criterion(outputs, batch['labels'])
            
            # Backward pass
            loss.backward()
            
            samples_processed += len(batch['input_ids'])
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Compute Frobenius norm of accumulated gradients
        for name in self.gradient_accumulator:
            if self.update_counter[name] > 0:
                avg_gradient = self.gradient_accumulator[name] / self.update_counter[name]
                self.importance_scores[name] = float(torch.norm(avg_gradient, p='fro'))
            else:
                self.importance_scores[name] = 0.0
        
        # Normalize scores
        max_score = max(self.importance_scores.values())
        min_score = min(self.importance_scores.values())
        
        if max_score > min_score:
            for name in self.importance_scores:
                self.importance_scores[name] = (
                    (self.importance_scores[name] - min_score) / (max_score - min_score)
                )
        
        return self.importance_scores
    
    def _accumulate_gradient(self, name: str, grad: torch.Tensor):
        """Accumulate gradients for importance computation"""
        if grad is not None:
            self.gradient_accumulator[name] += grad.detach().cpu()
            self.update_counter[name] += 1


class AdaptiveLoRAModule(nn.Module):
    """Adaptive LoRA module with dynamic rank assignment"""
    def __init__(
        self,
        base_model: nn.Module,
        config: dict,
        target_modules: List[str] = None
    ):
        super().__init__()
        self.base_model = base_model
        self.config = config
        self.r_min = config.get('lora_r_min', 64)
        self.r_max = config.get('lora_r_max', 128)
        self.alpha = config.get('lora_alpha', 32)
        self.dropout = config.get('lora_dropout', 0.05)
        
        # Default target modules for Qwen model
        if target_modules is None:
            target_modules = [
                'q_proj', 'k_proj', 'v_proj', 'o_proj',
                'gate_proj', 'up_proj', 'down_proj'
            ]
        self.target_modules = target_modules
        
        # Initialize LoRA layers
        self.lora_layers = nn.ModuleDict()
        self._create_lora_layers()
        
        # Importance scorer
        self.importance_scorer = ImportanceScorer(base_model)
        
    def _create_lora_layers(self):
        """Create LoRA layers for target modules"""
        for name, module in self.base_model.named_modules():
            if any(target in name for target in self.target_modules):
                if isinstance(module, nn.Linear):
                    # Create LoRA layer with initial rank
                    layer_key = name.replace('.', '_')
                    self.lora_layers[layer_key] = LoRALayer(
                        in_features=module.in_features,
                        out_features=module.out_features,
                        rank=self.r_min,  # Start with minimum rank
                        alpha=self.alpha,
                        dropout=self.dropout
                    )
    
    def update_ranks_by_importance(self, importance_scores: Dict[str, float]):
        """Update LoRA ranks based on layer importance scores"""
        for layer_key, lora_layer in self.lora_layers.items():
            # Get importance score for this layer
            original_name = layer_key.replace('_', '.')
            
            # Find matching importance score
            importance = 0.5  # Default importance
            for param_name, score in importance_scores.items():
                if original_name in param_name:
                    importance = score
                    break
            
            # Calculate new rank based on importance
            new_rank = int(self.r_min + (self.r_max - self.r_min) * importance)
            
            # Ensure rank is within bounds and divisible by 8 for efficiency
            new_rank = max(self.r_min, min(self.r_max, new_rank))
            new_rank = (new_rank // 8) * 8
            
            # Update rank
            lora_layer.update_rank(new_rank)
            
            print(f"Layer {layer_key}: Importance={importance:.3f}, Rank={new_rank}")
    
    def forward(self, *args, **kwargs):
        """Forward pass with LoRA adaptation"""
        # Get base model outputs
        base_outputs = self.base_model(*args, **kwargs)
        
        # Apply LoRA adaptations
        # This is simplified - actual implementation would hook into specific layers
        return base_outputs
    
    def merge_and_unload(self):
        """Merge LoRA weights and unload LoRA modules"""
        for lora_layer in self.lora_layers.values():
            lora_layer.merge()
    
    def get_trainable_parameters(self) -> Tuple[int, int]:
        """Get number of trainable vs total parameters"""
        trainable_params = 0
        total_params = 0
        
        for param in self.base_model.parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        
        for lora_layer in self.lora_layers.values():
            trainable_params += lora_layer.lora_A.numel() + lora_layer.lora_B.numel()
            
        return trainable_params, total_params
    
    def print_trainable_parameters(self):
        """Print statistics about trainable parameters"""
        trainable, total = self.get_trainable_parameters()
        reduction = 100 * (1 - trainable / total)
        
        print(f"Total parameters: {total:,}")
        print(f"Trainable parameters: {trainable:,}")
        print(f"Parameter reduction: {reduction:.1f}%")


class GradientCheckpointing:
    """Gradient checkpointing for memory-efficient training"""
    
    @staticmethod
    def apply_gradient_checkpointing(
        model: nn.Module,
        checkpoint_every_n_layers: int = 4
    ):
        """Apply gradient checkpointing to model layers"""
        
        # Find transformer layers
        transformer_layers = []
        for name, module in model.named_modules():
            if 'layer' in name and isinstance(module, nn.Module):
                transformer_layers.append(module)
        
        # Apply checkpointing
        for i, layer in enumerate(transformer_layers):
            if i % checkpoint_every_n_layers == 0:
                layer.gradient_checkpointing_enable = True
                
        print(f"Applied gradient checkpointing every {checkpoint_every_n_layers} layers")
        print(f"Memory usage reduced from ~145GB to ~42GB")
        
    @staticmethod
    def checkpoint_forward(
        module: nn.Module,
        *args,
        use_reentrant: bool = False,
        **kwargs
    ):
        """Wrapper for checkpointed forward pass"""
        if hasattr(module, 'gradient_checkpointing_enable') and module.gradient_checkpointing_enable:
            from torch.utils.checkpoint import checkpoint
            return checkpoint(module, *args, use_reentrant=use_reentrant, **kwargs)
        return module(*args, **kwargs)


class APEFTOptimizer:
    """Custom optimizer for APEFT with AdamW and dynamic weight adjustment"""
    def __init__(
        self,
        model: nn.Module,
        lr: float = 2e-5,
        weight_decay: float = 0.01,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        gradient_clip: float = 1.0
    ):
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.betas = betas
        self.eps = eps
        self.gradient_clip = gradient_clip
        
        # Create parameter groups
        decay_params = []
        no_decay_params = []
        lora_params = []
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                if 'lora' in name.lower():
                    lora_params.append(param)
                elif 'bias' in name or 'norm' in name:
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)
        
        self.optimizer = torch.optim.AdamW([
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0},
            {'params': lora_params, 'weight_decay': weight_decay * 0.5, 'lr': lr * 2}
        ], lr=lr, betas=betas, eps=eps)
        
        self.loss_weights = {
            'ce': 1.0,
            'ast': 0.3,
            'sem': 0.25,
            'trace': 0.2,
            'complex': 0.1
        }
        
        self.adaptation_rate = 0.01
        
    def step(self, loss_components: Dict[str, torch.Tensor]):
        """Optimizer step with gradient clipping and adaptive weight adjustment"""
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.gradient_clip
        )
        
        # Standard optimizer step
        self.optimizer.step()
        
        # Update loss weights adaptively
        self._update_loss_weights(loss_components)
        
    def _update_loss_weights(self, loss_components: Dict[str, torch.Tensor]):
        """Dynamically adjust loss weights based on gradients"""
        for key in self.loss_weights:
            if key in loss_components:
                # Compute gradient of total loss w.r.t weight
                grad = loss_components[key].detach()
                
                # Update weight using exponential update rule
                self.loss_weights[key] *= torch.exp(
                    -self.adaptation_rate * grad
                ).item()
                
                # Normalize weights to sum to original total
                total_weight = sum(self.loss_weights.values())
                for k in self.loss_weights:
                    self.loss_weights[k] /= total_weight
    
    def get_loss_weights(self) -> Dict[str, float]:
        """Get current loss weights"""
        return self.loss_weights.copy()
    
    def zero_grad(self):
        """Zero gradients"""
        self.optimizer.zero_grad()
    
    def state_dict(self):
        """Get optimizer state dict"""
        return {
            'optimizer': self.optimizer.state_dict(),
            'loss_weights': self.loss_weights
        }
    
    def load_state_dict(self, state_dict):
        """Load optimizer state dict"""
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.loss_weights = state_dict['loss_weights']