"""
Main training script for CodeForge-Quantum
Implements the complete training pipeline with all components
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import json
import os
import sys
import argparse
import logging
import wandb
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import yaml
from tqdm import tqdm
import time
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import CodeForge modules
from models.codeforge_quantum import CodeForgeQuantum, CodeForgeConfig
from fine_tuning.apeft_lora import (
    AdaptiveLoRAModule,
    GradientCheckpointing,
    APEFTOptimizer
)
from rag.vector_database import (
    FAISSVectorDatabase,
    DensePassageRetriever,
    CrossAttentionRetrieval,
    RetrievalAugmentedGeneration,
    CodeSnippet
)
from prompting.chain_of_thought import (
    ChainOfThoughtLSTM,
    StructuredReasoning,
    PromptEngineering
)
from execution.abstract_interpreter import (
    AbstractInterpreter,
    TraceAnalyzer,
    CompilerFeedback
)
from losses.multi_objective_loss import MultiObjectiveLoss
from evaluation.metrics import MetricsAggregator
from preprocessing.code_augmentation import CodeAugmentation
from preprocessing.curriculum_scheduler import CurriculumScheduler


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Complete training configuration"""
    # Model configuration
    model_name: str = "Qwen/Qwen-72B"
    num_layers: int = 80
    num_heads: int = 64
    hidden_dim: int = 8192
    vocab_size: int = 65536
    
    # Training parameters
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    num_epochs: int = 15
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    
    # LoRA configuration
    lora_r_min: int = 64
    lora_r_max: int = 128
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    
    # RAG configuration
    retrieval_top_k: int = 10
    vector_db_path: str = "data/code_database"
    faiss_index_type: str = "HNSW"
    
    # Loss weights
    weight_ce: float = 1.0
    weight_ast: float = 0.3
    weight_sem: float = 0.25
    weight_trace: float = 0.2
    weight_complex: float = 0.1
    
    # Data paths
    train_data_path: str = "data/train.json"
    val_data_path: str = "data/val.json"
    test_data_path: str = "data/test.json"
    
    # Output paths
    output_dir: str = "outputs"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    
    # Hardware configuration
    device: str = "cuda"
    num_gpus: int = 8
    fp16: bool = True
    gradient_checkpointing: bool = True
    checkpoint_every_n_layers: int = 4
    
    # Evaluation
    eval_steps: int = 500
    save_steps: int = 1000
    logging_steps: int = 10
    
    # Wandb
    use_wandb: bool = True
    wandb_project: str = "codeforge-quantum"
    wandb_entity: Optional[str] = None
    
    @classmethod
    def from_yaml(cls, path: str):
        """Load configuration from YAML file"""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    def save(self, path: str):
        """Save configuration to YAML file"""
        with open(path, 'w') as f:
            yaml.dump(asdict(self), f, default_flow_style=False)


class CodeDataset(Dataset):
    """Dataset for code generation tasks"""
    
    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 2048,
        augmentation: bool = True
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.augmentation = augmentation
        
        # Load data
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        
        # Initialize augmentation
        if augmentation:
            self.augmenter = CodeAugmentation()
        
        logger.info(f"Loaded {len(self.data)} samples from {data_path}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Extract fields
        problem = item['problem']
        solution = item['solution']
        difficulty = item.get('difficulty', 'medium')
        test_cases = item.get('test_cases', [])
        
        # Apply augmentation if enabled
        if self.augmentation and np.random.random() < 0.5:
            solution = self.augmenter.augment(solution)
        
        # Tokenize natural language description
        nl_tokens = self.tokenizer(
            problem,
            max_length=self.max_length // 2,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        # Tokenize solution
        solution_tokens = self.tokenizer(
            solution,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'nl_input_ids': nl_tokens['input_ids'].squeeze(),
            'nl_attention_mask': nl_tokens['attention_mask'].squeeze(),
            'target_ids': solution_tokens['input_ids'].squeeze(),
            'target_attention_mask': solution_tokens['attention_mask'].squeeze(),
            'difficulty': difficulty,
            'test_cases': test_cases,
            'reference_code': solution
        }


class Trainer:
    """Main trainer class for CodeForge-Quantum"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Initialize model components
        self._setup_model()
        self._setup_rag()
        self._setup_execution()
        self._setup_optimization()
        self._setup_data()
        self._setup_logging()
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_metric = 0.0
        
    def _setup_model(self):
        """Initialize model and apply APEFT"""
        logger.info("Setting up CodeForge-Quantum model...")
        
        # Model configuration
        model_config = CodeForgeConfig(
            model_name=self.config.model_name,
            num_layers=self.config.num_layers,
            num_heads=self.config.num_heads,
            hidden_dim=self.config.hidden_dim,
            vocab_size=self.config.vocab_size,
            lora_r_min=self.config.lora_r_min,
            lora_r_max=self.config.lora_r_max,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout
        )
        
        # Initialize model
        self.model = CodeForgeQuantum(model_config)
        
        # Apply LoRA
        self.lora_module = AdaptiveLoRAModule(
            self.model,
            asdict(model_config)
        )
        
        # Apply gradient checkpointing
        if self.config.gradient_checkpointing:
            GradientCheckpointing.apply_gradient_checkpointing(
                self.model,
                self.config.checkpoint_every_n_layers
            )
        
        # Move to device
        self.model.to(self.device)
        
        # Setup DDP if using multiple GPUs
        if self.config.num_gpus > 1:
            self.model = DDP(self.model)
        
        # Print parameter statistics
        self.lora_module.print_trainable_parameters()
    
    def _setup_rag(self):
        """Initialize RAG components"""
        logger.info("Setting up RAG system...")
        
        # Initialize vector database
        self.vector_db = FAISSVectorDatabase(
            dimension=768,
            index_type=self.config.faiss_index_type
        )
        
        # Load pre-built database if exists
        if os.path.exists(self.config.vector_db_path):
            self.vector_db.load(self.config.vector_db_path)
            logger.info(f"Loaded vector database with {self.vector_db.stats['total_snippets']} snippets")
        
        # Initialize retriever
        self.retriever = DensePassageRetriever()
        self.retriever.to(self.device)
        
        # Initialize cross-attention
        self.cross_attention = CrossAttentionRetrieval(
            hidden_dim=self.config.hidden_dim
        )
        self.cross_attention.to(self.device)
        
        # Setup RAG system
        self.rag_system = RetrievalAugmentedGeneration(
            self.vector_db,
            self.retriever,
            self.cross_attention,
            top_k=self.config.retrieval_top_k
        )
    
    def _setup_execution(self):
        """Initialize execution-aware components"""
        logger.info("Setting up execution-aware components...")
        
        self.abstract_interpreter = AbstractInterpreter(
            hidden_dim=self.config.hidden_dim
        )
        self.abstract_interpreter.to(self.device)
        
        self.trace_analyzer = TraceAnalyzer(
            hidden_dim=self.config.hidden_dim
        )
        self.trace_analyzer.to(self.device)
        
        self.compiler_feedback = CompilerFeedback()
        
        # Chain-of-thought reasoning
        self.cot_reasoner = StructuredReasoning(
            hidden_dim=self.config.hidden_dim
        )
    
    def _setup_optimization(self):
        """Setup optimizer and loss"""
        logger.info("Setting up optimization...")
        
        # Multi-objective loss
        self.criterion = MultiObjectiveLoss(asdict(self.config))
        
        # APEFT optimizer
        self.optimizer = APEFTOptimizer(
            self.model,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            gradient_clip=self.config.max_grad_norm
        )
        
        # Learning rate scheduler
        from transformers import get_linear_schedule_with_warmup
        
        total_steps = (
            len(self.train_loader) // self.config.gradient_accumulation_steps * 
            self.config.num_epochs
        )
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=total_steps
        )
    
    def _setup_data(self):
        """Setup data loaders"""
        logger.info("Setting up data loaders...")
        
        # Load tokenizer
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        
        # Create datasets
        self.train_dataset = CodeDataset(
            self.config.train_data_path,
            self.tokenizer,
            augmentation=True
        )
        
        self.val_dataset = CodeDataset(
            self.config.val_data_path,
            self.tokenizer,
            augmentation=False
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size * 2,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        # Setup curriculum learning
        self.curriculum_scheduler = CurriculumScheduler(
            total_epochs=self.config.num_epochs,
            warmup_epochs=3
        )
    
    def _setup_logging(self):
        """Setup logging and monitoring"""
        logger.info("Setting up logging...")
        
        # Create directories
        os.makedirs(self.config.output_dir, exist_ok=True)
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        os.makedirs(self.config.log_dir, exist_ok=True)
        
        # TensorBoard
        self.writer = SummaryWriter(self.config.log_dir)
        
        # Weights & Biases
        if self.config.use_wandb:
            wandb.init(
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                config=asdict(self.config),
                name=f"codeforge_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
        
        # Metrics aggregator
        self.metrics_aggregator = MetricsAggregator()
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0.0
        epoch_metrics = defaultdict(float)
        
        # Get curriculum difficulty
        difficulty_weights = self.curriculum_scheduler.get_difficulty_weights(
            self.current_epoch
        )
        
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch + 1}/{self.config.num_epochs}"
        )
        
        for step, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Apply curriculum learning (filter by difficulty)
            if np.random.random() > difficulty_weights.get(batch['difficulty'][0], 1.0):
                continue
            
            # Retrieve relevant code (RAG)
            retrieved_context = None
            if self.rag_system and np.random.random() < 0.7:  # Use RAG 70% of the time
                problem_text = self.tokenizer.decode(
                    batch['nl_input_ids'][0],
                    skip_special_tokens=True
                )
                _, retrieved_snippets = self.rag_system.retrieve_and_augment(
                    problem_text,
                    torch.zeros(1, self.config.hidden_dim).to(self.device)
                )
                # Encode retrieved snippets
                if retrieved_snippets:
                    retrieved_text = retrieved_snippets[0].code
                    retrieved_tokens = self.tokenizer(
                        retrieved_text,
                        max_length=512,
                        truncation=True,
                        return_tensors='pt'
                    ).to(self.device)
                    retrieved_context = retrieved_tokens['input_ids']
            
            # Forward pass
            outputs = self.model(
                nl_input_ids=batch['nl_input_ids'],
                nl_attention_mask=batch['nl_attention_mask'],
                retrieved_context=retrieved_context,
                target_ids=batch['target_ids']
            )
            
            # Compute losses
            loss_components = self.criterion(
                outputs,
                {
                    'token_ids': batch['target_ids'],
                    'reference_code': batch['reference_code'][0] if len(batch['reference_code']) > 0 else "",
                    'test_inputs': batch.get('test_cases', [])
                }
            )
            
            loss = loss_components.total_loss / self.config.gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Gradient accumulation
            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm
                )
                
                # Optimizer step
                self.optimizer.step(loss_components.to_dict())
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                self.global_step += 1
            
            # Update metrics
            epoch_loss += loss.item() * self.config.gradient_accumulation_steps
            for key, value in loss_components.to_dict().items():
                if isinstance(value, torch.Tensor):
                    epoch_metrics[key] += value.item()
            
            # Logging
            if self.global_step % self.config.logging_steps == 0:
                self._log_metrics(loss_components.to_dict())
            
            # Evaluation
            if self.global_step % self.config.eval_steps == 0:
                eval_metrics = self.evaluate()
                self._log_eval_metrics(eval_metrics)
                
                # Save best model
                if eval_metrics['pass_at_1'] > self.best_metric:
                    self.best_metric = eval_metrics['pass_at_1']
                    self.save_checkpoint('best')
            
            # Regular checkpoint
            if self.global_step % self.config.save_steps == 0:
                self.save_checkpoint(f'step_{self.global_step}')
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
            })
        
        # Return epoch metrics
        num_batches = len(self.train_loader)
        return {
            'loss': epoch_loss / num_batches,
            **{k: v / num_batches for k, v in epoch_metrics.items()}
        }
    
    def evaluate(self):
        """Evaluate on validation set"""
        self.model.eval()
        
        all_problems = []
        all_generated = []
        all_references = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluating"):
                # Move to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Generate solutions
                generated = self.model.generate(
                    nl_input_ids=batch['nl_input_ids'],
                    nl_attention_mask=batch['nl_attention_mask'],
                    max_length=512,
                    num_beams=5
                )
                
                # Decode
                for i in range(len(generated)):
                    gen_code = self.tokenizer.decode(
                        generated[i],
                        skip_special_tokens=True
                    )
                    all_generated.append([gen_code])  # List of solutions per problem
                    all_references.append(batch['reference_code'][i])
                    all_problems.append({
                        'test_cases': batch.get('test_cases', [[]])[i]
                    })
        
        # Compute metrics
        metrics = self.metrics_aggregator.evaluate(
            all_problems,
            all_generated,
            all_references,
            k_values=[1, 5, 10]
        )
        
        self.model.train()
        return metrics['summary']
    
    def save_checkpoint(self, name: str):
        """Save model checkpoint"""
        checkpoint_path = os.path.join(self.config.checkpoint_dir, name)
        os.makedirs(checkpoint_path, exist_ok=True)
        
        # Save model state
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'current_epoch': self.current_epoch,
            'best_metric': self.best_metric,
            'config': asdict(self.config)
        }, os.path.join(checkpoint_path, 'checkpoint.pt'))
        
        # Save LoRA weights separately
        self.lora_module.save_pretrained(checkpoint_path)
        
        logger.info(f"Saved checkpoint: {checkpoint_path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(
            os.path.join(path, 'checkpoint.pt'),
            map_location=self.device
        )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['global_step']
        self.current_epoch = checkpoint['current_epoch']
        self.best_metric = checkpoint['best_metric']
        
        logger.info(f"Loaded checkpoint from {path}")
    
    def _log_metrics(self, metrics: Dict[str, Any]):
        """Log training metrics"""
        # TensorBoard
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f'train/{key}', value, self.global_step)
        
        # Weights & Biases
        if self.config.use_wandb:
            wandb.log({f'train/{k}': v for k, v in metrics.items()}, step=self.global_step)
    
    def _log_eval_metrics(self, metrics: Dict[str, Any]):
        """Log evaluation metrics"""
        # TensorBoard
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f'eval/{key}', value, self.global_step)
        
        # Weights & Biases
        if self.config.use_wandb:
            wandb.log({f'eval/{k}': v for k, v in metrics.items()}, step=self.global_step)
        
        # Print to console
        logger.info(f"Evaluation at step {self.global_step}: {metrics}")
    
    def train(self):
        """Main training loop"""
        logger.info("Starting training...")
        
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            logger.info(f"Starting epoch {epoch + 1}/{self.config.num_epochs}")
            
            # Train for one epoch
            epoch_metrics = self.train_epoch()
            
            # Log epoch metrics
            logger.info(f"Epoch {epoch + 1} metrics: {epoch_metrics}")
            
            # Update curriculum
            self.curriculum_scheduler.step()
            
            # Update LoRA ranks based on importance
            if epoch % 3 == 0 and epoch > 0:
                logger.info("Updating LoRA ranks based on importance...")
                importance_scores = self.lora_module.importance_scorer.compute_importance(
                    self.train_loader,
                    self.criterion,
                    num_samples=100
                )
                self.lora_module.update_ranks_by_importance(importance_scores)
            
            # Save epoch checkpoint
            self.save_checkpoint(f'epoch_{epoch + 1}')
        
        logger.info("Training completed!")
        
        # Final evaluation
        final_metrics = self.evaluate()
        logger.info(f"Final evaluation metrics: {final_metrics}")
        
        # Compare with baselines
        from baselines.codellama_starcoder_baselines import BaselineComparison
        comparison = BaselineComparison()
        comparison_results = comparison.compare_all()
        logger.info(f"Comparison with baselines:\n{comparison_results}")
        
        # Save final model
        self.save_checkpoint('final')
        
        # Close logging
        self.writer.close()
        if self.config.use_wandb:
            wandb.finish()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Train CodeForge-Quantum")
    parser.add_argument(
        '--config',
        type=str,
        default='config/training_config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    parser.add_argument(
        '--eval_only',
        action='store_true',
        help='Only run evaluation'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    if os.path.exists(args.config):
        config = TrainingConfig.from_yaml(args.config)
    else:
        config = TrainingConfig()
        config.save(args.config)
        logger.info(f"Created default config at {args.config}")
    
    # Initialize trainer
    trainer = Trainer(config)
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Run evaluation only if specified
    if args.eval_only:
        metrics = trainer.evaluate()
        print(f"Evaluation metrics: {metrics}")
    else:
        # Start training
        trainer.train()


if __name__ == "__main__":
    main()