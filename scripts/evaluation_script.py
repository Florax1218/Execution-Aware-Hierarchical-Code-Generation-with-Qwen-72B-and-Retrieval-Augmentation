"""
Evaluation script for CodeForge-Quantum
Comprehensive evaluation across all metrics and difficulty levels
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import json
import argparse
import os
import sys
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from tqdm import tqdm
import time
from datetime import datetime
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.codeforge_quantum import CodeForgeQuantum, CodeForgeConfig
from evaluation.metrics import (
    PassAtK,
    EnhancedCodeBLEU,
    SemanticConsistencyScore,
    CompilationSuccessRate,
    MetricsAggregator
)
from baselines.gpt4_baseline import GPT4ComparisonBenchmark
from baselines.codellama_starcoder_baselines import BaselineComparison
from scripts.train import CodeDataset
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CodeForgeEvaluator:
    """Comprehensive evaluator for CodeForge-Quantum"""
    
    def __init__(
        self,
        model_checkpoint: str,
        test_data_path: str,
        device: str = "cuda",
        output_dir: str = "evaluation_results"
    ):
        self.model_checkpoint = model_checkpoint
        self.test_data_path = test_data_path
        self.device = torch.device(device)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model
        self._load_model()
        
        # Initialize metrics
        self.pass_at_k = PassAtK()
        self.e_codebleu = EnhancedCodeBLEU()
        self.scs = SemanticConsistencyScore()
        self.csr = CompilationSuccessRate()
        self.metrics_aggregator = MetricsAggregator()
        
        # Initialize baseline comparisons
        self.gpt4_benchmark = GPT4ComparisonBenchmark()
        self.baseline_comparison = BaselineComparison()
        
        # Results storage
        self.results = {
            'overall': {},
            'by_difficulty': {},
            'by_category': {},
            'individual_problems': [],
            'comparison': {}
        }
        
    def _load_model(self):
        """Load model from checkpoint"""
        logger.info(f"Loading model from {self.model_checkpoint}")
        
        # Load checkpoint
        checkpoint = torch.load(
            os.path.join(self.model_checkpoint, 'checkpoint.pt'),
            map_location=self.device
        )
        
        # Reconstruct config
        config = CodeForgeConfig(**checkpoint['config'])
        
        # Initialize model
        self.model = CodeForgeQuantum(config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info("Model loaded successfully")
    
    def evaluate(
        self,
        batch_size: int = 8,
        k_values: List[int] = [1, 5, 10],
        num_samples: int = 100,
        save_outputs: bool = True
    ):
        """Run comprehensive evaluation"""
        logger.info(f"Starting evaluation on {num_samples} problems")
        
        # Load test data
        test_dataset = CodeDataset(
            self.test_data_path,
            self.tokenizer,
            augmentation=False
        )
        
        # Limit to num_samples
        if num_samples < len(test_dataset):
            indices = np.random.choice(len(test_dataset), num_samples, replace=False)
            test_dataset.data = [test_dataset.data[i] for i in indices]
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4
        )
        
        # Collect all results
        all_problems = []
        all_generated = []
        all_references = []
        all_metadata = []
        
        # Evaluation loop
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
                # Move to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}
                
                # Generate solutions
                generated_ids = self._generate_batch(batch, k=max(k_values))
                
                # Process each sample
                for i in range(len(generated_ids)):
                    # Decode solutions
                    solutions = []
                    for k in range(len(generated_ids[i])):
                        solution = self.tokenizer.decode(
                            generated_ids[i][k],
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=True
                        )
                        solutions.append(solution)
                    
                    # Store results
                    all_generated.append(solutions)
                    all_references.append(batch['reference_code'][i] if i < len(batch['reference_code']) else "")
                    
                    problem = {
                        'difficulty': batch['difficulty'][i] if i < len(batch['difficulty']) else 'medium',
                        'test_cases': batch.get('test_cases', [[]])[i] if i < len(batch.get('test_cases', [[]])) else []
                    }
                    all_problems.append(problem)
                    all_metadata.append({
                        'batch_idx': batch_idx,
                        'sample_idx': i
                    })
        
        # Compute metrics
        logger.info("Computing metrics...")
        
        # Pass@k
        pass_at_k_results = self.pass_at_k.compute(
            all_problems,
            all_generated,
            k_values=k_values
        )
        self.results['overall']['pass_at_k'] = pass_at_k_results
        
        # Enhanced CodeBLEU
        e_cb_scores = []
        for gen_sols, ref_sol in zip(all_generated, all_references):
            if gen_sols and ref_sol:
                score = self.e_codebleu.compute(gen_sols[0], ref_sol)
                e_cb_scores.append(score)
        self.results['overall']['e_codebleu'] = np.mean(e_cb_scores) if e_cb_scores else 0.0
        
        # Compilation Success Rate
        all_generated_flat = [sol for sols in all_generated for sol in sols]
        csr_score, error_dist = self.csr.compute(all_generated_flat)
        self.results['overall']['csr_ec'] = csr_score
        self.results['overall']['error_distribution'] = error_dist
        
        # Semantic Consistency (sample a subset for efficiency)
        scs_scores = []
        sample_indices = np.random.choice(
            len(all_generated),
            min(20, len(all_generated)),
            replace=False
        )
        for idx in sample_indices:
            if all_generated[idx] and all_references[idx]:
                try:
                    # Create function objects (simplified)
                    gen_func = self._create_function(all_generated[idx][0])
                    ref_func = self._create_function(all_references[idx])
                    
                    if gen_func and ref_func:
                        # Generate test inputs
                        test_inputs = self._generate_test_inputs(all_problems[idx])
                        score = self.scs.compute(gen_func, ref_func, test_inputs)
                        scs_scores.append(score)
                except Exception as e:
                    logger.debug(f"SCS computation failed: {e}")
        
        self.results['overall']['scs'] = np.mean(scs_scores) if scs_scores else 0.0
        
        # Results by difficulty
        self._compute_difficulty_breakdown(
            all_problems, all_generated, all_references
        )
        
        # Category breakdown (if available)
        self._compute_category_breakdown(
            all_problems, all_generated, all_references
        )
        
        # Compare with baselines
        self._compare_with_baselines()
        
        # Generate report
        self._generate_report()
        
        # Save outputs if requested
        if save_outputs:
            self._save_outputs(
                all_problems, all_generated, all_references, all_metadata
            )
        
        logger.info("Evaluation completed!")
        return self.results
    
    def _generate_batch(
        self,
        batch: Dict,
        k: int = 1
    ) -> List[List[torch.Tensor]]:
        """Generate k solutions for each problem in batch"""
        batch_size = batch['nl_input_ids'].size(0)
        all_solutions = []
        
        for i in range(batch_size):
            solutions = []
            for _ in range(k):
                # Generate single solution
                generated = self.model.generate(
                    nl_input_ids=batch['nl_input_ids'][i:i+1],
                    nl_attention_mask=batch['nl_attention_mask'][i:i+1],
                    max_length=512,
                    temperature=0.7,
                    num_beams=5
                )
                solutions.append(generated[0])
            all_solutions.append(solutions)
        
        return all_solutions
    
    def _compute_difficulty_breakdown(
        self,
        problems: List[Dict],
        generated: List[List[str]],
        references: List[str]
    ):
        """Compute metrics breakdown by difficulty"""
        difficulties = ['easy', 'medium', 'hard']
        
        for difficulty in difficulties:
            # Filter by difficulty
            indices = [
                i for i, p in enumerate(problems)
                if p.get('difficulty') == difficulty
            ]
            
            if not indices:
                continue
            
            diff_problems = [problems[i] for i in indices]
            diff_generated = [generated[i] for i in indices]
            diff_references = [references[i] for i in indices]
            
            # Compute Pass@1
            pass_at_1 = self.pass_at_k.compute(
                diff_problems,
                diff_generated,
                k_values=[1]
            )
            
            # Compute E-CodeBLEU
            e_cb_scores = []
            for gen, ref in zip(diff_generated, diff_references):
                if gen and ref:
                    score = self.e_codebleu.compute(gen[0], ref)
                    e_cb_scores.append(score)
            
            # Store results
            self.results['by_difficulty'][difficulty] = {
                'pass_at_1': pass_at_1.get(1, 0.0),
                'e_codebleu': np.mean(e_cb_scores) if e_cb_scores else 0.0,
                'num_problems': len(indices)
            }
    
    def _compute_category_breakdown(
        self,
        problems: List[Dict],
        generated: List[List[str]],
        references: List[str]
    ):
        """Compute metrics breakdown by problem category"""
        # Extract categories (if available in data)
        categories = defaultdict(list)
        
        for i, problem in enumerate(problems):
            category = problem.get('category', 'uncategorized')
            categories[category].append(i)
        
        for category, indices in categories.items():
            if len(indices) < 3:  # Skip categories with too few samples
                continue
            
            cat_problems = [problems[i] for i in indices]
            cat_generated = [generated[i] for i in indices]
            cat_references = [references[i] for i in indices]
            
            # Compute Pass@1
            pass_at_1 = self.pass_at_k.compute(
                cat_problems,
                cat_generated,
                k_values=[1]
            )
            
            self.results['by_category'][category] = {
                'pass_at_1': pass_at_1.get(1, 0.0),
                'num_problems': len(indices)
            }
    
    def _compare_with_baselines(self):
        """Compare with baseline models"""
        # Prepare CodeForge metrics
        cf_metrics = {
            'pass_at_1': self.results['overall']['pass_at_k'].get(1, 0.0),
            'e_codebleu': self.results['overall']['e_codebleu'],
            'scs': self.results['overall']['scs'],
            'csr': self.results['overall']['csr_ec']
        }
        
        # Compare with GPT-4
        gpt4_comparison = self.gpt4_benchmark.compare_with_codeforge(cf_metrics)
        self.results['comparison']['gpt4'] = gpt4_comparison
        
        # Compare with all baselines
        all_comparisons = self.baseline_comparison.compare_all()
        self.results['comparison']['all_baselines'] = all_comparisons
        
        # Get rankings
        rankings = {}
        for metric in ['pass_at_1', 'e_codebleu', 'scs', 'csr']:
            ranking = self.baseline_comparison.get_ranking(metric)
            rankings[metric] = ranking
        self.results['comparison']['rankings'] = rankings
    
    def _generate_report(self):
        """Generate comprehensive evaluation report"""
        report_path = self.output_dir / f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("CODEFORGE-QUANTUM EVALUATION REPORT\n")
            f.write("="*80 + "\n\n")
            
            # Overall metrics
            f.write("OVERALL METRICS\n")
            f.write("-"*40 + "\n")
            for k, v in self.results['overall']['pass_at_k'].items():
                f.write(f"Pass@{k}: {v:.3f}\n")
            f.write(f"E-CodeBLEU: {self.results['overall']['e_codebleu']:.3f}\n")
            f.write(f"SCS: {self.results['overall']['scs']:.3f}\n")
            f.write(f"CSR-EC: {self.results['overall']['csr_ec']:.3f}\n")
            f.write("\n")
            
            # Difficulty breakdown
            f.write("PERFORMANCE BY DIFFICULTY\n")
            f.write("-"*40 + "\n")
            f.write(f"{'Difficulty':<10} {'Pass@1':<10} {'E-CB':<10} {'Count':<10}\n")
            for diff, metrics in self.results['by_difficulty'].items():
                f.write(f"{diff:<10} {metrics['pass_at_1']:<10.3f} ")
                f.write(f"{metrics['e_codebleu']:<10.3f} ")
                f.write(f"{metrics['num_problems']:<10}\n")
            f.write("\n")
            
            # Baseline comparison
            f.write("BASELINE COMPARISON\n")
            f.write("-"*40 + "\n")
            
            if 'gpt4' in self.results['comparison']:
                comparison = self.results['comparison']['gpt4']
                f.write("\nvs GPT-4:\n")
                for metric, data in comparison.get('detailed_comparison', {}).items():
                    f.write(f"  {metric}: {data['codeforge_quantum']:.3f} ")
                    f.write(f"(+{data['improvement_over_gpt4']:.1f}%)\n")
                
                f.write(f"\nSummary: {comparison.get('summary', {}).get('average_improvement_over_gpt4', 'N/A')}\n")
            
            # Rankings
            if 'rankings' in self.results['comparison']:
                f.write("\nMODEL RANKINGS\n")
                f.write("-"*40 + "\n")
                for metric, ranking in self.results['comparison']['rankings'].items():
                    f.write(f"\n{metric.upper()}:\n")
                    for i, (model, score) in enumerate(ranking[:5], 1):
                        f.write(f"  {i}. {model:<20} {score:.3f}\n")
            
            # Error analysis
            if 'error_distribution' in self.results['overall']:
                f.write("\nERROR DISTRIBUTION\n")
                f.write("-"*40 + "\n")
                for error_type, count in self.results['overall']['error_distribution'].items():
                    f.write(f"  {error_type}: {count}\n")
        
        logger.info(f"Report saved to {report_path}")
    
    def _save_outputs(
        self,
        problems: List[Dict],
        generated: List[List[str]],
        references: List[str],
        metadata: List[Dict]
    ):
        """Save detailed outputs"""
        outputs_path = self.output_dir / f"outputs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        outputs = []
        for i in range(len(problems)):
            output = {
                'id': i,
                'problem': problems[i],
                'generated_solutions': generated[i],
                'reference_solution': references[i],
                'metadata': metadata[i]
            }
            outputs.append(output)
        
        with open(outputs_path, 'w') as f:
            json.dump(outputs, f, indent=2)
        
        logger.info(f"Outputs saved to {outputs_path}")
    
    def _create_function(self, code: str) -> Optional[callable]:
        """Create executable function from code string"""
        try:
            # Simple approach - assumes function named 'solution'
            namespace = {}
            exec(code, namespace)
            if 'solution' in namespace:
                return namespace['solution']
            # Try to find any function
            for name, obj in namespace.items():
                if callable(obj) and not name.startswith('_'):
                    return obj
            return None
        except:
            return None
    
    def _generate_test_inputs(self, problem: Dict) -> List[Any]:
        """Generate test inputs for a problem"""
        # Use provided test cases if available
        if problem.get('test_cases'):
            inputs = []
            for tc in problem['test_cases']:
                try:
                    # Parse input (simplified)
                    input_str = tc.get('input', '')
                    # Try to evaluate as Python literal
                    input_val = eval(input_str) if input_str else []
                    inputs.append(input_val if isinstance(input_val, tuple) else (input_val,))
                except:
                    pass
            return inputs[:10]  # Limit to 10 test cases
        
        # Generate random inputs (fallback)
        return [
            (i,) for i in range(10)
        ]
    
    def visualize_results(self):
        """Create visualizations of evaluation results"""
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Overall metrics comparison
        metrics_names = list(self.results['overall']['pass_at_k'].keys())
        metrics_values = list(self.results['overall']['pass_at_k'].values())
        
        axes[0, 0].bar([f"Pass@{k}" for k in metrics_names], metrics_values)
        axes[0, 0].set_title("Pass@k Performance")
        axes[0, 0].set_ylabel("Score")
        axes[0, 0].set_ylim(0, 1)
        
        # 2. Difficulty breakdown
        if self.results['by_difficulty']:
            difficulties = list(self.results['by_difficulty'].keys())
            pass_scores = [self.results['by_difficulty'][d]['pass_at_1'] for d in difficulties]
            
            axes[0, 1].bar(difficulties, pass_scores)
            axes[0, 1].set_title("Performance by Difficulty")
            axes[0, 1].set_ylabel("Pass@1 Score")
            axes[0, 1].set_ylim(0, 1)
        
        # 3. Baseline comparison
        if 'all_baselines' in self.results['comparison']:
            models = ['CodeForge-Quantum', 'GPT-4', 'GPT-4-Turbo', 'CodeLlama-70B', 'StarCoder2']
            pass_scores = [
                self.results['overall']['pass_at_k'].get(1, 0.0),
                0.746, 0.768, 0.721, 0.739  # From paper
            ]
            
            axes[1, 0].bar(models, pass_scores)
            axes[1, 0].set_title("Model Comparison (Pass@1)")
            axes[1, 0].set_ylabel("Score")
            axes[1, 0].set_xticklabels(models, rotation=45, ha='right')
            axes[1, 0].set_ylim(0, 1)
        
        # 4. Error distribution
        if 'error_distribution' in self.results['overall']:
            errors = list(self.results['overall']['error_distribution'].keys())
            counts = list(self.results['overall']['error_distribution'].values())
            
            if errors:
                axes[1, 1].pie(counts, labels=errors, autopct='%1.1f%%')
                axes[1, 1].set_title("Error Distribution")
        
        plt.suptitle("CodeForge-Quantum Evaluation Results", fontsize=16)
        plt.tight_layout()
        
        # Save figure
        fig_path = self.output_dir / f"evaluation_plots_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plots saved to {fig_path}")
        
        plt.show()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Evaluate CodeForge-Quantum")
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--test-data',
        type=str,
        required=True,
        help='Path to test data'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='evaluation_results',
        help='Output directory for results'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help='Batch size for evaluation'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=100,
        help='Number of test samples to evaluate'
    )
    parser.add_argument(
        '--k-values',
        type=int,
        nargs='+',
        default=[1, 5, 10],
        help='Values of k for Pass@k metric'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Generate visualization plots'
    )
    parser.add_argument(
        '--save-outputs',
        action='store_true',
        help='Save detailed outputs'
    )
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = CodeForgeEvaluator(
        model_checkpoint=args.checkpoint,
        test_data_path=args.test_data,
        device=args.device,
        output_dir=args.output_dir
    )
    
    # Run evaluation
    results = evaluator.evaluate(
        batch_size=args.batch_size,
        k_values=args.k_values,
        num_samples=args.num_samples,
        save_outputs=args.save_outputs
    )
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Pass@1: {results['overall']['pass_at_k'].get(1, 0.0):.3f}")
    print(f"Pass@5: {results['overall']['pass_at_k'].get(5, 0.0):.3f}")
    print(f"Pass@10: {results['overall']['pass_at_k'].get(10, 0.0):.3f}")
    print(f"E-CodeBLEU: {results['overall']['e_codebleu']:.3f}")
    print(f"SCS: {results['overall']['scs']:.3f}")
    print(f"CSR-EC: {results['overall']['csr_ec']:.3f}")
    print("="*60)
    
    # Generate visualizations if requested
    if args.visualize:
        evaluator.visualize_results()


if __name__ == "__main__":
    main()