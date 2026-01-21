"""
Baseline implementations for CodeLlama, StarCoder2, DeepSeek, and Qwen models
"""

import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    CodeLlamaTokenizer,
    GenerationConfig
)
from typing import Dict, List, Optional, Any
import numpy as np


class CodeLlamaBaseline:
    """CodeLlama-70B baseline model"""
    
    def __init__(
        self,
        model_name: str = "codellama/CodeLlama-70b-Python-hf",
        device: str = "auto"
    ):
        self.model_name = model_name
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model and tokenizer
        print(f"Loading CodeLlama model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if device == "auto" else None
        )
        
        if device != "auto":
            self.model.to(self.device)
        
        # Generation config
        self.generation_config = GenerationConfig(
            temperature=0.7,
            top_p=0.95,
            top_k=50,
            max_new_tokens=512,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        
        # Performance metrics from paper
        self.metrics = {
            'pass_at_1': 0.721,
            'e_codebleu': 0.498,
            'scs': 0.695,
            'csr': 0.897,
            'by_difficulty': {
                'easy': 0.867,
                'medium': 0.726,
                'hard': 0.554
            }
        }
    
    def generate(
        self,
        prompt: str,
        max_length: int = 512,
        temperature: float = 0.7,
        num_return_sequences: int = 1
    ) -> List[str]:
        """Generate code using CodeLlama"""
        
        # Format prompt for CodeLlama
        formatted_prompt = self._format_prompt(prompt)
        
        # Tokenize
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.device)
        
        # Update generation config
        self.generation_config.temperature = temperature
        self.generation_config.max_new_tokens = max_length
        self.generation_config.num_return_sequences = num_return_sequences
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                generation_config=self.generation_config
            )
        
        # Decode
        solutions = []
        for output in outputs:
            solution = self.tokenizer.decode(
                output[inputs.input_ids.shape[1]:],  # Remove prompt
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            solutions.append(self._extract_code(solution))
        
        return solutions
    
    def _format_prompt(self, prompt: str) -> str:
        """Format prompt for CodeLlama"""
        return f"""# Task: {prompt}
# Please provide a complete Python solution:

def solution():
    \"\"\"Solution to the problem\"\"\"
"""
    
    def _extract_code(self, text: str) -> str:
        """Extract code from generated text"""
        # Look for code blocks
        if "```python" in text:
            start = text.find("```python") + 9
            end = text.find("```", start)
            if end > start:
                return text[start:end].strip()
        elif "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            if end > start:
                return text[start:end].strip()
        return text.strip()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get baseline metrics"""
        return self.metrics


class StarCoder2Baseline:
    """StarCoder2 baseline model"""
    
    def __init__(
        self,
        model_name: str = "bigcode/starcoder2-15b",
        device: str = "auto"
    ):
        self.model_name = model_name
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model and tokenizer
        print(f"Loading StarCoder2 model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if device == "auto" else None
        )
        
        if device != "auto":
            self.model.to(self.device)
        
        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Performance metrics from paper
        self.metrics = {
            'pass_at_1': 0.739,
            'e_codebleu': 0.512,
            'scs': 0.704,
            'csr': 0.908,
            'by_difficulty': {
                'easy': 0.872,
                'medium': 0.741,
                'hard': 0.573
            }
        }
    
    def generate(
        self,
        prompt: str,
        max_length: int = 512,
        temperature: float = 0.7,
        num_return_sequences: int = 1
    ) -> List[str]:
        """Generate code using StarCoder2"""
        
        # Format prompt
        formatted_prompt = f"<fim_prefix>{prompt}\n<fim_suffix><fim_middle>"
        
        # Tokenize
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=temperature,
                do_sample=True,
                top_p=0.95,
                num_return_sequences=num_return_sequences,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        solutions = []
        for output in outputs:
            solution = self.tokenizer.decode(
                output[inputs.input_ids.shape[1]:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            solutions.append(solution.strip())
        
        return solutions
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get baseline metrics"""
        return self.metrics


class DeepSeekBaseline:
    """DeepSeek-33B baseline model"""
    
    def __init__(
        self,
        model_name: str = "deepseek-ai/deepseek-coder-33b-instruct",
        device: str = "auto"
    ):
        self.model_name = model_name
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model and tokenizer
        print(f"Loading DeepSeek model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if device == "auto" else None,
            trust_remote_code=True
        )
        
        if device != "auto":
            self.model.to(self.device)
        
        # Performance metrics from paper
        self.metrics = {
            'pass_at_1': 0.714,
            'e_codebleu': 0.486,
            'scs': 0.687,
            'csr': 0.885,
            'by_difficulty': {
                'easy': 0.859,
                'medium': 0.712,
                'hard': 0.541
            }
        }
    
    def generate(
        self,
        prompt: str,
        max_length: int = 512,
        temperature: float = 0.7,
        num_return_sequences: int = 1
    ) -> List[str]:
        """Generate code using DeepSeek"""
        
        # Format as instruction
        messages = [
            {"role": "system", "content": "You are an expert programmer."},
            {"role": "user", "content": prompt}
        ]
        
        # Apply chat template
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=temperature,
                do_sample=True,
                top_p=0.95,
                num_return_sequences=num_return_sequences,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        solutions = []
        for output in outputs:
            solution = self.tokenizer.decode(
                output[inputs.input_ids.shape[1]:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            solutions.append(solution.strip())
        
        return solutions
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get baseline metrics"""
        return self.metrics


class QwenBaseline:
    """Qwen-72B-Base baseline model"""
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen-72B",
        device: str = "auto"
    ):
        self.model_name = model_name
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model and tokenizer
        print(f"Loading Qwen model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if device == "auto" else None,
            trust_remote_code=True
        )
        
        if device != "auto":
            self.model.to(self.device)
        
        # Performance metrics from paper
        self.metrics = {
            'pass_at_1': 0.753,
            'e_codebleu': 0.531,
            'scs': 0.719,
            'csr': 0.919,
            'by_difficulty': {
                'easy': 0.881,
                'medium': 0.759,
                'hard': 0.592
            }
        }
    
    def generate(
        self,
        prompt: str,
        max_length: int = 512,
        temperature: float = 0.7,
        num_return_sequences: int = 1
    ) -> List[str]:
        """Generate code using Qwen"""
        
        # Format prompt
        formatted_prompt = f"<|im_start|>system\nYou are an expert programmer.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        # Tokenize
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=temperature,
                do_sample=True,
                top_p=0.95,
                num_return_sequences=num_return_sequences,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        solutions = []
        for output in outputs:
            solution = self.tokenizer.decode(
                output[inputs.input_ids.shape[1]:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            solutions.append(solution.strip())
        
        return solutions
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get baseline metrics"""
        return self.metrics


class BaselineComparison:
    """Compare all baseline models"""
    
    def __init__(self):
        # Metrics from paper for all baselines
        self.baseline_metrics = {
            'GPT-4': {
                'pass_at_1': 0.746,
                'e_codebleu': 0.523,
                'scs': 0.712,
                'csr': 0.914
            },
            'GPT-4-Turbo': {
                'pass_at_1': 0.768,
                'e_codebleu': 0.547,
                'scs': 0.728,
                'csr': 0.926
            },
            'CodeLlama-70B': {
                'pass_at_1': 0.721,
                'e_codebleu': 0.498,
                'scs': 0.695,
                'csr': 0.897
            },
            'StarCoder2': {
                'pass_at_1': 0.739,
                'e_codebleu': 0.512,
                'scs': 0.704,
                'csr': 0.908
            },
            'DeepSeek-33B': {
                'pass_at_1': 0.714,
                'e_codebleu': 0.486,
                'scs': 0.687,
                'csr': 0.885
            },
            'Qwen-72B-Base': {
                'pass_at_1': 0.753,
                'e_codebleu': 0.531,
                'scs': 0.719,
                'csr': 0.919
            },
            'CodeForge-Quantum': {
                'pass_at_1': 0.873,
                'e_codebleu': 0.642,
                'scs': 0.852,
                'csr': 0.962
            }
        }
    
    def compare_all(self) -> Dict[str, Any]:
        """Compare all models"""
        comparison = {}
        
        # Get CodeForge metrics
        cf_metrics = self.baseline_metrics['CodeForge-Quantum']
        
        # Compare with each baseline
        for model_name, metrics in self.baseline_metrics.items():
            if model_name != 'CodeForge-Quantum':
                model_comparison = {}
                for metric_name in ['pass_at_1', 'e_codebleu', 'scs', 'csr']:
                    improvement = (
                        (cf_metrics[metric_name] - metrics[metric_name]) / 
                        metrics[metric_name] * 100
                    )
                    model_comparison[metric_name] = {
                        'baseline': metrics[metric_name],
                        'codeforge': cf_metrics[metric_name],
                        'improvement': f"+{improvement:.1f}%"
                    }
                comparison[model_name] = model_comparison
        
        return comparison
    
    def get_ranking(self, metric: str = 'pass_at_1') -> List[Tuple[str, float]]:
        """Get model ranking by metric"""
        ranking = [
            (model, metrics[metric])
            for model, metrics in self.baseline_metrics.items()
        ]
        ranking.sort(key=lambda x: x[1], reverse=True)
        return ranking
    
    def generate_comparison_chart_data(self) -> Dict[str, List]:
        """Generate data for comparison charts"""
        models = list(self.baseline_metrics.keys())
        
        chart_data = {
            'models': models,
            'pass_at_1': [self.baseline_metrics[m]['pass_at_1'] for m in models],
            'e_codebleu': [self.baseline_metrics[m]['e_codebleu'] for m in models],
            'scs': [self.baseline_metrics[m]['scs'] for m in models],
            'csr': [self.baseline_metrics[m]['csr'] for m in models]
        }
        
        return chart_data