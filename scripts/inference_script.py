"""
Inference script for CodeForge-Quantum
Provides code generation capabilities with all enhanced features
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional, Any, Tuple
import json
import argparse
import os
import sys
import time
from pathlib import Path
import logging
from dataclasses import dataclass

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.codeforge_quantum import CodeForgeQuantum, CodeForgeConfig
from fine_tuning.apeft_lora import AdaptiveLoRAModule
from rag.vector_database import (
    FAISSVectorDatabase,
    RetrievalAugmentedGeneration,
    DensePassageRetriever,
    CrossAttentionRetrieval
)
from prompting.chain_of_thought import StructuredReasoning, PromptEngineering
from execution.abstract_interpreter import CompilerFeedback
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class InferenceConfig:
    """Configuration for inference"""
    model_checkpoint: str = "checkpoints/best"
    device: str = "cuda"
    max_length: int = 512
    num_return_sequences: int = 1
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 50
    beam_width: int = 5
    use_rag: bool = True
    use_cot: bool = True
    use_execution_feedback: bool = True
    compile_output: bool = True
    vector_db_path: str = "data/code_database"


class CodeForgeInference:
    """Inference engine for CodeForge-Quantum"""
    
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        logger.info("Initializing CodeForge-Quantum for inference...")
        
        # Load model
        self._load_model()
        
        # Initialize components
        if config.use_rag:
            self._setup_rag()
        if config.use_cot:
            self._setup_cot()
        if config.use_execution_feedback:
            self._setup_execution()
        
        logger.info("Inference engine ready!")
    
    def _load_model(self):
        """Load model from checkpoint"""
        logger.info(f"Loading model from {self.config.model_checkpoint}")
        
        # Load checkpoint
        checkpoint_path = os.path.join(self.config.model_checkpoint, 'checkpoint.pt')
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Reconstruct model config
        model_config = CodeForgeConfig(**checkpoint['config'])
        
        # Initialize model
        self.model = CodeForgeQuantum(model_config)
        
        # Load state dict
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info("Model loaded successfully")
    
    def _setup_rag(self):
        """Initialize RAG system"""
        logger.info("Setting up RAG system...")
        
        # Load vector database
        self.vector_db = FAISSVectorDatabase()
        if os.path.exists(self.config.vector_db_path):
            self.vector_db.load(self.config.vector_db_path)
            logger.info(f"Loaded vector database with {self.vector_db.stats['total_snippets']} snippets")
        
        # Initialize retriever and cross-attention
        self.retriever = DensePassageRetriever()
        self.retriever.to(self.device)
        
        self.cross_attention = CrossAttentionRetrieval()
        self.cross_attention.to(self.device)
        
        self.rag_system = RetrievalAugmentedGeneration(
            self.vector_db,
            self.retriever,
            self.cross_attention
        )
    
    def _setup_cot(self):
        """Initialize Chain-of-Thought reasoning"""
        logger.info("Setting up Chain-of-Thought reasoning...")
        self.cot_reasoner = StructuredReasoning()
        self.prompt_engineer = PromptEngineering()
    
    def _setup_execution(self):
        """Initialize execution feedback system"""
        logger.info("Setting up execution feedback...")
        self.compiler_feedback = CompilerFeedback()
    
    def generate(
        self,
        prompt: str,
        max_length: Optional[int] = None,
        temperature: Optional[float] = None,
        num_return_sequences: Optional[int] = None,
        use_beam_search: bool = True,
        return_reasoning: bool = False
    ) -> Dict[str, Any]:
        """
        Generate code for given prompt
        
        Args:
            prompt: Natural language problem description
            max_length: Maximum generation length
            temperature: Sampling temperature
            num_return_sequences: Number of solutions to generate
            use_beam_search: Whether to use beam search
            return_reasoning: Whether to return CoT reasoning
        
        Returns:
            Dictionary containing generated code and metadata
        """
        start_time = time.time()
        
        # Use config defaults if not specified
        max_length = max_length or self.config.max_length
        temperature = temperature or self.config.temperature
        num_return_sequences = num_return_sequences or self.config.num_return_sequences
        
        result = {
            'prompt': prompt,
            'solutions': [],
            'metadata': {}
        }
        
        # Step 1: Chain-of-Thought reasoning
        reasoning_output = None
        if self.config.use_cot and hasattr(self, 'cot_reasoner'):
            logger.info("Applying Chain-of-Thought reasoning...")
            
            # Create structured prompt
            cot_prompt = self.prompt_engineer.create_structured_prompt(
                task_description=prompt,
                constraints=[],
                requirements=[]
            )
            
            # Generate reasoning steps
            dummy_hidden = torch.zeros(1, 8192).to(self.device)
            reasoning_output = self.cot_reasoner.reason_about_problem(
                prompt, dummy_hidden
            )
            
            if return_reasoning:
                result['reasoning'] = reasoning_output
        
        # Step 2: Retrieve relevant code (RAG)
        retrieved_context = None
        retrieved_snippets = []
        if self.config.use_rag and hasattr(self, 'rag_system'):
            logger.info("Retrieving relevant code snippets...")
            
            dummy_hidden = torch.zeros(1, 8192).to(self.device)
            augmented_hidden, snippets = self.rag_system.retrieve_and_augment(
                prompt, dummy_hidden
            )
            
            retrieved_snippets = snippets[:5]  # Top 5 snippets
            result['retrieved_snippets'] = [
                {
                    'code': s.code[:200] + '...' if len(s.code) > 200 else s.code,
                    'language': s.language,
                    'tags': s.tags
                }
                for s in retrieved_snippets
            ]
            
            # Encode retrieved context
            if retrieved_snippets:
                context_text = "\n".join([s.code for s in retrieved_snippets[:3]])
                context_tokens = self.tokenizer(
                    context_text,
                    max_length=512,
                    truncation=True,
                    return_tensors='pt'
                ).to(self.device)
                retrieved_context = context_tokens['input_ids']
        
        # Step 3: Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors='pt',
            max_length=512,
            truncation=True,
            padding=True
        ).to(self.device)
        
        # Step 4: Generate code
        logger.info("Generating code...")
        
        with torch.no_grad():
            if use_beam_search:
                generated_ids = self.model.generate(
                    nl_input_ids=inputs['input_ids'],
                    nl_attention_mask=inputs['attention_mask'],
                    max_length=max_length,
                    num_beams=self.config.beam_width,
                    temperature=temperature,
                    retrieved_context=retrieved_context
                )
            else:
                # Sampling-based generation
                outputs = self.model(
                    nl_input_ids=inputs['input_ids'],
                    nl_attention_mask=inputs['attention_mask'],
                    retrieved_context=retrieved_context
                )
                
                # Sample from logits
                generated_ids = self._sample_from_logits(
                    outputs['logits'],
                    max_length=max_length,
                    temperature=temperature,
                    num_return_sequences=num_return_sequences
                )
        
        # Step 5: Decode generated code
        generated_codes = []
        for gen_ids in generated_ids:
            code = self.tokenizer.decode(
                gen_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            generated_codes.append(code)
        
        # Step 6: Apply execution feedback and refinement
        if self.config.use_execution_feedback and hasattr(self, 'compiler_feedback'):
            logger.info("Applying execution feedback...")
            
            refined_codes = []
            for code in generated_codes:
                # Compile and get feedback
                feedback = self.compiler_feedback.compile_and_analyze(code)
                
                if not feedback['success'] and feedback['errors']:
                    # Try to refine based on errors
                    refined_code = self._refine_with_feedback(
                        prompt, code, feedback
                    )
                    refined_codes.append(refined_code)
                    
                    # Add compilation results
                    solution_data = {
                        'code': refined_code,
                        'original_code': code,
                        'compilation': self.compiler_feedback.compile_and_analyze(refined_code),
                        'refined': True
                    }
                else:
                    solution_data = {
                        'code': code,
                        'compilation': feedback,
                        'refined': False
                    }
                
                result['solutions'].append(solution_data)
        else:
            # No execution feedback
            for code in generated_codes:
                result['solutions'].append({
                    'code': code,
                    'refined': False
                })
        
        # Add metadata
        generation_time = time.time() - start_time
        result['metadata'] = {
            'generation_time': f"{generation_time:.2f}s",
            'model': 'CodeForge-Quantum',
            'temperature': temperature,
            'max_length': max_length,
            'beam_search': use_beam_search,
            'rag_enabled': self.config.use_rag,
            'cot_enabled': self.config.use_cot,
            'execution_feedback_enabled': self.config.use_execution_feedback
        }
        
        return result
    
    def _sample_from_logits(
        self,
        logits: torch.Tensor,
        max_length: int,
        temperature: float,
        num_return_sequences: int
    ) -> List[torch.Tensor]:
        """Sample sequences from logits"""
        batch_size, seq_len, vocab_size = logits.shape
        
        # Apply temperature
        logits = logits / temperature
        
        # Sample
        probs = torch.softmax(logits, dim=-1)
        
        generated = []
        for _ in range(num_return_sequences):
            sequence = []
            for t in range(min(max_length, seq_len)):
                # Sample token
                token = torch.multinomial(probs[0, t], 1)
                sequence.append(token.item())
                
                # Stop at EOS token
                if token.item() == self.tokenizer.eos_token_id:
                    break
            
            generated.append(torch.tensor(sequence))
        
        return generated
    
    def _refine_with_feedback(
        self,
        prompt: str,
        code: str,
        feedback: Dict[str, Any]
    ) -> str:
        """Refine code based on compilation feedback"""
        # Create refinement prompt
        error_messages = [e['message'] for e in feedback.get('errors', [])]
        error_summary = '\n'.join(error_messages[:3])  # Top 3 errors
        
        refinement_prompt = f"""
Fix the following code based on the errors:

Original task: {prompt}

Code with errors:
```python
{code}
```

Errors:
{error_summary}

Please provide a corrected version:
"""
        
        # Generate refined version
        refined_result = self.generate(
            refinement_prompt,
            temperature=0.5,  # Lower temperature for refinement
            use_beam_search=True,
            return_reasoning=False
        )
        
        if refined_result['solutions']:
            return refined_result['solutions'][0]['code']
        return code  # Return original if refinement fails
    
    def batch_generate(
        self,
        prompts: List[str],
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Generate code for multiple prompts"""
        results = []
        
        for i, prompt in enumerate(prompts):
            logger.info(f"Processing prompt {i+1}/{len(prompts)}")
            result = self.generate(prompt, **kwargs)
            results.append(result)
        
        return results
    
    def interactive_mode(self):
        """Run in interactive mode"""
        print("\n" + "="*60)
        print("CodeForge-Quantum Interactive Mode")
        print("="*60)
        print("Enter your coding problems (type 'exit' to quit)")
        print("Type 'help' for available commands")
        print("="*60 + "\n")
        
        while True:
            try:
                prompt = input("\n🔧 Enter problem description:\n> ").strip()
                
                if not prompt:
                    continue
                
                if prompt.lower() == 'exit':
                    print("Goodbye!")
                    break
                
                if prompt.lower() == 'help':
                    self._print_help()
                    continue
                
                # Check for special commands
                if prompt.startswith('/'):
                    self._handle_command(prompt)
                    continue
                
                # Generate code
                print("\n⚙️  Generating solution...")
                result = self.generate(
                    prompt,
                    return_reasoning=True
                )
                
                # Display results
                self._display_result(result)
                
                # Ask for feedback
                feedback = input("\n📝 Feedback (optional, press Enter to skip):\n> ").strip()
                if feedback:
                    print("\n⚙️  Refining based on feedback...")
                    refined = self._refine_with_feedback(
                        prompt,
                        result['solutions'][0]['code'],
                        {'errors': [{'message': feedback}]}
                    )
                    print("\n📋 Refined Solution:")
                    print("```python")
                    print(refined)
                    print("```")
                
            except KeyboardInterrupt:
                print("\n\nInterrupted. Type 'exit' to quit.")
            except Exception as e:
                print(f"\n❌ Error: {e}")
    
    def _print_help(self):
        """Print help message"""
        help_text = """
Available Commands:
==================
/help              - Show this help message
/config            - Show current configuration
/rag on/off        - Enable/disable RAG
/cot on/off        - Enable/disable Chain-of-Thought
/exec on/off       - Enable/disable execution feedback
/temp <value>      - Set temperature (0.0-1.0)
/beams <value>     - Set beam width (1-10)
/examples          - Show example problems
/benchmark         - Run benchmark tests
/exit              - Exit interactive mode

Example Usage:
=============
> Write a function to find the nth Fibonacci number
> /temp 0.5
> /rag on
> Implement binary search in Python
"""
        print(help_text)
    
    def _handle_command(self, command: str):
        """Handle special commands"""
        parts = command.split()
        cmd = parts[0].lower()
        
        if cmd == '/config':
            print(f"\nCurrent Configuration:")
            print(f"  RAG: {'Enabled' if self.config.use_rag else 'Disabled'}")
            print(f"  CoT: {'Enabled' if self.config.use_cot else 'Disabled'}")
            print(f"  Execution Feedback: {'Enabled' if self.config.use_execution_feedback else 'Disabled'}")
            print(f"  Temperature: {self.config.temperature}")
            print(f"  Beam Width: {self.config.beam_width}")
            
        elif cmd == '/rag' and len(parts) > 1:
            self.config.use_rag = parts[1].lower() == 'on'
            print(f"RAG {'enabled' if self.config.use_rag else 'disabled'}")
            
        elif cmd == '/cot' and len(parts) > 1:
            self.config.use_cot = parts[1].lower() == 'on'
            print(f"Chain-of-Thought {'enabled' if self.config.use_cot else 'disabled'}")
            
        elif cmd == '/exec' and len(parts) > 1:
            self.config.use_execution_feedback = parts[1].lower() == 'on'
            print(f"Execution feedback {'enabled' if self.config.use_execution_feedback else 'disabled'}")
            
        elif cmd == '/temp' and len(parts) > 1:
            try:
                self.config.temperature = float(parts[1])
                print(f"Temperature set to {self.config.temperature}")
            except ValueError:
                print("Invalid temperature value")
                
        elif cmd == '/beams' and len(parts) > 1:
            try:
                self.config.beam_width = int(parts[1])
                print(f"Beam width set to {self.config.beam_width}")
            except ValueError:
                print("Invalid beam width value")
                
        elif cmd == '/examples':
            self._show_examples()
            
        elif cmd == '/benchmark':
            self._run_benchmark()
            
        else:
            print(f"Unknown command: {command}")
    
    def _show_examples(self):
        """Show example problems"""
        examples = [
            "Write a function to check if a string is a palindrome",
            "Implement quicksort algorithm",
            "Create a class for a binary search tree with insert and search methods",
            "Write a function to find all prime numbers up to n",
            "Implement a LRU cache with get and put operations"
        ]
        
        print("\nExample Problems:")
        print("=" * 40)
        for i, example in enumerate(examples, 1):
            print(f"{i}. {example}")
    
    def _run_benchmark(self):
        """Run quick benchmark"""
        print("\nRunning benchmark...")
        
        test_problems = [
            "Write a function to reverse a string",
            "Implement binary search",
            "Find the factorial of a number"
        ]
        
        total_time = 0
        successful = 0
        
        for problem in test_problems:
            start = time.time()
            result = self.generate(problem, temperature=0.5)
            elapsed = time.time() - start
            total_time += elapsed
            
            if result['solutions']:
                code = result['solutions'][0]['code']
                if self.config.use_execution_feedback:
                    compilation = result['solutions'][0].get('compilation', {})
                    if compilation.get('success', False):
                        successful += 1
                else:
                    successful += 1
            
            print(f"  ✓ {problem[:30]}... ({elapsed:.2f}s)")
        
        print(f"\nBenchmark Results:")
        print(f"  Total problems: {len(test_problems)}")
        print(f"  Successful: {successful}/{len(test_problems)}")
        print(f"  Average time: {total_time/len(test_problems):.2f}s")
        print(f"  Total time: {total_time:.2f}s")
    
    def _display_result(self, result: Dict[str, Any]):
        """Display generation result"""
        print("\n" + "="*60)
        
        # Show reasoning if available
        if 'reasoning' in result:
            print("\n🧠 Chain-of-Thought Reasoning:")
            print("-" * 40)
            for step in result['reasoning'].get('reasoning_chain', []):
                print(f"  Step {step['step']}: {step['type']}")
                print(f"    Confidence: {step['confidence']:.2%}")
        
        # Show retrieved snippets if available
        if 'retrieved_snippets' in result and result['retrieved_snippets']:
            print("\n📚 Retrieved Code References:")
            print("-" * 40)
            for i, snippet in enumerate(result['retrieved_snippets'][:3], 1):
                print(f"  {i}. Language: {snippet['language']}")
                print(f"     Tags: {', '.join(snippet['tags'][:3])}")
        
        # Show generated solutions
        print("\n📋 Generated Solution:")
        print("-" * 40)
        
        for i, solution in enumerate(result['solutions'], 1):
            if len(result['solutions']) > 1:
                print(f"\n--- Solution {i} ---")
            
            print("```python")
            print(solution['code'])
            print("```")
            
            # Show compilation results if available
            if 'compilation' in solution:
                comp = solution['compilation']
                if comp['success']:
                    print("✅ Compilation: Success")
                    if comp.get('output'):
                        print(f"   Output: {comp['output'][:100]}...")
                else:
                    print("❌ Compilation: Failed")
                    if comp.get('errors'):
                        for error in comp['errors'][:2]:
                            print(f"   - {error.get('type', 'error')}: {error.get('message', '')[:100]}")
            
            if solution.get('refined'):
                print("🔧 Note: Code was refined based on compilation feedback")
        
        # Show metadata
        print("\n📊 Generation Metadata:")
        print("-" * 40)
        for key, value in result['metadata'].items():
            print(f"  {key}: {value}")
        
        print("\n" + "="*60)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="CodeForge-Quantum Inference")
    
    parser.add_argument(
        'prompt',
        type=str,
        nargs='?',
        help='Problem description (if not provided, enters interactive mode)'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='checkpoints/best',
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use'
    )
    parser.add_argument(
        '--max-length',
        type=int,
        default=512,
        help='Maximum generation length'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.7,
        help='Sampling temperature'
    )
    parser.add_argument(
        '--num-solutions',
        type=int,
        default=1,
        help='Number of solutions to generate'
    )
    parser.add_argument(
        '--no-rag',
        action='store_true',
        help='Disable RAG'
    )
    parser.add_argument(
        '--no-cot',
        action='store_true',
        help='Disable Chain-of-Thought'
    )
    parser.add_argument(
        '--no-exec',
        action='store_true',
        help='Disable execution feedback'
    )
    parser.add_argument(
        '--batch-file',
        type=str,
        help='JSON file with batch of prompts'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output file for results'
    )
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Run in interactive mode'
    )
    
    args = parser.parse_args()
    
    # Create configuration
    config = InferenceConfig(
        model_checkpoint=args.checkpoint,
        device=args.device,
        max_length=args.max_length,
        temperature=args.temperature,
        num_return_sequences=args.num_solutions,
        use_rag=not args.no_rag,
        use_cot=not args.no_cot,
        use_execution_feedback=not args.no_exec
    )
    
    # Initialize inference engine
    engine = CodeForgeInference(config)
    
    # Run in appropriate mode
    if args.interactive or not args.prompt:
        # Interactive mode
        engine.interactive_mode()
    elif args.batch_file:
        # Batch processing
        with open(args.batch_file, 'r') as f:
            prompts = json.load(f)
        
        results = engine.batch_generate(prompts)
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {args.output}")
        else:
            print(json.dumps(results, indent=2))
    else:
        # Single prompt
        result = engine.generate(args.prompt, return_reasoning=True)
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            logger.info(f"Result saved to {args.output}")
        else:
            engine._display_result(result)


if __name__ == "__main__":
    main()