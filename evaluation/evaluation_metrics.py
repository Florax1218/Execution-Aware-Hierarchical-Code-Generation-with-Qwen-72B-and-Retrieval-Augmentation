"""
Evaluation Metrics for CodeForge-Quantum
Includes Pass@k, Enhanced CodeBLEU, Semantic Consistency Score, and CSR-EC
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Tuple, Any, Callable
import ast
import subprocess
import tempfile
import os
from collections import defaultdict
import multiprocessing as mp
from functools import partial
import re
import math
from dataclasses import dataclass
from enum import Enum
from sacrebleu import sentence_bleu
from rouge_score import rouge_scorer
import edit_distance


class ErrorCategory(Enum):
    """Categories of compilation/runtime errors"""
    SYNTAX = "syntax"
    NAME = "name"
    TYPE = "type"
    INDEX = "index"
    KEY = "key"
    VALUE = "value"
    IMPORT = "import"
    ATTRIBUTE = "attribute"
    RUNTIME = "runtime"
    TIMEOUT = "timeout"
    MEMORY = "memory"


@dataclass
class TestResult:
    """Result of a single test case execution"""
    passed: bool
    output: Any
    expected: Any
    error: Optional[str] = None
    execution_time: float = 0.0
    memory_usage: float = 0.0


class PassAtK:
    """Pass@k metric for functional correctness"""
    
    @staticmethod
    def compute(
        problems: List[Dict],
        generated_solutions: List[List[str]],
        k_values: List[int] = [1, 5, 10],
        timeout: int = 5,
        num_workers: int = 4
    ) -> Dict[int, float]:
        """
        Compute Pass@k for multiple k values
        
        Args:
            problems: List of problem dictionaries with test cases
            generated_solutions: List of lists of generated solutions per problem
            k_values: Values of k to compute
            timeout: Timeout for each execution
            num_workers: Number of parallel workers
        
        Returns:
            Dictionary mapping k to Pass@k score
        """
        results = {k: [] for k in k_values}
        
        # Process each problem
        for problem, solutions in zip(problems, generated_solutions):
            # Evaluate all solutions for this problem
            test_results = PassAtK._evaluate_solutions(
                problem,
                solutions,
                timeout,
                num_workers
            )
            
            # Compute Pass@k for each k value
            for k in k_values:
                if k <= len(test_results):
                    # Exact Pass@k computation
                    num_correct = sum(test_results[:k])
                    pass_at_k = 1.0 if num_correct > 0 else 0.0
                else:
                    # Estimate Pass@k using the formula
                    n = len(test_results)
                    c = sum(test_results)
                    if n == 0:
                        pass_at_k = 0.0
                    else:
                        pass_at_k = PassAtK._estimate_pass_at_k(n, c, k)
                
                results[k].append(pass_at_k)
        
        # Average across all problems
        avg_results = {
            k: np.mean(scores) if scores else 0.0
            for k, scores in results.items()
        }
        
        return avg_results
    
    @staticmethod
    def _estimate_pass_at_k(n: int, c: int, k: int) -> float:
        """
        Estimate Pass@k using the formula from the paper
        
        Args:
            n: Total number of generated solutions
            c: Number of correct solutions
            k: k value
        
        Returns:
            Estimated Pass@k
        """
        if n - c < k:
            return 1.0
        
        # Compute using combination formula
        def comb(n, k):
            if k > n or k < 0:
                return 0
            if k == 0 or k == n:
                return 1
            return math.factorial(n) // (math.factorial(k) * math.factorial(n - k))
        
        pass_at_k = 1 - comb(n - c, k) / comb(n, k)
        return pass_at_k
    
    @staticmethod
    def _evaluate_solutions(
        problem: Dict,
        solutions: List[str],
        timeout: int,
        num_workers: int
    ) -> List[bool]:
        """Evaluate multiple solutions for a problem"""
        test_cases = problem.get('test_cases', [])
        
        # Use multiprocessing for parallel evaluation
        with mp.Pool(num_workers) as pool:
            eval_func = partial(
                PassAtK._evaluate_single_solution,
                test_cases=test_cases,
                timeout=timeout
            )
            results = pool.map(eval_func, solutions)
        
        return results
    
    @staticmethod
    def _evaluate_single_solution(
        solution: str,
        test_cases: List[Dict],
        timeout: int
    ) -> bool:
        """Evaluate a single solution against test cases"""
        # Write solution to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(solution)
            temp_file = f.name
        
        try:
            # Test each case
            for test_case in test_cases:
                input_data = test_case.get('input', '')
                expected_output = test_case.get('output', '')
                
                # Run the solution
                process = subprocess.run(
                    ['python', temp_file],
                    input=input_data,
                    capture_output=True,
                    text=True,
                    timeout=timeout
                )
                
                # Check output
                actual_output = process.stdout.strip()
                if actual_output != expected_output.strip():
                    return False
            
            return True
            
        except (subprocess.TimeoutExpired, Exception):
            return False
        finally:
            os.unlink(temp_file)


class EnhancedCodeBLEU:
    """Enhanced CodeBLEU with execution traces"""
    
    def __init__(
        self,
        alpha_ngram: float = 0.25,
        alpha_weighted: float = 0.25,
        alpha_syntax: float = 0.25,
        alpha_semantic: float = 0.15,
        alpha_execution: float = 0.10
    ):
        self.alpha_ngram = alpha_ngram
        self.alpha_weighted = alpha_weighted
        self.alpha_syntax = alpha_syntax
        self.alpha_semantic = alpha_semantic
        self.alpha_execution = alpha_execution
        
        # Initialize ROUGE scorer for n-gram matching
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'],
            use_stemmer=True
        )
    
    def compute(
        self,
        generated: str,
        reference: str,
        execution_trace_gen: Optional[List] = None,
        execution_trace_ref: Optional[List] = None
    ) -> float:
        """
        Compute Enhanced CodeBLEU score
        
        Args:
            generated: Generated code
            reference: Reference code
            execution_trace_gen: Execution trace of generated code
            execution_trace_ref: Execution trace of reference code
        
        Returns:
            E-CodeBLEU score
        """
        components = {}
        
        # N-gram match score
        components['ngram'] = self._compute_ngram_score(generated, reference)
        
        # Weighted n-gram match score
        components['weighted'] = self._compute_weighted_ngram_score(
            generated, reference
        )
        
        # Syntax match score
        components['syntax'] = self._compute_syntax_score(generated, reference)
        
        # Semantic/dataflow match score
        components['semantic'] = self._compute_semantic_score(generated, reference)
        
        # Execution trace match score
        if execution_trace_gen and execution_trace_ref:
            components['execution'] = self._compute_execution_score(
                execution_trace_gen, execution_trace_ref
            )
        else:
            components['execution'] = 0.0
        
        # Combine scores
        e_codebleu = (
            self.alpha_ngram * components['ngram'] +
            self.alpha_weighted * components['weighted'] +
            self.alpha_syntax * components['syntax'] +
            self.alpha_semantic * components['semantic'] +
            self.alpha_execution * components['execution']
        )
        
        return e_codebleu
    
    def _compute_ngram_score(self, generated: str, reference: str) -> float:
        """Compute n-gram matching score"""
        # Use ROUGE for n-gram matching
        scores = self.rouge_scorer.score(reference, generated)
        
        # Average ROUGE scores
        avg_score = (
            scores['rouge1'].fmeasure +
            scores['rouge2'].fmeasure +
            scores['rougeL'].fmeasure
        ) / 3
        
        return avg_score
    
    def _compute_weighted_ngram_score(
        self,
        generated: str,
        reference: str
    ) -> float:
        """Compute weighted n-gram score based on keyword importance"""
        # Define keyword weights
        keyword_weights = {
            'def': 2.0, 'class': 2.0, 'return': 1.5,
            'if': 1.5, 'for': 1.5, 'while': 1.5,
            'import': 1.3, 'try': 1.3, 'except': 1.3
        }
        
        gen_tokens = generated.split()
        ref_tokens = reference.split()
        
        weighted_matches = 0
        total_weight = 0
        
        for token in ref_tokens:
            weight = keyword_weights.get(token, 1.0)
            total_weight += weight
            if token in gen_tokens:
                weighted_matches += weight
        
        return weighted_matches / total_weight if total_weight > 0 else 0.0
    
    def _compute_syntax_score(self, generated: str, reference: str) -> float:
        """Compute syntax similarity using AST"""
        try:
            gen_ast = ast.parse(generated)
            ref_ast = ast.parse(reference)
            
            # Extract AST node types
            gen_nodes = self._extract_ast_nodes(gen_ast)
            ref_nodes = self._extract_ast_nodes(ref_ast)
            
            # Compute Jaccard similarity
            intersection = len(gen_nodes & ref_nodes)
            union = len(gen_nodes | ref_nodes)
            
            return intersection / union if union > 0 else 0.0
            
        except SyntaxError:
            return 0.0
    
    def _extract_ast_nodes(self, tree: ast.AST) -> set:
        """Extract AST node types"""
        nodes = set()
        
        def visit(node):
            nodes.add(type(node).__name__)
            for child in ast.iter_child_nodes(node):
                visit(child)
        
        visit(tree)
        return nodes
    
    def _compute_semantic_score(self, generated: str, reference: str) -> float:
        """Compute semantic similarity using dataflow analysis"""
        try:
            gen_ast = ast.parse(generated)
            ref_ast = ast.parse(reference)
            
            # Extract variable usage patterns
            gen_vars = self._extract_variable_usage(gen_ast)
            ref_vars = self._extract_variable_usage(ref_ast)
            
            # Compare variable usage patterns
            score = 0.0
            for var in ref_vars:
                if var in gen_vars:
                    # Compare usage patterns
                    gen_usage = gen_vars[var]
                    ref_usage = ref_vars[var]
                    
                    # Simple similarity based on usage counts
                    similarity = min(gen_usage, ref_usage) / max(gen_usage, ref_usage)
                    score += similarity
            
            return score / len(ref_vars) if ref_vars else 0.0
            
        except SyntaxError:
            return 0.0
    
    def _extract_variable_usage(self, tree: ast.AST) -> Dict[str, int]:
        """Extract variable usage patterns"""
        usage = defaultdict(int)
        
        class VariableVisitor(ast.NodeVisitor):
            def __init__(self, usage_dict):
                self.usage = usage_dict
                
            def visit_Name(self, node):
                self.usage[node.id] += 1
                self.generic_visit(node)
        
        visitor = VariableVisitor(usage)
        visitor.visit(tree)
        
        return dict(usage)
    
    def _compute_execution_score(
        self,
        trace_gen: List,
        trace_ref: List
    ) -> float:
        """Compute execution trace similarity"""
        # Compare execution traces
        if not trace_gen or not trace_ref:
            return 0.0
        
        # Compare outputs at each step
        matches = 0
        total = min(len(trace_gen), len(trace_ref))
        
        for i in range(total):
            if trace_gen[i] == trace_ref[i]:
                matches += 1
        
        return matches / total if total > 0 else 0.0


class SemanticConsistencyScore:
    """Semantic Consistency Score for behavioral equivalence"""
    
    def __init__(
        self,
        num_test_cases: int = 100,
        execution_penalty: float = 0.01
    ):
        self.num_test_cases = num_test_cases
        self.execution_penalty = execution_penalty
    
    def compute(
        self,
        generated_func: Callable,
        reference_func: Callable,
        test_inputs: List[Any],
        timeout: int = 5
    ) -> float:
        """
        Compute Semantic Consistency Score
        
        Args:
            generated_func: Generated function
            reference_func: Reference function
            test_inputs: List of test inputs
            timeout: Execution timeout
        
        Returns:
            SCS score
        """
        consistent_count = 0
        total_time = 0.0
        
        for i, test_input in enumerate(test_inputs[:self.num_test_cases]):
            try:
                # Execute both functions
                import time
                
                start_time = time.time()
                gen_output = generated_func(*test_input)
                gen_time = time.time() - start_time
                
                ref_output = reference_func(*test_input)
                
                # Check equivalence
                if self._outputs_equivalent(gen_output, ref_output):
                    consistent_count += 1
                
                total_time += gen_time
                
            except Exception:
                # If execution fails, no consistency
                pass
        
        # Compute score with execution time penalty
        consistency_rate = consistent_count / self.num_test_cases
        time_penalty = np.exp(-self.execution_penalty * total_time)
        
        scs = consistency_rate * time_penalty
        
        return scs
    
    def _outputs_equivalent(self, output1: Any, output2: Any) -> bool:
        """Check if two outputs are equivalent"""
        if type(output1) != type(output2):
            return False
        
        if isinstance(output1, (int, float)):
            # Numerical comparison with tolerance
            return abs(output1 - output2) < 1e-6
        elif isinstance(output1, str):
            return output1 == output2
        elif isinstance(output1, (list, tuple)):
            if len(output1) != len(output2):
                return False
            return all(
                self._outputs_equivalent(a, b)
                for a, b in zip(output1, output2)
            )
        elif isinstance(output1, dict):
            if set(output1.keys()) != set(output2.keys()):
                return False
            return all(
                self._outputs_equivalent(output1[k], output2[k])
                for k in output1.keys()
            )
        else:
            return output1 == output2


class CompilationSuccessRate:
    """CSR-EC: Compilation Success Rate with Error Categorization"""
    
    def __init__(self):
        self.error_weights = {
            ErrorCategory.SYNTAX: 0.3,
            ErrorCategory.NAME: 0.15,
            ErrorCategory.TYPE: 0.15,
            ErrorCategory.INDEX: 0.1,
            ErrorCategory.KEY: 0.1,
            ErrorCategory.VALUE: 0.1,
            ErrorCategory.IMPORT: 0.05,
            ErrorCategory.ATTRIBUTE: 0.05,
            ErrorCategory.RUNTIME: 0.2,
            ErrorCategory.TIMEOUT: 0.1,
            ErrorCategory.MEMORY: 0.1
        }
    
    def compute(
        self,
        generated_codes: List[str],
        language: str = 'python'
    ) -> Tuple[float, Dict[ErrorCategory, int]]:
        """
        Compute compilation success rate with error categorization
        
        Args:
            generated_codes: List of generated code strings
            language: Programming language
        
        Returns:
            Tuple of (CSR-EC score, error counts by category)
        """
        total = len(generated_codes)
        if total == 0:
            return 0.0, {}
        
        error_counts = defaultdict(int)
        successful = 0
        
        for code in generated_codes:
            result = self._compile_and_categorize(code, language)
            
            if result['success']:
                successful += 1
            else:
                for error_type in result['errors']:
                    error_counts[error_type] += 1
        
        # Compute weighted CSR-EC
        csr_base = successful / total
        
        # Apply error category weights
        error_penalty = 0.0
        for error_type, count in error_counts.items():
            weight = self.error_weights.get(error_type, 0.1)
            error_penalty += weight * (count / total)
        
        # CSR-EC combines success rate with error categorization
        csr_ec = csr_base * (1 - error_penalty / 2)  # Soften penalty impact
        
        return csr_ec, dict(error_counts)
    
    def _compile_and_categorize(
        self,
        code: str,
        language: str
    ) -> Dict[str, Any]:
        """Compile code and categorize errors"""
        result = {
            'success': False,
            'errors': []
        }
        
        if language == 'python':
            # Check syntax first
            try:
                ast.parse(code)
            except SyntaxError:
                result['errors'].append(ErrorCategory.SYNTAX)
                return result
            
            # Try to execute
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name
            
            try:
                process = subprocess.run(
                    ['python', temp_file],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                
                if process.returncode == 0:
                    result['success'] = True
                else:
                    # Parse error message
                    error_msg = process.stderr
                    result['errors'] = self._parse_python_error(error_msg)
                    
            except subprocess.TimeoutExpired:
                result['errors'].append(ErrorCategory.TIMEOUT)
            except Exception:
                result['errors'].append(ErrorCategory.RUNTIME)
            finally:
                os.unlink(temp_file)
        
        return result
    
    def _parse_python_error(self, error_msg: str) -> List[ErrorCategory]:
        """Parse Python error message and categorize"""
        errors = []
        
        error_patterns = {
            ErrorCategory.NAME: r'NameError:',
            ErrorCategory.TYPE: r'TypeError:',
            ErrorCategory.INDEX: r'IndexError:',
            ErrorCategory.KEY: r'KeyError:',
            ErrorCategory.VALUE: r'ValueError:',
            ErrorCategory.IMPORT: r'ImportError:|ModuleNotFoundError:',
            ErrorCategory.ATTRIBUTE: r'AttributeError:'
        }
        
        for error_type, pattern in error_patterns.items():
            if re.search(pattern, error_msg):
                errors.append(error_type)
        
        if not errors:
            errors.append(ErrorCategory.RUNTIME)
        
        return errors


class MetricsAggregator:
    """Aggregate all metrics for comprehensive evaluation"""
    
    def __init__(self):
        self.pass_at_k = PassAtK()
        self.e_codebleu = EnhancedCodeBLEU()
        self.scs = SemanticConsistencyScore()
        self.csr = CompilationSuccessRate()
    
    def evaluate(
        self,
        problems: List[Dict],
        generated_solutions: List[List[str]],
        reference_solutions: List[str],
        k_values: List[int] = [1, 5, 10]
    ) -> Dict[str, Any]:
        """
        Evaluate using all metrics
        
        Returns:
            Dictionary with all metric results
        """
        results = {}
        
        # Pass@k
        results['pass_at_k'] = self.pass_at_k.compute(
            problems, generated_solutions, k_values
        )
        
        # Enhanced CodeBLEU (average across problems)
        e_cb_scores = []
        for gen_sols, ref_sol in zip(generated_solutions, reference_solutions):
            if gen_sols:
                score = self.e_codebleu.compute(gen_sols[0], ref_sol)
                e_cb_scores.append(score)
        results['e_codebleu'] = np.mean(e_cb_scores) if e_cb_scores else 0.0
        
        # Compilation Success Rate
        all_generated = [sol for sols in generated_solutions for sol in sols]
        csr_score, error_counts = self.csr.compute(all_generated)
        results['csr_ec'] = csr_score
        results['error_distribution'] = error_counts
        
        # Summary statistics
        results['summary'] = {
            'pass@1': results['pass_at_k'].get(1, 0.0),
            'e_codebleu': results['e_codebleu'],
            'csr': csr_score,
            'total_problems': len(problems),
            'total_solutions': len(all_generated)
        }
        
        return results