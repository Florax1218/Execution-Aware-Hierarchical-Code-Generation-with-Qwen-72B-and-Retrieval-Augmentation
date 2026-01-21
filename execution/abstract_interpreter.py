"""
Execution-Aware Components: Abstract Interpreter, Trace Analyzer, and Compiler Feedback
Integrates execution traces and compiler feedback into the generation process
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import ast
import subprocess
import tempfile
import os
import traceback
from enum import Enum


class ExecutionState(Enum):
    """Execution state types"""
    SUCCESS = "success"
    RUNTIME_ERROR = "runtime_error"
    COMPILE_ERROR = "compile_error"
    TIMEOUT = "timeout"
    MEMORY_EXCEEDED = "memory_exceeded"


@dataclass
class VariableState:
    """State of a variable during execution"""
    name: str
    value: Any
    type: str
    line_number: int
    scope: str
    
    def to_tensor(self, hidden_dim: int) -> torch.Tensor:
        """Convert variable state to tensor representation"""
        # Simplified encoding - in practice would use more sophisticated encoding
        tensor = torch.zeros(hidden_dim)
        # Encode type information
        type_map = {'int': 1, 'float': 2, 'str': 3, 'list': 4, 'dict': 5, 'bool': 6}
        tensor[0] = type_map.get(self.type, 0)
        # Encode line number
        tensor[1] = self.line_number
        # Encode scope depth
        tensor[2] = len(self.scope.split('.'))
        return tensor


@dataclass
class ExecutionTrace:
    """Complete execution trace of a program"""
    states: List[Dict[str, VariableState]]
    control_flow: List[Tuple[int, str]]  # (line_number, flow_type)
    outputs: List[Tuple[int, Any]]  # (line_number, output)
    error: Optional[str] = None
    status: ExecutionState = ExecutionState.SUCCESS
    memory_usage: float = 0.0
    execution_time: float = 0.0


class AbstractInterpreter(nn.Module):
    """Differentiable abstract interpreter for code execution modeling"""
    def __init__(
        self,
        hidden_dim: int = 8192,
        max_variables: int = 100,
        max_depth: int = 10
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_variables = max_variables
        self.max_depth = max_depth
        
        # Variable state encoder
        self.variable_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        
        # Control flow encoder
        self.flow_encoder = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        
        # Memory state encoder
        self.memory_encoder = nn.Sequential(
            nn.Linear(max_variables * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Execution state predictor
        self.state_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, len(ExecutionState))
        )
        
    def interpret_abstract(self, code: str) -> Dict[str, torch.Tensor]:
        """Perform abstract interpretation on code"""
        try:
            # Parse code to AST
            tree = ast.parse(code)
            
            # Extract abstract states
            variables = self._extract_variables(tree)
            control_flow = self._extract_control_flow(tree)
            
            # Encode states
            var_tensors = [self._encode_variable(v) for v in variables[:self.max_variables]]
            if var_tensors:
                var_tensor = torch.stack(var_tensors)
                var_encoded = self.variable_encoder(var_tensor)
            else:
                var_encoded = torch.zeros(self.hidden_dim)
            
            # Encode control flow
            flow_tensor = self._encode_control_flow(control_flow)
            flow_encoded, _ = self.flow_encoder(flow_tensor.unsqueeze(0))
            flow_encoded = flow_encoded.squeeze(0).mean(dim=0)
            
            # Encode memory state
            memory_tensor = self._encode_memory_state(variables)
            memory_encoded = self.memory_encoder(memory_tensor)
            
            # Combine encodings
            combined = torch.cat([
                var_encoded.mean(dim=0) if var_encoded.dim() > 1 else var_encoded,
                flow_encoded,
                memory_encoded
            ])
            
            # Predict execution state
            state_logits = self.state_predictor(combined)
            state_probs = F.softmax(state_logits, dim=-1)
            
            return {
                'variable_encoding': var_encoded,
                'flow_encoding': flow_encoded,
                'memory_encoding': memory_encoded,
                'state_prediction': state_probs,
                'combined_encoding': combined
            }
            
        except Exception as e:
            # Return error encoding
            return {
                'error': str(e),
                'combined_encoding': torch.zeros(self.hidden_dim * 3)
            }
    
    def _extract_variables(self, tree: ast.AST) -> List[VariableState]:
        """Extract variable information from AST"""
        variables = []
        
        class VariableVisitor(ast.NodeVisitor):
            def __init__(self):
                self.vars = []
                self.scope_stack = ['global']
                
            def visit_Assign(self, node):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        var = VariableState(
                            name=target.id,
                            value=None,  # Abstract value
                            type=self._infer_type(node.value),
                            line_number=node.lineno,
                            scope='.'.join(self.scope_stack)
                        )
                        self.vars.append(var)
                self.generic_visit(node)
                
            def visit_FunctionDef(self, node):
                self.scope_stack.append(node.name)
                self.generic_visit(node)
                self.scope_stack.pop()
                
            def _infer_type(self, node):
                if isinstance(node, ast.Num):
                    return 'int' if isinstance(node.n, int) else 'float'
                elif isinstance(node, ast.Str):
                    return 'str'
                elif isinstance(node, ast.List):
                    return 'list'
                elif isinstance(node, ast.Dict):
                    return 'dict'
                return 'unknown'
        
        visitor = VariableVisitor()
        visitor.visit(tree)
        return visitor.vars[:self.max_variables]
    
    def _extract_control_flow(self, tree: ast.AST) -> List[Tuple[int, str]]:
        """Extract control flow information from AST"""
        flow = []
        
        class FlowVisitor(ast.NodeVisitor):
            def __init__(self):
                self.flow = []
                
            def visit_If(self, node):
                self.flow.append((node.lineno, 'if'))
                self.generic_visit(node)
                
            def visit_For(self, node):
                self.flow.append((node.lineno, 'for'))
                self.generic_visit(node)
                
            def visit_While(self, node):
                self.flow.append((node.lineno, 'while'))
                self.generic_visit(node)
                
            def visit_Return(self, node):
                self.flow.append((node.lineno, 'return'))
                self.generic_visit(node)
        
        visitor = FlowVisitor()
        visitor.visit(tree)
        return visitor.flow
    
    def _encode_variable(self, var: VariableState) -> torch.Tensor:
        """Encode a single variable state"""
        return var.to_tensor(self.hidden_dim)
    
    def _encode_control_flow(self, flow: List[Tuple[int, str]]) -> torch.Tensor:
        """Encode control flow sequence"""
        flow_tensor = torch.zeros(len(flow) if flow else 1, self.hidden_dim)
        
        flow_type_map = {'if': 1, 'for': 2, 'while': 3, 'return': 4}
        for i, (line, flow_type) in enumerate(flow):
            flow_tensor[i, 0] = line
            flow_tensor[i, 1] = flow_type_map.get(flow_type, 0)
        
        return flow_tensor
    
    def _encode_memory_state(self, variables: List[VariableState]) -> torch.Tensor:
        """Encode memory state from variables"""
        memory_tensor = torch.zeros(self.max_variables * self.hidden_dim)
        
        for i, var in enumerate(variables[:self.max_variables]):
            start_idx = i * self.hidden_dim
            end_idx = (i + 1) * self.hidden_dim
            memory_tensor[start_idx:end_idx] = self._encode_variable(var)
        
        return memory_tensor


class TraceAnalyzer(nn.Module):
    """Analyze execution traces and extract patterns"""
    def __init__(
        self,
        hidden_dim: int = 8192,
        num_heads: int = 8
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # GRU for trace sequence encoding
        self.trace_gru = nn.GRU(
            hidden_dim,
            hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        
        # Trace projection
        self.trace_projector = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Pattern extraction attention
        self.pattern_attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # Anomaly detector
        self.anomaly_detector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def analyze_trace(self, trace: ExecutionTrace) -> Dict[str, torch.Tensor]:
        """Analyze an execution trace"""
        # Encode trace steps
        trace_tensors = []
        
        # Encode variable states
        for state_dict in trace.states:
            state_tensor = torch.zeros(self.hidden_dim)
            for i, (name, var_state) in enumerate(state_dict.items()):
                if i < 10:  # Limit to first 10 variables
                    state_tensor[i * 10:(i + 1) * 10] = var_state.to_tensor(10)[:10]
            trace_tensors.append(state_tensor)
        
        # Encode control flow
        for line, flow_type in trace.control_flow:
            flow_tensor = torch.zeros(self.hidden_dim)
            flow_tensor[0] = line
            flow_map = {'if': 1, 'for': 2, 'while': 3, 'return': 4}
            flow_tensor[1] = flow_map.get(flow_type, 0)
            trace_tensors.append(flow_tensor)
        
        if not trace_tensors:
            trace_tensors = [torch.zeros(self.hidden_dim)]
        
        # Stack and encode
        trace_sequence = torch.stack(trace_tensors).unsqueeze(0)
        
        # GRU encoding
        gru_out, _ = self.trace_gru(trace_sequence)
        trace_encoded = self.trace_projector(gru_out.squeeze(0))
        
        # Extract patterns
        patterns, pattern_weights = self.pattern_attention(
            trace_encoded.unsqueeze(0),
            trace_encoded.unsqueeze(0),
            trace_encoded.unsqueeze(0)
        )
        
        # Detect anomalies
        anomaly_scores = self.anomaly_detector(trace_encoded)
        
        return {
            'trace_encoding': trace_encoded.mean(dim=0),
            'patterns': patterns.squeeze(0),
            'pattern_weights': pattern_weights,
            'anomaly_scores': anomaly_scores,
            'status': self._encode_status(trace.status)
        }
    
    def _encode_status(self, status: ExecutionState) -> torch.Tensor:
        """Encode execution status"""
        status_tensor = torch.zeros(len(ExecutionState))
        status_map = {s: i for i, s in enumerate(ExecutionState)}
        status_tensor[status_map[status]] = 1.0
        return status_tensor


class CompilerFeedback:
    """Process and integrate compiler feedback"""
    def __init__(self):
        self.error_patterns = {
            'syntax': r'SyntaxError:',
            'name': r'NameError:',
            'type': r'TypeError:',
            'index': r'IndexError:',
            'key': r'KeyError:',
            'value': r'ValueError:',
            'attribute': r'AttributeError:',
            'import': r'ImportError:',
            'indentation': r'IndentationError:'
        }
        
        self.warning_patterns = {
            'unused': r'unused',
            'redefined': r'redefined',
            'unreachable': r'unreachable',
            'deprecated': r'deprecated'
        }
    
    def compile_and_analyze(
        self,
        code: str,
        language: str = 'python',
        timeout: int = 5
    ) -> Dict[str, Any]:
        """Compile code and analyze feedback"""
        if language == 'python':
            return self._compile_python(code, timeout)
        elif language == 'javascript':
            return self._compile_javascript(code, timeout)
        elif language == 'cpp':
            return self._compile_cpp(code, timeout)
        elif language == 'java':
            return self._compile_java(code, timeout)
        else:
            return {'error': f'Unsupported language: {language}'}
    
    def _compile_python(self, code: str, timeout: int) -> Dict[str, Any]:
        """Compile and analyze Python code"""
        result = {
            'success': False,
            'errors': [],
            'warnings': [],
            'output': '',
            'execution_time': 0.0,
            'memory_usage': 0.0
        }
        
        # First, check syntax
        try:
            ast.parse(code)
        except SyntaxError as e:
            result['errors'].append({
                'type': 'syntax',
                'line': e.lineno,
                'message': str(e),
                'suggestion': self._suggest_fix_for_syntax_error(e)
            })
            return result
        
        # Try to execute in controlled environment
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_file = f.name
        
        try:
            import time
            import psutil
            
            start_time = time.time()
            process = subprocess.Popen(
                ['python', temp_file],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Monitor process
            ps_process = psutil.Process(process.pid)
            
            try:
                stdout, stderr = process.communicate(timeout=timeout)
                execution_time = time.time() - start_time
                memory_info = ps_process.memory_info()
                
                result['execution_time'] = execution_time
                result['memory_usage'] = memory_info.rss / 1024 / 1024  # MB
                
                if process.returncode == 0:
                    result['success'] = True
                    result['output'] = stdout
                else:
                    # Parse errors
                    self._parse_python_errors(stderr, result)
                    
            except subprocess.TimeoutExpired:
                process.kill()
                result['errors'].append({
                    'type': 'timeout',
                    'message': f'Execution exceeded {timeout} seconds'
                })
                
        except Exception as e:
            result['errors'].append({
                'type': 'runtime',
                'message': str(e)
            })
        finally:
            os.unlink(temp_file)
        
        return result
    
    def _compile_javascript(self, code: str, timeout: int) -> Dict[str, Any]:
        """Compile and analyze JavaScript code"""
        result = {
            'success': False,
            'errors': [],
            'warnings': [],
            'output': ''
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as f:
            f.write(code)
            temp_file = f.name
        
        try:
            # Use node.js to run JavaScript
            process = subprocess.run(
                ['node', temp_file],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            if process.returncode == 0:
                result['success'] = True
                result['output'] = process.stdout
            else:
                self._parse_js_errors(process.stderr, result)
                
        except subprocess.TimeoutExpired:
            result['errors'].append({
                'type': 'timeout',
                'message': f'Execution exceeded {timeout} seconds'
            })
        except FileNotFoundError:
            result['errors'].append({
                'type': 'environment',
                'message': 'Node.js not found'
            })
        finally:
            os.unlink(temp_file)
        
        return result
    
    def _compile_cpp(self, code: str, timeout: int) -> Dict[str, Any]:
        """Compile and analyze C++ code"""
        result = {
            'success': False,
            'errors': [],
            'warnings': [],
            'output': ''
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.cpp', delete=False) as f:
            f.write(code)
            source_file = f.name
        
        executable = source_file.replace('.cpp', '')
        
        try:
            # Compile
            compile_process = subprocess.run(
                ['g++', '-std=c++17', '-Wall', '-O2', source_file, '-o', executable],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            if compile_process.returncode != 0:
                self._parse_cpp_errors(compile_process.stderr, result)
                return result
            
            # Parse warnings
            if compile_process.stderr:
                self._parse_cpp_warnings(compile_process.stderr, result)
            
            # Execute
            run_process = subprocess.run(
                [executable],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            result['success'] = True
            result['output'] = run_process.stdout
            
        except subprocess.TimeoutExpired:
            result['errors'].append({
                'type': 'timeout',
                'message': f'Execution exceeded {timeout} seconds'
            })
        except FileNotFoundError:
            result['errors'].append({
                'type': 'environment',
                'message': 'g++ compiler not found'
            })
        finally:
            os.unlink(source_file)
            if os.path.exists(executable):
                os.unlink(executable)
        
        return result
    
    def _compile_java(self, code: str, timeout: int) -> Dict[str, Any]:
        """Compile and analyze Java code"""
        result = {
            'success': False,
            'errors': [],
            'warnings': [],
            'output': ''
        }
        
        # Extract class name
        import re
        class_match = re.search(r'public\s+class\s+(\w+)', code)
        if not class_match:
            result['errors'].append({
                'type': 'syntax',
                'message': 'No public class found'
            })
            return result
        
        class_name = class_match.group(1)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix=f'{class_name}.java', delete=False) as f:
            f.write(code)
            source_file = f.name
        
        try:
            # Compile
            compile_process = subprocess.run(
                ['javac', source_file],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            if compile_process.returncode != 0:
                self._parse_java_errors(compile_process.stderr, result)
                return result
            
            # Execute
            class_dir = os.path.dirname(source_file)
            run_process = subprocess.run(
                ['java', '-cp', class_dir, class_name],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            result['success'] = True
            result['output'] = run_process.stdout
            
        except subprocess.TimeoutExpired:
            result['errors'].append({
                'type': 'timeout',
                'message': f'Execution exceeded {timeout} seconds'
            })
        except FileNotFoundError:
            result['errors'].append({
                'type': 'environment',
                'message': 'Java compiler not found'
            })
        finally:
            os.unlink(source_file)
            class_file = source_file.replace('.java', '.class')
            if os.path.exists(class_file):
                os.unlink(class_file)
        
        return result
    
    def _parse_python_errors(self, stderr: str, result: Dict):
        """Parse Python error messages"""
        lines = stderr.split('\n')
        for line in lines:
            for error_type, pattern in self.error_patterns.items():
                if pattern in line:
                    result['errors'].append({
                        'type': error_type,
                        'message': line,
                        'suggestion': self._suggest_fix_for_error(error_type, line)
                    })
                    break
    
    def _parse_js_errors(self, stderr: str, result: Dict):
        """Parse JavaScript error messages"""
        # Similar to Python error parsing
        lines = stderr.split('\n')
        for line in lines:
            if 'Error:' in line:
                result['errors'].append({
                    'type': 'runtime',
                    'message': line
                })
    
    def _parse_cpp_errors(self, stderr: str, result: Dict):
        """Parse C++ compilation errors"""
        lines = stderr.split('\n')
        for line in lines:
            if 'error:' in line.lower():
                result['errors'].append({
                    'type': 'compilation',
                    'message': line
                })
    
    def _parse_cpp_warnings(self, stderr: str, result: Dict):
        """Parse C++ warnings"""
        lines = stderr.split('\n')
        for line in lines:
            if 'warning:' in line.lower():
                result['warnings'].append({
                    'type': 'warning',
                    'message': line
                })
    
    def _parse_java_errors(self, stderr: str, result: Dict):
        """Parse Java compilation errors"""
        lines = stderr.split('\n')
        for line in lines:
            if 'error:' in line:
                result['errors'].append({
                    'type': 'compilation',
                    'message': line
                })
    
    def _suggest_fix_for_syntax_error(self, error: SyntaxError) -> str:
        """Suggest fix for syntax error"""
        if 'unexpected EOF' in str(error):
            return "Check for missing closing brackets or quotes"
        elif 'invalid syntax' in str(error):
            return "Check for typos or incorrect Python syntax"
        elif 'unexpected indent' in str(error):
            return "Check indentation consistency"
        return "Review syntax near the reported line"
    
    def _suggest_fix_for_error(self, error_type: str, message: str) -> str:
        """Suggest fix for runtime error"""
        suggestions = {
            'name': "Check if variable is defined before use",
            'type': "Check data types and operations",
            'index': "Check list/array bounds",
            'key': "Check if dictionary key exists",
            'value': "Check input values and conversions",
            'attribute': "Check object attributes and methods",
            'import': "Check module installation and import path"
        }
        return suggestions.get(error_type, "Review error message for details")