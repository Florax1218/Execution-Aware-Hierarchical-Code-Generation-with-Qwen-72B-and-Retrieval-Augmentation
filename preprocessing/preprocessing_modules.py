"""
Code Augmentation and Preprocessing Pipeline
Implements sophisticated transformations for training data augmentation
"""

import ast
import random
import re
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np
from collections import defaultdict


class CodeAugmentation:
    """Advanced code augmentation techniques"""
    
    def __init__(self, augmentation_prob: float = 0.5):
        self.augmentation_prob = augmentation_prob
        
        # Variable name pools for renaming
        self.var_name_pools = {
            'generic': ['x', 'y', 'z', 'a', 'b', 'c', 'i', 'j', 'k', 'n', 'm'],
            'descriptive': ['result', 'temp', 'value', 'data', 'item', 'element',
                          'count', 'total', 'sum', 'product', 'difference'],
            'typed': ['num', 'str_val', 'list_data', 'dict_obj', 'bool_flag',
                     'int_val', 'float_num', 'tuple_data', 'set_items']
        }
        
    def augment(self, code: str) -> str:
        """Apply random augmentation to code"""
        if random.random() > self.augmentation_prob:
            return code
        
        augmentations = [
            self.rename_variables,
            self.reorder_independent_statements,
            self.add_comments,
            self.change_loop_style,
            self.inline_simple_functions,
            self.extract_constants,
            self.add_type_hints
        ]
        
        # Apply 1-3 random augmentations
        num_augmentations = random.randint(1, 3)
        selected = random.sample(augmentations, num_augmentations)
        
        augmented_code = code
        for aug_func in selected:
            try:
                augmented_code = aug_func(augmented_code)
            except:
                # If augmentation fails, keep original
                pass
        
        return augmented_code
    
    def rename_variables(self, code: str) -> str:
        """Rename variables consistently throughout the code"""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return code
        
        # Collect all variable names
        class VariableCollector(ast.NodeVisitor):
            def __init__(self):
                self.variables = set()
                self.functions = set()
                self.classes = set()
                
            def visit_Name(self, node):
                if isinstance(node.ctx, (ast.Store, ast.Del)):
                    self.variables.add(node.id)
                self.generic_visit(node)
                
            def visit_FunctionDef(self, node):
                self.functions.add(node.name)
                for arg in node.args.args:
                    self.variables.add(arg.arg)
                self.generic_visit(node)
                
            def visit_ClassDef(self, node):
                self.classes.add(node.name)
                self.generic_visit(node)
        
        collector = VariableCollector()
        collector.visit(tree)
        
        # Create renaming map (excluding built-ins and special names)
        builtin_names = {'print', 'len', 'range', 'str', 'int', 'float', 'list',
                        'dict', 'set', 'tuple', 'True', 'False', 'None'}
        
        rename_map = {}
        pool = random.choice(list(self.var_name_pools.values()))
        
        for var in collector.variables:
            if var not in builtin_names and not var.startswith('_'):
                if pool:
                    new_name = random.choice(pool)
                    # Ensure unique names
                    suffix = 1
                    while new_name in rename_map.values():
                        new_name = f"{random.choice(pool)}{suffix}"
                        suffix += 1
                    rename_map[var] = new_name
        
        # Apply renaming
        class VariableRenamer(ast.NodeTransformer):
            def visit_Name(self, node):
                if node.id in rename_map:
                    node.id = rename_map[node.id]
                return node
            
            def visit_arg(self, node):
                if node.arg in rename_map:
                    node.arg = rename_map[node.arg]
                return node
        
        renamer = VariableRenamer()
        new_tree = renamer.visit(tree)
        
        return ast.unparse(new_tree)
    
    def reorder_independent_statements(self, code: str) -> str:
        """Reorder statements that don't have dependencies"""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return code
        
        # Simple reordering for module-level statements
        if isinstance(tree, ast.Module):
            statements = tree.body
            
            # Group statements by type
            imports = []
            functions = []
            classes = []
            assignments = []
            others = []
            
            for stmt in statements:
                if isinstance(stmt, (ast.Import, ast.ImportFrom)):
                    imports.append(stmt)
                elif isinstance(stmt, ast.FunctionDef):
                    functions.append(stmt)
                elif isinstance(stmt, ast.ClassDef):
                    classes.append(stmt)
                elif isinstance(stmt, ast.Assign):
                    assignments.append(stmt)
                else:
                    others.append(stmt)
            
            # Shuffle within groups (maintaining dependencies would require more analysis)
            random.shuffle(assignments)
            random.shuffle(functions)
            
            # Reconstruct in a sensible order
            tree.body = imports + classes + functions + assignments + others
        
        return ast.unparse(tree)
    
    def add_comments(self, code: str) -> str:
        """Add explanatory comments to code"""
        lines = code.split('\n')
        augmented_lines = []
        
        comment_templates = [
            "# Process the {operation}",
            "# Handle {task}",
            "# Compute {calculation}",
            "# Check {condition}",
            "# Initialize {variable}",
            "# Update {state}",
            "# Return {result}"
        ]
        
        for line in lines:
            augmented_lines.append(line)
            
            # Add comments before certain patterns
            if random.random() < 0.3:
                if 'def ' in line:
                    augmented_lines.insert(-1, "# Function definition")
                elif 'for ' in line:
                    augmented_lines.insert(-1, "# Iterate through elements")
                elif 'if ' in line:
                    augmented_lines.insert(-1, "# Conditional check")
                elif 'return ' in line:
                    augmented_lines.insert(-1, "# Return result")
                elif '=' in line and '==' not in line:
                    augmented_lines.insert(-1, "# Variable assignment")
        
        return '\n'.join(augmented_lines)
    
    def change_loop_style(self, code: str) -> str:
        """Convert between different loop styles where possible"""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return code
        
        class LoopTransformer(ast.NodeTransformer):
            def visit_For(self, node):
                # Try to convert simple range loops to while loops
                if (isinstance(node.iter, ast.Call) and
                    isinstance(node.iter.func, ast.Name) and
                    node.iter.func.id == 'range'):
                    
                    # Only convert simple range(n) cases
                    if len(node.iter.args) == 1:
                        # Create equivalent while loop
                        # This is complex and might break code, so we skip it
                        pass
                
                return self.generic_visit(node)
        
        transformer = LoopTransformer()
        new_tree = transformer.visit(tree)
        
        return ast.unparse(new_tree)
    
    def inline_simple_functions(self, code: str) -> str:
        """Inline very simple function calls"""
        # This is complex and could break functionality
        # For safety, we'll just return the original code
        return code
    
    def extract_constants(self, code: str) -> str:
        """Extract magic numbers as named constants"""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return code
        
        class ConstantExtractor(ast.NodeTransformer):
            def __init__(self):
                self.constants = {}
                self.counter = 0
                
            def visit_Constant(self, node):
                # Extract numeric constants > 1
                if isinstance(node.value, (int, float)) and abs(node.value) > 1:
                    const_name = f"CONST_{self.counter}"
                    self.constants[const_name] = node.value
                    self.counter += 1
                    # Replace with variable name
                    return ast.Name(id=const_name, ctx=ast.Load())
                return node
        
        extractor = ConstantExtractor()
        new_tree = extractor.visit(tree)
        
        # Add constant definitions at the beginning
        if extractor.constants:
            const_assigns = []
            for name, value in extractor.constants.items():
                const_assigns.append(
                    ast.Assign(
                        targets=[ast.Name(id=name, ctx=ast.Store())],
                        value=ast.Constant(value=value)
                    )
                )
            
            if isinstance(new_tree, ast.Module):
                # Add after imports
                import_end = 0
                for i, stmt in enumerate(new_tree.body):
                    if not isinstance(stmt, (ast.Import, ast.ImportFrom)):
                        import_end = i
                        break
                
                new_tree.body = (new_tree.body[:import_end] + 
                                const_assigns + 
                                new_tree.body[import_end:])
        
        return ast.unparse(new_tree)
    
    def add_type_hints(self, code: str) -> str:
        """Add type hints to function signatures"""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return code
        
        class TypeHintAdder(ast.NodeTransformer):
            def visit_FunctionDef(self, node):
                # Add simple type hints based on parameter names
                for arg in node.args.args:
                    if not arg.annotation:
                        # Guess type from name
                        if 'num' in arg.arg or 'count' in arg.arg:
                            arg.annotation = ast.Name(id='int', ctx=ast.Load())
                        elif 'str' in arg.arg or 'text' in arg.arg:
                            arg.annotation = ast.Name(id='str', ctx=ast.Load())
                        elif 'list' in arg.arg or 'items' in arg.arg:
                            arg.annotation = ast.Name(id='list', ctx=ast.Load())
                        elif 'dict' in arg.arg:
                            arg.annotation = ast.Name(id='dict', ctx=ast.Load())
                
                return self.generic_visit(node)
        
        adder = TypeHintAdder()
        new_tree = adder.visit(tree)
        
        return ast.unparse(new_tree)


class VariableNormalization:
    """Normalize variable names for consistency"""
    
    def __init__(self):
        self.normalization_rules = {
            'camelCase': self.to_camel_case,
            'snake_case': self.to_snake_case,
            'PascalCase': self.to_pascal_case
        }
        
    def normalize(self, code: str, style: str = 'snake_case') -> str:
        """Normalize all variable names to specified style"""
        if style not in self.normalization_rules:
            return code
        
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return code
        
        normalizer = self.normalization_rules[style]
        
        class NameNormalizer(ast.NodeTransformer):
            def visit_Name(self, node):
                # Don't normalize built-ins
                if not self._is_builtin(node.id):
                    node.id = normalizer(node.id)
                return node
            
            def visit_arg(self, node):
                if not self._is_builtin(node.arg):
                    node.arg = normalizer(node.arg)
                return node
            
            def visit_FunctionDef(self, node):
                if not self._is_builtin(node.name):
                    node.name = normalizer(node.name)
                return self.generic_visit(node)
            
            def _is_builtin(self, name):
                builtins = {'print', 'len', 'range', 'str', 'int', 'float',
                          'list', 'dict', 'set', 'tuple', 'True', 'False', 'None'}
                return name in builtins or name.startswith('_')
        
        transformer = NameNormalizer()
        new_tree = transformer.visit(tree)
        
        return ast.unparse(new_tree)
    
    def to_snake_case(self, name: str) -> str:
        """Convert to snake_case"""
        # Insert underscore before capitals
        result = re.sub('([A-Z])', r'_\1', name)
        # Handle multiple capitals
        result = re.sub('([A-Z]+)([A-Z][a-z])', r'\1_\2', result)
        # Clean up and lowercase
        result = result.strip('_').lower()
        # Replace multiple underscores with single
        result = re.sub('_+', '_', result)
        return result
    
    def to_camel_case(self, name: str) -> str:
        """Convert to camelCase"""
        parts = name.split('_')
        if parts:
            return parts[0].lower() + ''.join(p.capitalize() for p in parts[1:])
        return name
    
    def to_pascal_case(self, name: str) -> str:
        """Convert to PascalCase"""
        parts = name.split('_')
        return ''.join(p.capitalize() for p in parts)


class CurriculumScheduler:
    """Schedule training samples by difficulty"""
    
    def __init__(
        self,
        total_epochs: int,
        warmup_epochs: int = 3,
        difficulty_levels: List[str] = None
    ):
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.difficulty_levels = difficulty_levels or ['easy', 'medium', 'hard']
        self.current_epoch = 0
        
        # Difficulty progression schedule
        self.schedule = self._create_schedule()
        
    def _create_schedule(self) -> Dict[int, Dict[str, float]]:
        """Create curriculum learning schedule"""
        schedule = {}
        
        for epoch in range(self.total_epochs):
            if epoch < self.warmup_epochs:
                # Start with easy problems
                progress = epoch / self.warmup_epochs
                schedule[epoch] = {
                    'easy': 1.0,
                    'medium': 0.3 * progress,
                    'hard': 0.1 * progress
                }
            elif epoch < self.total_epochs // 2:
                # Gradually increase difficulty
                progress = (epoch - self.warmup_epochs) / (self.total_epochs // 2 - self.warmup_epochs)
                schedule[epoch] = {
                    'easy': 1.0,
                    'medium': 0.3 + 0.7 * progress,
                    'hard': 0.1 + 0.4 * progress
                }
            else:
                # Full difficulty range
                schedule[epoch] = {
                    'easy': 1.0,
                    'medium': 1.0,
                    'hard': 0.5 + 0.5 * (epoch - self.total_epochs // 2) / (self.total_epochs // 2)
                }
        
        return schedule
    
    def get_difficulty_weights(self, epoch: int) -> Dict[str, float]:
        """Get sampling weights for each difficulty level"""
        return self.schedule.get(epoch, {'easy': 1.0, 'medium': 1.0, 'hard': 1.0})
    
    def should_include_sample(self, difficulty: str, epoch: int) -> bool:
        """Decide whether to include a sample based on its difficulty"""
        weights = self.get_difficulty_weights(epoch)
        probability = weights.get(difficulty, 1.0)
        return random.random() < probability
    
    def step(self):
        """Move to next epoch"""
        self.current_epoch = min(self.current_epoch + 1, self.total_epochs - 1)
    
    def get_curriculum_info(self) -> Dict[str, Any]:
        """Get information about curriculum schedule"""
        return {
            'current_epoch': self.current_epoch,
            'total_epochs': self.total_epochs,
            'warmup_epochs': self.warmup_epochs,
            'current_weights': self.get_difficulty_weights(self.current_epoch),
            'schedule': self.schedule
        }


class DataBalancer:
    """Balance training data using focal sampling"""
    
    def __init__(self, gamma: float = 2.0):
        self.gamma = gamma
        self.sample_difficulties = {}
        
    def compute_sampling_weights(
        self,
        samples: List[Dict],
        model_predictions: Optional[Dict[str, float]] = None
    ) -> np.ndarray:
        """Compute sampling weights using focal sampling"""
        n = len(samples)
        weights = np.ones(n)
        
        if model_predictions:
            # Use model confidence for focal sampling
            for i, sample in enumerate(samples):
                sample_id = sample.get('id', str(i))
                confidence = model_predictions.get(sample_id, 0.5)
                
                # Focal sampling: upweight hard examples
                weight = (1 - confidence) ** self.gamma
                weights[i] = weight
        else:
            # Use difficulty-based weights
            difficulty_weights = {'easy': 0.5, 'medium': 1.0, 'hard': 2.0}
            
            for i, sample in enumerate(samples):
                difficulty = sample.get('difficulty', 'medium')
                weights[i] = difficulty_weights.get(difficulty, 1.0)
        
        # Normalize weights
        weights = weights / weights.sum()
        
        return weights
    
    def sample_batch(
        self,
        samples: List[Dict],
        batch_size: int,
        weights: Optional[np.ndarray] = None
    ) -> List[Dict]:
        """Sample a balanced batch"""
        if weights is None:
            weights = self.compute_sampling_weights(samples)
        
        indices = np.random.choice(
            len(samples),
            size=batch_size,
            p=weights,
            replace=True
        )
        
        return [samples[i] for i in indices]