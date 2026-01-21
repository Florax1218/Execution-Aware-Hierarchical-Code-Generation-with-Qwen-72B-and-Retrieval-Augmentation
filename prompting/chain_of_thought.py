"""
Chain-of-Thought (CoT) Prompting Enhancement Module
Multi-step reasoning through structured prompting templates and Markov Decision Process
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import json
from enum import Enum


class ReasoningStep(Enum):
    """Types of reasoning steps in CoT"""
    PROBLEM_UNDERSTANDING = "problem_understanding"
    DECOMPOSITION = "decomposition"
    ALGORITHM_SELECTION = "algorithm_selection"
    DATA_STRUCTURE = "data_structure"
    EDGE_CASES = "edge_cases"
    OPTIMIZATION = "optimization"
    IMPLEMENTATION = "implementation"
    VERIFICATION = "verification"


@dataclass
class ThoughtStep:
    """Single step in chain of thought reasoning"""
    step_type: ReasoningStep
    content: str
    confidence: float
    dependencies: List[int]  # Indices of dependent steps
    metadata: Dict[str, Any]


class PromptTemplate:
    """Structured prompting templates for different reasoning steps"""
    
    TEMPLATES = {
        ReasoningStep.PROBLEM_UNDERSTANDING: """
Let's understand the problem step by step:
1. Input: {input_description}
2. Expected Output: {output_description}
3. Constraints: {constraints}
4. Key Requirements: {requirements}

Problem Type: {problem_type}
Core Challenge: {core_challenge}
""",
        
        ReasoningStep.DECOMPOSITION: """
Breaking down the problem into subproblems:
{subproblems}

Dependencies between subproblems:
{dependencies}

Solution approach:
{approach}
""",
        
        ReasoningStep.ALGORITHM_SELECTION: """
Selecting appropriate algorithm:
- Algorithm: {algorithm_name}
- Time Complexity: {time_complexity}
- Space Complexity: {space_complexity}
- Justification: {justification}

Alternative algorithms considered:
{alternatives}
""",
        
        ReasoningStep.DATA_STRUCTURE: """
Data structures needed:
{data_structures}

Rationale for each choice:
{rationale}
""",
        
        ReasoningStep.EDGE_CASES: """
Identifying edge cases:
{edge_cases}

Handling strategies:
{handling_strategies}
""",
        
        ReasoningStep.OPTIMIZATION: """
Optimization opportunities:
{optimizations}

Trade-offs:
{tradeoffs}
""",
        
        ReasoningStep.IMPLEMENTATION: """
Implementation plan:
{implementation_steps}

Key functions:
{functions}
""",
        
        ReasoningStep.VERIFICATION: """
Verification strategy:
- Test cases: {test_cases}
- Expected behavior: {expected_behavior}
- Correctness proof sketch: {proof_sketch}
"""
    }
    
    @classmethod
    def fill_template(cls, step_type: ReasoningStep, **kwargs) -> str:
        """Fill a template with provided values"""
        template = cls.TEMPLATES.get(step_type, "")
        try:
            return template.format(**kwargs)
        except KeyError as e:
            return f"Error filling template: missing key {e}"


class ChainOfThoughtLSTM(nn.Module):
    """LSTM-based chain of thought reasoning module"""
    def __init__(
        self,
        hidden_dim: int = 8192,
        num_layers: int = 3,
        dropout: float = 0.1,
        temperature: float = 0.8
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.temperature = temperature
        
        # LSTM for sequential reasoning
        self.reasoning_lstm = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=False
        )
        
        # Reasoning projection
        self.reasoning_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Step type classifier
        self.step_classifier = nn.Linear(hidden_dim, len(ReasoningStep))
        
        # Confidence predictor
        self.confidence_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Dependency attention
        self.dependency_attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
    def forward(
        self,
        input_hidden: torch.Tensor,
        num_steps: int = 5,
        previous_steps: Optional[List[torch.Tensor]] = None
    ) -> Tuple[List[torch.Tensor], List[float], List[ReasoningStep]]:
        """Generate chain of thought reasoning steps"""
        batch_size = input_hidden.size(0)
        device = input_hidden.device
        
        # Initialize LSTM hidden state
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        
        reasoning_steps = []
        confidences = []
        step_types = []
        
        current_input = input_hidden.unsqueeze(1)  # Add sequence dimension
        hidden = (h_0, c_0)
        
        for step_idx in range(num_steps):
            # LSTM forward pass
            lstm_out, hidden = self.reasoning_lstm(current_input, hidden)
            
            # Apply reasoning projection with temperature
            reasoning = self.reasoning_projection(lstm_out.squeeze(1))
            reasoning = reasoning / self.temperature
            
            # If we have previous steps, apply dependency attention
            if previous_steps:
                prev_tensor = torch.stack(previous_steps, dim=1)
                attended_reasoning, _ = self.dependency_attention(
                    reasoning.unsqueeze(1),
                    prev_tensor,
                    prev_tensor
                )
                reasoning = reasoning + 0.5 * attended_reasoning.squeeze(1)
            
            # Classify step type
            step_logits = self.step_classifier(reasoning)
            step_probs = F.softmax(step_logits, dim=-1)
            step_type_idx = torch.argmax(step_probs, dim=-1)
            step_type = list(ReasoningStep)[step_type_idx[0].item()]
            
            # Predict confidence
            confidence = self.confidence_predictor(reasoning)
            
            # Store results
            reasoning_steps.append(reasoning)
            confidences.append(confidence.squeeze(-1))
            step_types.append(step_type)
            
            # Update input for next step
            current_input = reasoning.unsqueeze(1)
            
            # Add to previous steps for dependency attention
            if previous_steps is None:
                previous_steps = [reasoning]
            else:
                previous_steps.append(reasoning)
        
        return reasoning_steps, confidences, step_types


class MarkovDecisionProcess(nn.Module):
    """MDP for modeling reasoning step transitions"""
    def __init__(
        self,
        num_states: int,
        hidden_dim: int = 8192,
        gamma: float = 0.95
    ):
        super().__init__()
        self.num_states = num_states
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        
        # Transition model
        self.transition_model = nn.Sequential(
            nn.Linear(hidden_dim + num_states, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_states)
        )
        
        # Value function
        self.value_function = nn.Sequential(
            nn.Linear(hidden_dim + num_states, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Policy network
        self.policy_network = nn.Sequential(
            nn.Linear(hidden_dim + num_states, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_states)
        )
        
    def forward(
        self,
        state_hidden: torch.Tensor,
        current_state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute next state probabilities, value, and policy"""
        # Concatenate hidden state and current state
        combined = torch.cat([state_hidden, current_state], dim=-1)
        
        # Get transition probabilities
        next_state_logits = self.transition_model(combined)
        next_state_probs = F.softmax(next_state_logits, dim=-1)
        
        # Compute state value
        value = self.value_function(combined)
        
        # Get policy
        policy_logits = self.policy_network(combined)
        policy = F.softmax(policy_logits, dim=-1)
        
        return next_state_probs, value, policy
    
    def compute_returns(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor
    ) -> torch.Tensor:
        """Compute discounted returns for reinforcement learning"""
        returns = torch.zeros_like(rewards)
        running_return = 0
        
        for t in reversed(range(len(rewards))):
            if dones[t]:
                running_return = 0
            running_return = rewards[t] + self.gamma * running_return
            returns[t] = running_return
        
        return returns


class StructuredReasoning:
    """Structured reasoning system for code generation"""
    def __init__(
        self,
        hidden_dim: int = 8192,
        max_reasoning_steps: int = 8
    ):
        self.hidden_dim = hidden_dim
        self.max_reasoning_steps = max_reasoning_steps
        
        # Initialize components
        self.cot_lstm = ChainOfThoughtLSTM(hidden_dim)
        self.mdp = MarkovDecisionProcess(len(ReasoningStep), hidden_dim)
        
        # Reasoning state tracker
        self.reasoning_history = []
        
    def reason_about_problem(
        self,
        problem_description: str,
        problem_hidden: torch.Tensor
    ) -> Dict[str, Any]:
        """Complete reasoning process for a problem"""
        
        # Generate chain of thought
        reasoning_steps, confidences, step_types = self.cot_lstm(
            problem_hidden,
            num_steps=self.max_reasoning_steps
        )
        
        # Build structured reasoning output
        structured_output = {
            'problem': problem_description,
            'reasoning_chain': [],
            'final_approach': None,
            'confidence': 0.0
        }
        
        # Process each reasoning step
        for i, (step_hidden, confidence, step_type) in enumerate(
            zip(reasoning_steps, confidences, step_types)
        ):
            step_info = {
                'step': i + 1,
                'type': step_type.value,
                'confidence': confidence.item(),
                'content': self._generate_step_content(step_type, step_hidden)
            }
            structured_output['reasoning_chain'].append(step_info)
        
        # Aggregate final approach
        structured_output['final_approach'] = self._synthesize_approach(
            reasoning_steps, step_types
        )
        
        # Overall confidence
        structured_output['confidence'] = torch.stack(confidences).mean().item()
        
        # Store in history
        self.reasoning_history.append(structured_output)
        
        return structured_output
    
    def _generate_step_content(
        self,
        step_type: ReasoningStep,
        step_hidden: torch.Tensor
    ) -> str:
        """Generate natural language content for a reasoning step"""
        # This would use a language model in practice
        # For now, return a template-based response
        
        content_templates = {
            ReasoningStep.PROBLEM_UNDERSTANDING: "Understanding the core requirements and constraints",
            ReasoningStep.DECOMPOSITION: "Breaking down into manageable subproblems",
            ReasoningStep.ALGORITHM_SELECTION: "Selecting optimal algorithm for the task",
            ReasoningStep.DATA_STRUCTURE: "Choosing appropriate data structures",
            ReasoningStep.EDGE_CASES: "Identifying and handling edge cases",
            ReasoningStep.OPTIMIZATION: "Optimizing for performance",
            ReasoningStep.IMPLEMENTATION: "Planning the implementation",
            ReasoningStep.VERIFICATION: "Verifying correctness"
        }
        
        return content_templates.get(step_type, "Processing...")
    
    def _synthesize_approach(
        self,
        reasoning_steps: List[torch.Tensor],
        step_types: List[ReasoningStep]
    ) -> str:
        """Synthesize final approach from reasoning steps"""
        approach_parts = []
        
        for step_type in step_types:
            if step_type == ReasoningStep.ALGORITHM_SELECTION:
                approach_parts.append("Use efficient algorithm")
            elif step_type == ReasoningStep.DATA_STRUCTURE:
                approach_parts.append("with appropriate data structures")
            elif step_type == ReasoningStep.OPTIMIZATION:
                approach_parts.append("and optimization techniques")
        
        return " ".join(approach_parts) if approach_parts else "Standard implementation approach"


class PromptEngineering:
    """Advanced prompt engineering for code generation"""
    
    @staticmethod
    def create_few_shot_prompt(
        task_description: str,
        examples: List[Dict[str, str]],
        num_shots: int = 3
    ) -> str:
        """Create few-shot prompt with examples"""
        prompt_parts = [
            "You are an expert programmer. Generate code based on the following examples and task.\n"
        ]
        
        # Add examples
        for i, example in enumerate(examples[:num_shots]):
            prompt_parts.append(f"\nExample {i+1}:")
            prompt_parts.append(f"Task: {example['task']}")
            prompt_parts.append(f"Solution:\n{example['solution']}\n")
        
        # Add current task
        prompt_parts.append(f"\nNow, solve this task:")
        prompt_parts.append(f"Task: {task_description}")
        prompt_parts.append(f"Solution:")
        
        return "\n".join(prompt_parts)
    
    @staticmethod
    def create_cot_prompt(
        task_description: str,
        reasoning_steps: List[str]
    ) -> str:
        """Create chain-of-thought prompt"""
        prompt_parts = [
            "Let's solve this step-by-step:",
            f"Task: {task_description}\n",
            "Reasoning:"
        ]
        
        for i, step in enumerate(reasoning_steps, 1):
            prompt_parts.append(f"Step {i}: {step}")
        
        prompt_parts.append("\nBased on this reasoning, here's the solution:")
        
        return "\n".join(prompt_parts)
    
    @staticmethod
    def create_structured_prompt(
        task_description: str,
        constraints: List[str],
        requirements: List[str],
        hints: Optional[List[str]] = None
    ) -> str:
        """Create structured prompt with clear sections"""
        prompt = f"""
## Task
{task_description}

## Requirements
{chr(10).join(f"- {req}" for req in requirements)}

## Constraints
{chr(10).join(f"- {const}" for const in constraints)}
"""
        
        if hints:
            prompt += f"""
## Hints
{chr(10).join(f"- {hint}" for hint in hints)}
"""
        
        prompt += """
## Solution
Please provide a complete, efficient, and well-documented solution:
"""
        
        return prompt
    
    @staticmethod
    def create_refinement_prompt(
        original_code: str,
        feedback: str,
        specific_issues: List[str]
    ) -> str:
        """Create prompt for code refinement"""
        return f"""
## Original Code
```python
{original_code}
```

## Feedback
{feedback}

## Specific Issues to Address
{chr(10).join(f"- {issue}" for issue in specific_issues)}

## Refined Solution
Please provide an improved version that addresses all the issues:
"""