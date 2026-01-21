"""
Custom Tokenizer for CodeForge-Quantum
SentencePiece-based tokenizer with 65,536 vocabulary supporting code-specific tokens
"""

import sentencepiece as spm
import torch
from typing import List, Dict, Optional, Union, Tuple
import json
import os
from pathlib import Path
import numpy as np
from dataclasses import dataclass
import re


@dataclass 
class TokenizerConfig:
    """Configuration for custom tokenizer"""
    vocab_size: int = 65536
    model_type: str = "unigram"  # unigram, bpe, word, char
    normalization_rule_name: str = "identity"
    max_sentence_length: int = 8192
    pad_token: str = "<pad>"
    unk_token: str = "<unk>"
    bos_token: str = "<s>"
    eos_token: str = "</s>"
    additional_special_tokens: List[str] = None
    

class CodeTokenizer:
    """Custom tokenizer for code with special handling"""
    
    def __init__(self, config: TokenizerConfig):
        self.config = config
        self.sp_model = None
        self.vocab = {}
        self.reverse_vocab = {}
        
        # Special tokens
        self.special_tokens = {
            'pad': config.pad_token,
            'unk': config.unk_token,
            'bos': config.bos_token,
            'eos': config.eos_token
        }
        
        # Code-specific tokens
        self.code_tokens = [
            '<def>', '</def>',
            '<class>', '</class>',
            '<if>', '</if>',
            '<for>', '</for>',
            '<while>', '</while>',
            '<try>', '</try>',
            '<import>',
            '<return>',
            '<indent>', '</indent>',
            '<comment>', '</comment>',
            '<string>', '</string>',
            '<number>',
            '<operator>',
            '<keyword>',
            '<identifier>',
            '<newline>'
        ]
        
        # Language-specific tokens
        self.language_tokens = [
            '<python>', '<javascript>', '<java>', '<cpp>',
            '<csharp>', '<go>', '<rust>', '<swift>'
        ]
        
        # Initialize special token ids
        self.special_token_ids = {}
        self.code_token_ids = {}
        self.language_token_ids = {}
        
    def train(
        self,
        corpus_files: List[str],
        model_path: str,
        coverage: float = 0.9999
    ):
        """Train SentencePiece model on code corpus"""
        
        # Prepare training command
        train_args = [
            f"--input={','.join(corpus_files)}",
            f"--model_prefix={model_path}",
            f"--vocab_size={self.config.vocab_size}",
            f"--model_type={self.config.model_type}",
            f"--normalization_rule_name={self.config.normalization_rule_name}",
            f"--max_sentence_length={self.config.max_sentence_length}",
            f"--pad_id=0 --pad_piece={self.config.pad_token}",
            f"--unk_id=1 --unk_piece={self.config.unk_token}",
            f"--bos_id=2 --bos_piece={self.config.bos_token}",
            f"--eos_id=3 --eos_piece={self.config.eos_token}",
            f"--character_coverage={coverage}",
            "--shuffle_input_sentence=true",
            "--max_sentencepiece_length=16",
            "--split_by_unicode_script=true",
            "--split_by_number=true",
            "--split_by_whitespace=true",
            "--treat_whitespace_as_suffix=false",
            "--allow_whitespace_only_pieces=true"
        ]
        
        # Add user defined symbols
        all_special_tokens = (
            list(self.special_tokens.values()) +
            self.code_tokens +
            self.language_tokens
        )
        
        if self.config.additional_special_tokens:
            all_special_tokens.extend(self.config.additional_special_tokens)
        
        user_symbols = ','.join(all_special_tokens[4:])  # Skip pad, unk, bos, eos
        train_args.append(f"--user_defined_symbols={user_symbols}")
        
        # Train model
        spm.SentencePieceTrainer.train(' '.join(train_args))
        
        # Load trained model
        self.load(f"{model_path}.model")
        
    def load(self, model_path: str):
        """Load trained SentencePiece model"""
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.load(model_path)
        
        # Build vocabulary
        self._build_vocabulary()
        
    def _build_vocabulary(self):
        """Build vocabulary mappings"""
        for i in range(self.sp_model.get_piece_size()):
            token = self.sp_model.id_to_piece(i)
            self.vocab[token] = i
            self.reverse_vocab[i] = token
        
        # Map special tokens
        for name, token in self.special_tokens.items():
            if token in self.vocab:
                self.special_token_ids[name] = self.vocab[token]
        
        # Map code tokens
        for token in self.code_tokens:
            if token in self.vocab:
                self.code_token_ids[token] = self.vocab[token]
        
        # Map language tokens
        for token in self.language_tokens:
            if token in self.vocab:
                self.language_token_ids[token] = self.vocab[token]
    
    def tokenize(
        self,
        text: str,
        add_special_tokens: bool = True,
        language: Optional[str] = None
    ) -> List[str]:
        """Tokenize text into subwords"""
        
        # Preprocess code
        text = self._preprocess_code(text, language)
        
        # Tokenize with SentencePiece
        tokens = self.sp_model.encode_as_pieces(text)
        
        # Add special tokens
        if add_special_tokens:
            tokens = [self.config.bos_token] + tokens + [self.config.eos_token]
            
            # Add language token if specified
            if language and f"<{language}>" in self.language_tokens:
                tokens.insert(1, f"<{language}>")
        
        return tokens
    
    def encode(
        self,
        text: Union[str, List[str]],
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        padding: bool = False,
        truncation: bool = False,
        return_tensors: Optional[str] = None
    ) -> Union[List[int], torch.Tensor]:
        """Encode text to token ids"""
        
        if isinstance(text, str):
            tokens = self.tokenize(text, add_special_tokens)
        else:
            tokens = text
        
        # Convert to ids
        ids = [self.vocab.get(token, self.vocab[self.config.unk_token]) 
               for token in tokens]
        
        # Truncation
        if truncation and max_length:
            ids = ids[:max_length]
        
        # Padding
        if padding and max_length:
            pad_id = self.special_token_ids['pad']
            if len(ids) < max_length:
                ids = ids + [pad_id] * (max_length - len(ids))
        
        # Convert to tensor if requested
        if return_tensors == 'pt':
            return torch.tensor(ids)
        elif return_tensors == 'np':
            return np.array(ids)
        
        return ids
    
    def decode(
        self,
        ids: Union[List[int], torch.Tensor, np.ndarray],
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True
    ) -> str:
        """Decode token ids to text"""
        
        # Convert to list if tensor
        if isinstance(ids, (torch.Tensor, np.ndarray)):
            ids = ids.tolist()
        
        # Skip special tokens if requested
        if skip_special_tokens:
            special_ids = set(self.special_token_ids.values())
            special_ids.update(self.code_token_ids.values())
            ids = [id for id in ids if id not in special_ids]
        
        # Decode with SentencePiece
        text = self.sp_model.decode(ids)
        
        # Clean up
        if clean_up_tokenization_spaces:
            text = self._clean_up_tokenization(text)
        
        return text
    
    def _preprocess_code(self, code: str, language: Optional[str] = None) -> str:
        """Preprocess code before tokenization"""
        
        # Detect language if not specified
        if not language:
            language = self._detect_language(code)
        
        # Apply language-specific preprocessing
        if language == 'python':
            code = self._preprocess_python(code)
        elif language == 'javascript':
            code = self._preprocess_javascript(code)
        elif language == 'java':
            code = self._preprocess_java(code)
        # Add more languages as needed
        
        return code
    
    def _detect_language(self, code: str) -> str:
        """Simple language detection based on patterns"""
        
        # Python indicators
        if 'def ' in code or 'import ' in code or 'print(' in code:
            return 'python'
        
        # JavaScript indicators
        if 'function ' in code or 'const ' in code or 'console.log' in code:
            return 'javascript'
        
        # Java indicators
        if 'public class' in code or 'public static void main' in code:
            return 'java'
        
        # C++ indicators
        if '#include' in code or 'std::' in code or 'cout' in code:
            return 'cpp'
        
        return 'unknown'
    
    def _preprocess_python(self, code: str) -> str:
        """Python-specific preprocessing"""
        
        # Mark function definitions
        code = re.sub(r'^(\s*)def\s+', r'\1<def>def ', code, flags=re.MULTILINE)
        
        # Mark class definitions
        code = re.sub(r'^(\s*)class\s+', r'\1<class>class ', code, flags=re.MULTILINE)
        
        # Mark control flow
        code = re.sub(r'^(\s*)if\s+', r'\1<if>if ', code, flags=re.MULTILINE)
        code = re.sub(r'^(\s*)for\s+', r'\1<for>for ', code, flags=re.MULTILINE)
        code = re.sub(r'^(\s*)while\s+', r'\1<while>while ', code, flags=re.MULTILINE)
        
        # Mark imports
        code = re.sub(r'^(\s*)import\s+', r'\1<import>import ', code, flags=re.MULTILINE)
        code = re.sub(r'^(\s*)from\s+', r'\1<import>from ', code, flags=re.MULTILINE)
        
        return code
    
    def _preprocess_javascript(self, code: str) -> str:
        """JavaScript-specific preprocessing"""
        
        # Mark function definitions
        code = re.sub(r'function\s+', '<def>function ', code)
        code = re.sub(r'const\s+(\w+)\s*=\s*\(', r'<def>const \1 = (', code)
        
        # Mark control flow
        code = re.sub(r'if\s*\(', '<if>if (', code)
        code = re.sub(r'for\s*\(', '<for>for (', code)
        code = re.sub(r'while\s*\(', '<while>while (', code)
        
        return code
    
    def _preprocess_java(self, code: str) -> str:
        """Java-specific preprocessing"""
        
        # Mark class definitions
        code = re.sub(r'(public|private|protected)?\s*class\s+', 
                     r'\1 <class>class ', code)
        
        # Mark method definitions
        code = re.sub(r'(public|private|protected)?\s*(static)?\s*\w+\s+\w+\s*\(',
                     r'<def>\g<0>', code)
        
        return code
    
    def _clean_up_tokenization(self, text: str) -> str:
        """Clean up tokenization artifacts"""
        
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Fix spacing around punctuation
        text = re.sub(r'\s+([.,;:!?])', r'\1', text)
        text = re.sub(r'([({])\s+', r'\1', text)
        text = re.sub(r'\s+([)}])', r'\1', text)
        
        # Remove special tokens that shouldn't appear
        for token in self.code_tokens:
            text = text.replace(token, '')
        
        return text.strip()
    
    def batch_encode(
        self,
        texts: List[str],
        max_length: Optional[int] = None,
        padding: bool = True,
        truncation: bool = True,
        return_tensors: str = 'pt'
    ) -> Dict[str, torch.Tensor]:
        """Batch encode multiple texts"""
        
        # Encode each text
        encoded = []
        for text in texts:
            ids = self.encode(
                text,
                max_length=max_length,
                padding=False,
                truncation=truncation,
                return_tensors=None
            )
            encoded.append(ids)
        
        # Find max length
        if not max_length:
            max_length = max(len(ids) for ids in encoded)
        
        # Pad to same length
        if padding:
            pad_id = self.special_token_ids['pad']
            padded = []
            attention_masks = []
            
            for ids in encoded:
                if len(ids) < max_length:
                    padding_length = max_length - len(ids)
                    attention_mask = [1] * len(ids) + [0] * padding_length
                    ids = ids + [pad_id] * padding_length
                else:
                    attention_mask = [1] * max_length
                
                padded.append(ids)
                attention_masks.append(attention_mask)
            
            encoded = padded
        else:
            attention_masks = [[1] * len(ids) for ids in encoded]
        
        # Convert to tensors
        if return_tensors == 'pt':
            input_ids = torch.tensor(encoded)
            attention_mask = torch.tensor(attention_masks)
        elif return_tensors == 'np':
            input_ids = np.array(encoded)
            attention_mask = np.array(attention_masks)
        else:
            input_ids = encoded
            attention_mask = attention_masks
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size"""
        return len(self.vocab)
    
    def save_vocabulary(self, path: str):
        """Save vocabulary to file"""
        vocab_file = Path(path) / 'vocab.json'
        with open(vocab_file, 'w') as f:
            json.dump(self.vocab, f, indent=2)
        
        # Save config
        config_file = Path(path) / 'tokenizer_config.json'
        with open(config_file, 'w') as f:
            json.dump({
                'vocab_size': self.config.vocab_size,
                'model_type': self.config.model_type,
                'special_tokens': self.special_tokens,
                'code_tokens': self.code_tokens,
                'language_tokens': self.language_tokens
            }, f, indent=2)
    
    @classmethod
    def from_pretrained(cls, path: str) -> 'CodeTokenizer':
        """Load tokenizer from saved files"""
        
        # Load config
        config_file = Path(path) / 'tokenizer_config.json'
        with open(config_file, 'r') as f:
            config_dict = json.load(f)
        
        # Create config
        config = TokenizerConfig(
            vocab_size=config_dict['vocab_size'],
            model_type=config_dict['model_type']
        )
        
        # Create tokenizer
        tokenizer = cls(config)
        
        # Load model
        model_file = Path(path) / 'tokenizer.model'
        if model_file.exists():
            tokenizer.load(str(model_file))
        
        # Load vocabulary
        vocab_file = Path(path) / 'vocab.json'
        if vocab_file.exists():
            with open(vocab_file, 'r') as f:
                tokenizer.vocab = json.load(f)
                tokenizer.reverse_vocab = {v: k for k, v in tokenizer.vocab.items()}
        
        return tokenizer