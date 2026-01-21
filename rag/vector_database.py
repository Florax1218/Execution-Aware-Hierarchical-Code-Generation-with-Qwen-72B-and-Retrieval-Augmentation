"""
RAG System with FAISS Vector Database and Dense Retriever
Implements hierarchical navigable small world graphs for efficient retrieval
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import faiss
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import json
import pickle
from sentence_transformers import SentenceTransformer
from collections import defaultdict
import hashlib


@dataclass
class CodeSnippet:
    """Data structure for code snippets in vector database"""
    id: str
    code: str
    language: str
    description: str
    tags: List[str]
    complexity: int
    embedding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = None
    
    def to_dict(self):
        return {
            'id': self.id,
            'code': self.code,
            'language': self.language,
            'description': self.description,
            'tags': self.tags,
            'complexity': self.complexity,
            'metadata': self.metadata
        }


class DensePassageRetriever(nn.Module):
    """Dense retriever with contrastive learning for code retrieval"""
    def __init__(
        self,
        model_name: str = "microsoft/codebert-base",
        hidden_dim: int = 768,
        temperature: float = 0.07,
        similarity_beta: float = 0.3
    ):
        super().__init__()
        self.encoder = SentenceTransformer(model_name)
        self.hidden_dim = hidden_dim
        self.temperature = temperature
        self.similarity_beta = similarity_beta
        
        # Learned projection matrix for similarity computation
        self.W = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        nn.init.xavier_uniform_(self.W)
        
        # Query and document encoders
        self.query_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.doc_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def encode_query(self, query: str) -> torch.Tensor:
        """Encode query string to embedding"""
        # Get base embedding
        base_embedding = self.encoder.encode(query, convert_to_tensor=True)
        
        # Apply query-specific transformation
        query_embedding = self.query_encoder(base_embedding)
        return F.normalize(query_embedding, p=2, dim=-1)
    
    def encode_document(self, document: str) -> torch.Tensor:
        """Encode document to embedding"""
        # Get base embedding
        base_embedding = self.encoder.encode(document, convert_to_tensor=True)
        
        # Apply document-specific transformation
        doc_embedding = self.doc_encoder(base_embedding)
        return F.normalize(doc_embedding, p=2, dim=-1)
    
    def compute_similarity(
        self,
        query_embedding: torch.Tensor,
        doc_embedding: torch.Tensor,
        query_tokens: List[str],
        doc_tokens: List[str]
    ) -> torch.Tensor:
        """Compute hybrid similarity with learned metric"""
        # Semantic similarity with learned projection
        projected_doc = torch.matmul(doc_embedding, self.W)
        semantic_sim = F.cosine_similarity(
            query_embedding.unsqueeze(0),
            projected_doc.unsqueeze(0),
            dim=-1
        )
        
        # Lexical similarity (Jaccard)
        query_set = set(query_tokens)
        doc_set = set(doc_tokens)
        jaccard_sim = len(query_set & doc_set) / len(query_set | doc_set) if query_set | doc_set else 0
        
        # Combine similarities
        total_sim = semantic_sim + self.similarity_beta * jaccard_sim
        return total_sim
    
    def contrastive_loss(
        self,
        query: torch.Tensor,
        positive: torch.Tensor,
        negatives: torch.Tensor
    ) -> torch.Tensor:
        """Compute contrastive loss for training"""
        # Positive similarity
        pos_sim = F.cosine_similarity(query, positive, dim=-1) / self.temperature
        
        # Negative similarities
        neg_sims = torch.matmul(query, negatives.T) / self.temperature
        
        # Contrastive loss
        numerator = torch.exp(pos_sim)
        denominator = numerator + torch.sum(torch.exp(neg_sims), dim=-1)
        
        loss = -torch.log(numerator / denominator)
        return loss.mean()


class FAISSVectorDatabase:
    """FAISS-based vector database with hierarchical navigable small world graphs"""
    def __init__(
        self,
        dimension: int = 768,
        index_type: str = "HNSW",
        m: int = 32,  # HNSW parameter
        ef_construction: int = 200,  # HNSW parameter
        ef_search: int = 100  # HNSW parameter
    ):
        self.dimension = dimension
        self.index_type = index_type
        
        # Initialize FAISS index
        if index_type == "HNSW":
            self.index = faiss.IndexHNSWFlat(dimension, m, faiss.METRIC_INNER_PRODUCT)
            self.index.hnsw.efConstruction = ef_construction
            self.index.hnsw.efSearch = ef_search
        elif index_type == "IVF":
            quantizer = faiss.IndexFlatIP(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, 100, faiss.METRIC_INNER_PRODUCT)
        else:
            self.index = faiss.IndexFlatIP(dimension)
        
        # Storage for code snippets
        self.snippets = {}
        self.id_to_idx = {}
        self.idx_to_id = {}
        self.current_idx = 0
        
        # Inverted index for fast keyword lookup
        self.inverted_index = defaultdict(set)
        
        # Statistics
        self.stats = {
            'total_snippets': 0,
            'languages': defaultdict(int),
            'complexities': defaultdict(int)
        }
        
    def add_snippet(self, snippet: CodeSnippet):
        """Add a code snippet to the database"""
        if snippet.embedding is None:
            raise ValueError("Snippet must have an embedding")
        
        # Add to FAISS index
        embedding = snippet.embedding.astype('float32')
        embedding = embedding.reshape(1, -1)
        faiss.normalize_L2(embedding)
        self.index.add(embedding)
        
        # Store snippet and mappings
        snippet_id = snippet.id or self._generate_id(snippet.code)
        self.snippets[snippet_id] = snippet
        self.id_to_idx[snippet_id] = self.current_idx
        self.idx_to_id[self.current_idx] = snippet_id
        self.current_idx += 1
        
        # Update inverted index
        tokens = self._tokenize(snippet.code + " " + snippet.description)
        for token in tokens:
            self.inverted_index[token.lower()].add(snippet_id)
        
        # Update statistics
        self.stats['total_snippets'] += 1
        self.stats['languages'][snippet.language] += 1
        self.stats['complexities'][snippet.complexity] += 1
        
    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        filters: Optional[Dict] = None
    ) -> List[Tuple[CodeSnippet, float]]:
        """Search for similar code snippets"""
        # Normalize query embedding
        query_embedding = query_embedding.astype('float32')
        query_embedding = query_embedding.reshape(1, -1)
        faiss.normalize_L2(query_embedding)
        
        # Search in FAISS index (get more than k for filtering)
        search_k = min(k * 3, self.index.ntotal)
        distances, indices = self.index.search(query_embedding, search_k)
        
        # Collect results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:  # Invalid index
                continue
            
            snippet_id = self.idx_to_id[idx]
            snippet = self.snippets[snippet_id]
            
            # Apply filters if provided
            if filters:
                if 'language' in filters and snippet.language != filters['language']:
                    continue
                if 'min_complexity' in filters and snippet.complexity < filters['min_complexity']:
                    continue
                if 'max_complexity' in filters and snippet.complexity > filters['max_complexity']:
                    continue
                if 'tags' in filters:
                    if not any(tag in snippet.tags for tag in filters['tags']):
                        continue
            
            results.append((snippet, float(dist)))
            
            if len(results) >= k:
                break
        
        return results
    
    def keyword_search(self, keywords: List[str], k: int = 10) -> List[CodeSnippet]:
        """Search using keyword matching"""
        # Find snippet IDs that match keywords
        matching_ids = set()
        for keyword in keywords:
            keyword_lower = keyword.lower()
            if keyword_lower in self.inverted_index:
                matching_ids.update(self.inverted_index[keyword_lower])
        
        # Score snippets by number of matching keywords
        scores = defaultdict(int)
        for snippet_id in matching_ids:
            snippet = self.snippets[snippet_id]
            snippet_text = (snippet.code + " " + snippet.description).lower()
            for keyword in keywords:
                scores[snippet_id] += snippet_text.count(keyword.lower())
        
        # Sort by score and return top k
        sorted_snippets = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:k]
        
        return [self.snippets[sid] for sid, _ in sorted_snippets]
    
    def hybrid_search(
        self,
        query_embedding: np.ndarray,
        keywords: List[str],
        k: int = 10,
        alpha: float = 0.5
    ) -> List[Tuple[CodeSnippet, float]]:
        """Hybrid search combining vector and keyword search"""
        # Vector search
        vector_results = self.search(query_embedding, k=k*2)
        vector_scores = {snippet.id: score for snippet, score in vector_results}
        
        # Keyword search
        keyword_results = self.keyword_search(keywords, k=k*2)
        keyword_scores = {snippet.id: i for i, snippet in enumerate(keyword_results, 1)}
        
        # Combine scores
        combined_scores = {}
        all_snippet_ids = set(vector_scores.keys()) | set(keyword_scores.keys())
        
        for snippet_id in all_snippet_ids:
            vector_score = vector_scores.get(snippet_id, 0)
            keyword_score = 1.0 / keyword_scores.get(snippet_id, len(keyword_results) + 1)
            combined_scores[snippet_id] = (
                alpha * vector_score + (1 - alpha) * keyword_score
            )
        
        # Sort by combined score
        sorted_results = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:k]
        
        return [(self.snippets[sid], score) for sid, score in sorted_results]
    
    def _generate_id(self, content: str) -> str:
        """Generate unique ID for content"""
        return hashlib.md5(content.encode()).hexdigest()
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for inverted index"""
        import re
        # Split on non-alphanumeric characters
        tokens = re.findall(r'\w+', text)
        return tokens
    
    def save(self, path: str):
        """Save database to disk"""
        # Save FAISS index
        faiss.write_index(self.index, f"{path}.index")
        
        # Save metadata
        metadata = {
            'snippets': self.snippets,
            'id_to_idx': self.id_to_idx,
            'idx_to_id': self.idx_to_id,
            'current_idx': self.current_idx,
            'inverted_index': dict(self.inverted_index),
            'stats': self.stats
        }
        
        with open(f"{path}.metadata", 'wb') as f:
            pickle.dump(metadata, f)
    
    def load(self, path: str):
        """Load database from disk"""
        # Load FAISS index
        self.index = faiss.read_index(f"{path}.index")
        
        # Load metadata
        with open(f"{path}.metadata", 'rb') as f:
            metadata = pickle.load(f)
        
        self.snippets = metadata['snippets']
        self.id_to_idx = metadata['id_to_idx']
        self.idx_to_id = metadata['idx_to_id']
        self.current_idx = metadata['current_idx']
        self.inverted_index = defaultdict(set, metadata['inverted_index'])
        self.stats = metadata['stats']
    
    def get_statistics(self) -> Dict:
        """Get database statistics"""
        return {
            'total_snippets': self.stats['total_snippets'],
            'languages': dict(self.stats['languages']),
            'complexities': dict(self.stats['complexities']),
            'index_type': self.index_type,
            'dimension': self.dimension
        }


class CrossAttentionRetrieval(nn.Module):
    """Cross-attention mechanism for integrating retrieved context"""
    def __init__(
        self,
        hidden_dim: int = 8192,
        num_heads: int = 64,
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Cross-attention layer
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Gating mechanism
        self.gate_linear = nn.Linear(hidden_dim * 2, hidden_dim)
        self.gate_activation = nn.Sigmoid()
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(
        self,
        query_hidden: torch.Tensor,
        retrieved_hidden: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply cross-attention with gating"""
        # Cross-attention
        attended, attention_weights = self.cross_attention(
            query_hidden,
            retrieved_hidden,
            retrieved_hidden,
            key_padding_mask=attention_mask
        )
        
        # Compute gate
        gate_input = torch.cat([query_hidden, retrieved_hidden.mean(dim=1, keepdim=True).expand_as(query_hidden)], dim=-1)
        gate = self.gate_activation(self.gate_linear(gate_input))
        
        # Apply gating
        output = gate * query_hidden + (1 - gate) * attended
        
        # Layer normalization
        output = self.layer_norm(output)
        
        return output, attention_weights


class RetrievalAugmentedGeneration:
    """Main RAG system coordinating retrieval and generation"""
    def __init__(
        self,
        vector_db: FAISSVectorDatabase,
        retriever: DensePassageRetriever,
        cross_attention: CrossAttentionRetrieval,
        top_k: int = 10,
        rerank: bool = True
    ):
        self.vector_db = vector_db
        self.retriever = retriever
        self.cross_attention = cross_attention
        self.top_k = top_k
        self.rerank = rerank
        
    def retrieve_and_augment(
        self,
        query: str,
        query_hidden: torch.Tensor,
        keywords: Optional[List[str]] = None
    ) -> Tuple[torch.Tensor, List[CodeSnippet]]:
        """Retrieve relevant code and augment query representation"""
        
        # Encode query
        query_embedding = self.retriever.encode_query(query)
        
        # Retrieve from database
        if keywords:
            results = self.vector_db.hybrid_search(
                query_embedding.detach().cpu().numpy(),
                keywords,
                k=self.top_k
            )
        else:
            results = self.vector_db.search(
                query_embedding.detach().cpu().numpy(),
                k=self.top_k
            )
        
        # Rerank if enabled
        if self.rerank and len(results) > 1:
            results = self._rerank_results(query, results)
        
        # Extract retrieved snippets
        retrieved_snippets = [snippet for snippet, _ in results]
        
        # Encode retrieved snippets
        retrieved_texts = [s.code + "\n" + s.description for s in retrieved_snippets]
        retrieved_embeddings = torch.stack([
            self.retriever.encode_document(text) 
            for text in retrieved_texts
        ])
        
        # Apply cross-attention
        augmented_hidden, _ = self.cross_attention(
            query_hidden.unsqueeze(0),
            retrieved_embeddings.unsqueeze(0)
        )
        
        return augmented_hidden.squeeze(0), retrieved_snippets
    
    def _rerank_results(
        self,
        query: str,
        results: List[Tuple[CodeSnippet, float]]
    ) -> List[Tuple[CodeSnippet, float]]:
        """Rerank results using more sophisticated scoring"""
        reranked = []
        query_tokens = set(query.lower().split())
        
        for snippet, base_score in results:
            # Additional scoring factors
            relevance_score = base_score
            
            # Boost for exact token matches
            snippet_tokens = set((snippet.code + snippet.description).lower().split())
            token_overlap = len(query_tokens & snippet_tokens) / len(query_tokens)
            relevance_score += 0.1 * token_overlap
            
            # Boost for matching tags
            if hasattr(snippet, 'tags'):
                tag_match = any(tag.lower() in query.lower() for tag in snippet.tags)
                if tag_match:
                    relevance_score += 0.05
            
            reranked.append((snippet, relevance_score))
        
        # Sort by new scores
        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked