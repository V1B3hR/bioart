"""
Non-functional DNA sequence embedding for BioArt generation.

Creates embeddings from DNA sequences using k-mer tokenization.
These embeddings are purely for computational/artistic purposes.
No biological function prediction or sequence optimization.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from collections import Counter
import itertools
from pathlib import Path
import json

try:
    import torch
    import torch.nn as nn
    from torch.nn import TransformerEncoder, TransformerEncoderLayer
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. DNA embedding will use simpler methods.")


class KmerTokenizer:
    """Tokenizes DNA sequences into k-mers for embedding."""
    
    def __init__(self, k: int = 6, max_vocab_size: int = 4096):
        """
        Initialize k-mer tokenizer.
        
        Args:
            k: K-mer size
            max_vocab_size: Maximum vocabulary size
        """
        self.k = k
        self.max_vocab_size = max_vocab_size
        self.nucleotides = ['A', 'T', 'G', 'C']
        
        # Special tokens
        self.pad_token = '<PAD>'
        self.unk_token = '<UNK>'
        self.special_tokens = [self.pad_token, self.unk_token]
        
        # Vocabulary mappings
        self.token_to_id = {}
        self.id_to_token = {}
        self._build_initial_vocab()
        
    def _build_initial_vocab(self):
        """Build initial vocabulary with special tokens."""
        # Add special tokens
        for i, token in enumerate(self.special_tokens):
            self.token_to_id[token] = i
            self.id_to_token[i] = token
        
        # Add all possible k-mers (if small enough)
        if 4**self.k <= self.max_vocab_size - len(self.special_tokens):
            all_kmers = [''.join(kmer) for kmer in itertools.product(self.nucleotides, repeat=self.k)]
            for i, kmer in enumerate(all_kmers):
                token_id = len(self.special_tokens) + i
                self.token_to_id[kmer] = token_id
                self.id_to_token[token_id] = kmer
    
    def extract_kmers(self, sequence: str) -> List[str]:
        """Extract k-mers from sequence."""
        sequence = sequence.upper().replace('U', 'T')  # Normalize to DNA
        kmers = []
        
        for i in range(len(sequence) - self.k + 1):
            kmer = sequence[i:i+self.k]
            # Only include k-mers with valid nucleotides
            if all(base in self.nucleotides for base in kmer):
                kmers.append(kmer)
        
        return kmers
    
    def build_vocab_from_sequences(self, sequences: List[str]) -> None:
        """Build vocabulary from a collection of sequences."""
        if len(self.token_to_id) > len(self.special_tokens):
            return  # Vocabulary already built
        
        # Count all k-mers
        kmer_counts = Counter()
        for sequence in sequences:
            kmers = self.extract_kmers(sequence)
            kmer_counts.update(kmers)
        
        # Add most frequent k-mers to vocabulary
        available_slots = self.max_vocab_size - len(self.special_tokens)
        most_common = kmer_counts.most_common(available_slots)
        
        for i, (kmer, count) in enumerate(most_common):
            token_id = len(self.special_tokens) + i
            self.token_to_id[kmer] = token_id
            self.id_to_token[token_id] = kmer
    
    def tokenize(self, sequence: str) -> List[int]:
        """Convert sequence to token IDs."""
        kmers = self.extract_kmers(sequence)
        token_ids = []
        
        for kmer in kmers:
            token_id = self.token_to_id.get(kmer, self.token_to_id[self.unk_token])
            token_ids.append(token_id)
        
        return token_ids
    
    def pad_sequence(self, token_ids: List[int], max_length: int) -> List[int]:
        """Pad sequence to max length."""
        if len(token_ids) >= max_length:
            return token_ids[:max_length]
        else:
            padding = [self.token_to_id[self.pad_token]] * (max_length - len(token_ids))
            return token_ids + padding
    
    def save_vocab(self, path: Union[str, Path]):
        """Save vocabulary to file."""
        vocab_data = {
            'k': self.k,
            'max_vocab_size': self.max_vocab_size,
            'token_to_id': self.token_to_id,
            'id_to_token': self.id_to_token
        }
        
        with open(path, 'w') as f:
            json.dump(vocab_data, f, indent=2)
    
    def load_vocab(self, path: Union[str, Path]):
        """Load vocabulary from file."""
        with open(path, 'r') as f:
            vocab_data = json.load(f)
        
        self.k = vocab_data['k']
        self.max_vocab_size = vocab_data['max_vocab_size']
        self.token_to_id = vocab_data['token_to_id']
        self.id_to_token = {int(k): v for k, v in vocab_data['id_to_token'].items()}


class SimpleDNAEmbedding:
    """Simple DNA embedding without PyTorch."""
    
    def __init__(self, tokenizer: KmerTokenizer, embedding_dim: int = 256):
        """
        Initialize simple DNA embedding.
        
        Args:
            tokenizer: K-mer tokenizer
            embedding_dim: Embedding dimension
        """
        self.tokenizer = tokenizer
        self.embedding_dim = embedding_dim
        
        # Random embeddings for each token
        vocab_size = len(tokenizer.token_to_id)
        self.embeddings = np.random.normal(0, 0.1, (vocab_size, embedding_dim))
    
    def encode(self, sequence: str) -> np.ndarray:
        """Encode sequence to embedding."""
        token_ids = self.tokenizer.tokenize(sequence)
        
        if not token_ids:
            return np.zeros(self.embedding_dim)
        
        # Average embeddings
        token_embeddings = self.embeddings[token_ids]
        return np.mean(token_embeddings, axis=0)
    
    def encode_batch(self, sequences: List[str]) -> np.ndarray:
        """Encode batch of sequences."""
        embeddings = []
        for sequence in sequences:
            embedding = self.encode(sequence)
            embeddings.append(embedding)
        
        return np.array(embeddings)


class TransformerDNAEmbedding(nn.Module):
    """Transformer-based DNA embedding (requires PyTorch)."""
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        hidden_dim: int = 512,
        max_length: int = 1024,
        dropout: float = 0.1
    ):
        """
        Initialize transformer DNA embedding.
        
        Args:
            vocab_size: Vocabulary size
            embedding_dim: Embedding dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            hidden_dim: Hidden dimension
            max_length: Maximum sequence length
            dropout: Dropout rate
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        
        # Token and position embeddings
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(max_length, embedding_dim)
        
        # Transformer encoder
        encoder_layer = TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers)
        
        # Output projection
        self.output_projection = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, token_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            token_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Sequence embeddings [batch_size, embedding_dim]
        """
        batch_size, seq_len = token_ids.shape
        
        # Token embeddings
        token_emb = self.token_embedding(token_ids)
        
        # Position embeddings
        positions = torch.arange(seq_len, device=token_ids.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.position_embedding(positions)
        
        # Combined embeddings
        embeddings = token_emb + pos_emb
        embeddings = self.dropout(embeddings)
        
        # Create attention mask for padding
        if attention_mask is None:
            attention_mask = token_ids != 0  # Assume 0 is padding token
        
        # Transformer encoding
        # Convert attention mask for transformer (True = attend, False = ignore)
        src_key_padding_mask = ~attention_mask
        
        encoded = self.transformer(embeddings, src_key_padding_mask=src_key_padding_mask)
        
        # Pool to single vector (mean pooling over non-padded positions)
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).expand_as(encoded).float()
            pooled = (encoded * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
        else:
            pooled = encoded.mean(dim=1)
        
        # Final projection
        output = self.output_projection(pooled)
        
        return output


class DNAEmbedding:
    """Main DNA embedding class that handles both simple and transformer approaches."""
    
    def __init__(
        self,
        k: int = 6,
        embedding_dim: int = 256,
        max_vocab_size: int = 4096,
        use_transformer: bool = None,
        **transformer_kwargs
    ):
        """
        Initialize DNA embedding.
        
        Args:
            k: K-mer size
            embedding_dim: Embedding dimension
            max_vocab_size: Maximum vocabulary size
            use_transformer: Whether to use transformer (auto-detect if None)
            **transformer_kwargs: Additional arguments for transformer
        """
        self.k = k
        self.embedding_dim = embedding_dim
        
        # Initialize tokenizer
        self.tokenizer = KmerTokenizer(k=k, max_vocab_size=max_vocab_size)
        
        # Choose embedding method
        if use_transformer is None:
            use_transformer = TORCH_AVAILABLE
        
        self.use_transformer = use_transformer
        
        if use_transformer and TORCH_AVAILABLE:
            self.model = TransformerDNAEmbedding(
                vocab_size=max_vocab_size,
                embedding_dim=embedding_dim,
                **transformer_kwargs
            )
        else:
            self.model = SimpleDNAEmbedding(self.tokenizer, embedding_dim)
    
    def fit(self, sequences: List[str]):
        """Fit embedding on sequences (build vocabulary)."""
        # Build vocabulary
        self.tokenizer.build_vocab_from_sequences(sequences)
        
        # Update model if using simple embedding
        if not self.use_transformer:
            vocab_size = len(self.tokenizer.token_to_id)
            self.model = SimpleDNAEmbedding(self.tokenizer, self.embedding_dim)
    
    def encode(self, sequences: Union[str, List[str]]) -> np.ndarray:
        """
        Encode sequences to embeddings.
        
        Args:
            sequences: Single sequence or list of sequences
            
        Returns:
            Embeddings array
        """
        if isinstance(sequences, str):
            sequences = [sequences]
        
        if self.use_transformer and TORCH_AVAILABLE:
            return self._encode_transformer(sequences)
        else:
            return self.model.encode_batch(sequences)
    
    def _encode_transformer(self, sequences: List[str]) -> np.ndarray:
        """Encode using transformer model."""
        self.model.eval()
        
        embeddings = []
        with torch.no_grad():
            for sequence in sequences:
                token_ids = self.tokenizer.tokenize(sequence)
                token_ids = self.tokenizer.pad_sequence(token_ids, self.model.max_length)
                
                token_tensor = torch.tensor([token_ids])
                embedding = self.model(token_tensor)
                embeddings.append(embedding.numpy()[0])
        
        return np.array(embeddings)
    
    def save(self, path: Union[str, Path]):
        """Save embedding model."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save tokenizer
        self.tokenizer.save_vocab(path / 'vocab.json')
        
        # Save model
        if self.use_transformer and TORCH_AVAILABLE:
            torch.save(self.model.state_dict(), path / 'model.pth')
        else:
            np.save(path / 'embeddings.npy', self.model.embeddings)
        
        # Save config
        config = {
            'k': self.k,
            'embedding_dim': self.embedding_dim,
            'use_transformer': self.use_transformer
        }
        
        with open(path / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)
    
    def load(self, path: Union[str, Path]):
        """Load embedding model."""
        path = Path(path)
        
        # Load config
        with open(path / 'config.json', 'r') as f:
            config = json.load(f)
        
        # Load tokenizer
        self.tokenizer.load_vocab(path / 'vocab.json')
        
        # Load model
        if config['use_transformer'] and TORCH_AVAILABLE:
            self.model.load_state_dict(torch.load(path / 'model.pth'))
        else:
            embeddings = np.load(path / 'embeddings.npy')
            self.model = SimpleDNAEmbedding(self.tokenizer, config['embedding_dim'])
            self.model.embeddings = embeddings