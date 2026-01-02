"""
Vocabulary Bridge: Maps between chess tokens and Llama embedding space.

This module implements the hybrid tokenization approach:
- Chess moves are tokenized using our custom 287-token vocabulary
- Embeddings are projected to/from Llama's hidden space
- Preserves semantic meaning while leveraging pretrained weights
"""

from typing import Optional

import torch
import torch.nn as nn


class VocabBridge(nn.Module):
    """
    Bridge between chess vocabulary and Llama embedding space.

    Architecture:
        Chess Tokens (287) → Embedding (hidden_size) → Llama
        Llama → Hidden States → Linear (chess_vocab_size) → Chess Logits

    This approach:
    - Preserves semantic chess token meanings
    - Enables smaller vocab (287 vs 128K)
    - Faster inference
    - More interpretable predictions
    """

    def __init__(
        self,
        chess_vocab_size: int,
        hidden_size: int,
        pad_token_id: Optional[int] = None,
        dropout: float = 0.1,
    ):
        """
        Initialize vocabulary bridge.

        Args:
            chess_vocab_size: Size of chess vocabulary (typically 287)
            hidden_size: Llama hidden dimension (1024 for Llama-3.2-1B)
            pad_token_id: Optional padding token ID for masking
            dropout: Dropout rate for embeddings
        """
        super().__init__()

        self.chess_vocab_size = chess_vocab_size
        self.hidden_size = hidden_size
        self.pad_token_id = pad_token_id

        # Chess tokens → Llama embedding space
        self.chess_to_llama = nn.Embedding(
            num_embeddings=chess_vocab_size,
            embedding_dim=hidden_size,
            padding_idx=pad_token_id,
        )

        # Llama hidden states → Chess logits
        self.llama_to_chess = nn.Linear(
            in_features=hidden_size,
            out_features=chess_vocab_size,
            bias=False,  # Following modern LM design
        )

        # Optional dropout for regularization
        self.dropout = nn.Dropout(dropout)

        # Layer norm for stable training
        self.layer_norm = nn.LayerNorm(hidden_size)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """
        Initialize weights using Xavier/He initialization.

        For embeddings, use normal distribution with small std.
        For linear layers, use Xavier uniform (good for linear transformations).
        """
        # Embedding initialization (similar to Llama)
        nn.init.normal_(self.chess_to_llama.weight, mean=0.0, std=0.02)
        if self.pad_token_id is not None:
            # Zero out padding token embedding
            with torch.no_grad():
                self.chess_to_llama.weight[self.pad_token_id].fill_(0.0)

        # Linear layer initialization (Xavier uniform)
        nn.init.xavier_uniform_(self.llama_to_chess.weight)

    def embed_chess_moves(
        self, chess_token_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert chess token IDs to Llama-compatible embeddings.

        Args:
            chess_token_ids: Tensor of shape (batch_size, seq_len) with chess token IDs

        Returns:
            embeddings: Tensor of shape (batch_size, seq_len, hidden_size)
        """
        # Get embeddings
        embeddings = self.chess_to_llama(chess_token_ids)

        # Apply layer norm for stability
        embeddings = self.layer_norm(embeddings)

        # Apply dropout during training
        embeddings = self.dropout(embeddings)

        return embeddings

    def project_to_chess_logits(
        self, llama_hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """
        Project Llama hidden states to chess vocabulary logits.

        Args:
            llama_hidden_states: Tensor of shape (batch_size, seq_len, hidden_size)

        Returns:
            logits: Tensor of shape (batch_size, seq_len, chess_vocab_size)
        """
        # Project to chess vocabulary
        logits = self.llama_to_chess(llama_hidden_states)

        return logits

    def forward(
        self,
        chess_token_ids: torch.Tensor,
        llama_model: nn.Module,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Full forward pass: chess tokens → embeddings → Llama → chess logits.

        Args:
            chess_token_ids: Input chess token IDs (batch_size, seq_len)
            llama_model: The Llama model to pass embeddings through
            attention_mask: Optional attention mask (batch_size, seq_len)

        Returns:
            logits: Chess vocabulary logits (batch_size, seq_len, chess_vocab_size)
        """
        # Step 1: Convert chess tokens to Llama embeddings
        embeddings = self.embed_chess_moves(chess_token_ids)

        # Step 2: Pass through Llama
        # Note: We pass inputs_embeds instead of input_ids
        llama_outputs = llama_model(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
            return_dict=True,
        )

        # Step 3: Get hidden states
        hidden_states = llama_outputs.last_hidden_state

        # Step 4: Project back to chess vocabulary
        logits = self.project_to_chess_logits(hidden_states)

        return logits

    def get_embedding_weight(self) -> torch.Tensor:
        """
        Get the chess embedding weight matrix.

        Useful for:
        - Initialization of the output projection
        - Analysis of learned embeddings
        - Weight tying (if desired)

        Returns:
            weight: Tensor of shape (chess_vocab_size, hidden_size)
        """
        return self.chess_to_llama.weight

    def tie_weights(self):
        """
        Tie input and output embeddings (optional).

        This can reduce parameters and sometimes improve performance,
        but requires careful consideration for our use case since:
        - Input: chess token → hidden space
        - Output: hidden space → chess token probabilities

        Weight tying means: llama_to_chess.weight = chess_to_llama.weight.T
        """
        self.llama_to_chess.weight = nn.Parameter(
            self.chess_to_llama.weight.transpose(0, 1)
        )

    def save_pretrained(self, save_directory: str):
        """
        Save vocabulary bridge weights.

        Args:
            save_directory: Directory to save weights
        """
        import os

        os.makedirs(save_directory, exist_ok=True)
        save_path = os.path.join(save_directory, "vocab_bridge.pt")

        torch.save(
            {
                "chess_to_llama": self.chess_to_llama.state_dict(),
                "llama_to_chess": self.llama_to_chess.state_dict(),
                "layer_norm": self.layer_norm.state_dict(),
                "config": {
                    "chess_vocab_size": self.chess_vocab_size,
                    "hidden_size": self.hidden_size,
                    "pad_token_id": self.pad_token_id,
                },
            },
            save_path,
        )

        print(f"VocabBridge saved to {save_path}")

    @classmethod
    def from_pretrained(cls, load_directory: str) -> "VocabBridge":
        """
        Load vocabulary bridge weights.

        Args:
            load_directory: Directory to load weights from

        Returns:
            vocab_bridge: Loaded VocabBridge instance
        """
        import os

        load_path = os.path.join(load_directory, "vocab_bridge.pt")

        if not os.path.exists(load_path):
            raise FileNotFoundError(f"VocabBridge weights not found at {load_path}")

        checkpoint = torch.load(load_path, map_location="cpu")

        # Create instance
        config = checkpoint["config"]
        vocab_bridge = cls(
            chess_vocab_size=config["chess_vocab_size"],
            hidden_size=config["hidden_size"],
            pad_token_id=config["pad_token_id"],
        )

        # Load weights
        vocab_bridge.chess_to_llama.load_state_dict(checkpoint["chess_to_llama"])
        vocab_bridge.llama_to_chess.load_state_dict(checkpoint["llama_to_chess"])
        vocab_bridge.layer_norm.load_state_dict(checkpoint["layer_norm"])

        print(f"VocabBridge loaded from {load_path}")

        return vocab_bridge

    def __repr__(self) -> str:
        return (
            f"VocabBridge(\n"
            f"  chess_vocab_size={self.chess_vocab_size},\n"
            f"  hidden_size={self.hidden_size},\n"
            f"  parameters={sum(p.numel() for p in self.parameters()):,}\n"
            f")"
        )
