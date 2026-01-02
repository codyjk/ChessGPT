"""
Llama Chess Transformer: Llama 3.2 1B fine-tuned for chess.

Integrates:
- Pretrained Llama 3.2 1B base model
- LoRA adapters for parameter-efficient fine-tuning
- VocabBridge for chess tokenization
- Custom chess-specific forward pass
"""

from typing import Optional, Dict, Any, Tuple

import torch
import torch.nn as nn

try:
    from transformers import AutoModelForCausalLM, AutoConfig
    from peft import LoraConfig, get_peft_model, PeftModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    AutoModelForCausalLM = None
    AutoConfig = None
    LoraConfig = None
    get_peft_model = None
    PeftModel = None

from .vocab_bridge import VocabBridge


class LlamaChessTransformer(nn.Module):
    """
    Chess-playing language model based on Llama 3.2 1B.

    Architecture:
        Chess Tokens → VocabBridge → Llama (with LoRA) → VocabBridge → Chess Logits

    Key features:
    - Preserves Llama's pretrained knowledge
    - Parameter-efficient fine-tuning via LoRA
    - Chess-specific vocabulary (287 tokens)
    - Compatible with HuggingFace Trainer
    """

    def __init__(
        self,
        chess_vocab_size: int,
        base_model_name: str = "meta-llama/Llama-3.2-1B",
        use_lora: bool = True,
        lora_config: Optional[Dict[str, Any]] = None,
        pad_token_id: Optional[int] = None,
        device_map: str = "auto",
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        """
        Initialize Llama Chess Transformer.

        Args:
            chess_vocab_size: Size of chess vocabulary (typically 287)
            base_model_name: HuggingFace model identifier
            use_lora: Whether to use LoRA adapters
            lora_config: LoRA configuration dict (r, alpha, target_modules, dropout)
            pad_token_id: Padding token ID for chess vocabulary
            device_map: Device placement strategy ("auto", "cpu", etc.)
            torch_dtype: Model dtype (bfloat16 recommended for modern GPUs)
        """
        super().__init__()

        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers and peft are required. "
                "Install with: pip install transformers peft"
            )

        self.chess_vocab_size = chess_vocab_size
        self.base_model_name = base_model_name
        self.use_lora = use_lora
        self.pad_token_id = pad_token_id

        # Load Llama base model
        print(f"Loading Llama model: {base_model_name}")
        self.llama_config = AutoConfig.from_pretrained(base_model_name)
        self.hidden_size = self.llama_config.hidden_size

        self.llama_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            config=self.llama_config,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=True,  # Required for some models
        )

        # Apply LoRA if requested
        if use_lora:
            self._apply_lora(lora_config)
        else:
            # Freeze base model if not using LoRA
            print("Freezing Llama base model (no LoRA)")
            for param in self.llama_model.parameters():
                param.requires_grad = False

        # Create vocabulary bridge
        print(f"Creating VocabBridge (chess_vocab={chess_vocab_size}, hidden={self.hidden_size})")
        self.vocab_bridge = VocabBridge(
            chess_vocab_size=chess_vocab_size,
            hidden_size=self.hidden_size,
            pad_token_id=pad_token_id,
        )

        # Cache for generation
        self._past_key_values = None

    def _apply_lora(self, lora_config: Optional[Dict[str, Any]] = None):
        """
        Apply LoRA adapters to Llama model.

        Args:
            lora_config: Configuration dict with keys:
                - r: LoRA rank (default: 16)
                - lora_alpha: Scaling factor (default: 32)
                - target_modules: Modules to adapt (default: q_proj, v_proj, k_proj, o_proj)
                - lora_dropout: Dropout rate (default: 0.05)
        """
        # Default LoRA config
        default_config = {
            "r": 16,
            "lora_alpha": 32,
            "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
            "lora_dropout": 0.05,
            "bias": "none",
            "task_type": "CAUSAL_LM",
        }

        # Merge with user config
        if lora_config is not None:
            default_config.update(lora_config)

        print(f"Applying LoRA with config: {default_config}")

        # Create LoRA config
        peft_config = LoraConfig(**default_config)

        # Apply to model
        self.llama_model = get_peft_model(self.llama_model, peft_config)

        # Print trainable parameters
        self.llama_model.print_trainable_parameters()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.

        Args:
            input_ids: Chess token IDs (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
            labels: Target labels for training (batch_size, seq_len)
            return_dict: Whether to return a dict (for HF Trainer compatibility)

        Returns:
            outputs: Dict with keys:
                - logits: Chess vocabulary logits (batch_size, seq_len, chess_vocab_size)
                - loss: Optional loss if labels provided
                - hidden_states: Optional hidden states
        """
        # Step 1: Convert chess tokens to Llama embeddings
        embeddings = self.vocab_bridge.embed_chess_moves(input_ids)

        # Step 2: Pass through Llama
        llama_outputs = self.llama_model(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        # Step 3: Get hidden states
        hidden_states = llama_outputs.hidden_states[-1]  # Last layer

        # Step 4: Project to chess vocabulary
        logits = self.vocab_bridge.project_to_chess_logits(hidden_states)

        # Prepare output
        outputs = {"logits": logits}

        # Compute loss if labels provided (for training)
        if labels is not None:
            # Shift for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Flatten for loss calculation
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fn(
                shift_logits.view(-1, self.chess_vocab_size),
                shift_labels.view(-1),
            )
            outputs["loss"] = loss

        if return_dict:
            return outputs
        else:
            return (logits,)

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Generate chess moves autoregressively.

        Args:
            input_ids: Input chess token IDs (batch_size, seq_len)
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Keep only top k tokens for sampling
            top_p: Nucleus sampling threshold
            attention_mask: Optional attention mask

        Returns:
            generated_ids: Generated token IDs (batch_size, seq_len + max_new_tokens)
        """
        self.eval()
        batch_size = input_ids.size(0)
        device = input_ids.device

        # Initialize generation
        generated = input_ids.clone()

        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Get logits for next token
                outputs = self.forward(
                    input_ids=generated,
                    attention_mask=attention_mask,
                    return_dict=True,
                )

                # Get logits for last position
                next_token_logits = outputs["logits"][:, -1, :]

                # Apply temperature
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature

                # Apply top-k filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(
                        next_token_logits, top_k
                    )[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float("-inf")

                # Apply top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(
                        next_token_logits, descending=True
                    )
                    cumulative_probs = torch.cumsum(
                        torch.softmax(sorted_logits, dim=-1), dim=-1
                    )

                    # Remove tokens with cumulative probability above threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                        ..., :-1
                    ].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    next_token_logits[indices_to_remove] = float("-inf")

                # Sample from distribution
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                # Append to sequence
                generated = torch.cat([generated, next_token], dim=1)

                # Update attention mask if provided
                if attention_mask is not None:
                    attention_mask = torch.cat(
                        [
                            attention_mask,
                            torch.ones(batch_size, 1, device=device, dtype=attention_mask.dtype),
                        ],
                        dim=1,
                    )

        return generated

    def get_trainable_parameters(self) -> int:
        """
        Count trainable parameters.

        Returns:
            num_params: Number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_total_parameters(self) -> int:
        """
        Count total parameters.

        Returns:
            num_params: Total number of parameters
        """
        return sum(p.numel() for p in self.parameters())

    def save_pretrained(self, save_directory: str):
        """
        Save model weights and config.

        Args:
            save_directory: Directory to save to
        """
        import os

        os.makedirs(save_directory, exist_ok=True)

        # Save Llama (LoRA adapters if applicable)
        if self.use_lora:
            self.llama_model.save_pretrained(save_directory)
        else:
            # Save full model
            self.llama_model.save_pretrained(save_directory)

        # Save vocabulary bridge
        self.vocab_bridge.save_pretrained(save_directory)

        # Save config
        config = {
            "chess_vocab_size": self.chess_vocab_size,
            "base_model_name": self.base_model_name,
            "use_lora": self.use_lora,
            "pad_token_id": self.pad_token_id,
        }

        import json
        with open(os.path.join(save_directory, "chess_config.json"), "w") as f:
            json.dump(config, f, indent=2)

        print(f"Model saved to {save_directory}")

    @classmethod
    def from_pretrained(
        cls,
        load_directory: str,
        device_map: str = "auto",
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> "LlamaChessTransformer":
        """
        Load pretrained model.

        Args:
            load_directory: Directory to load from
            device_map: Device placement strategy
            torch_dtype: Model dtype

        Returns:
            model: Loaded LlamaChessTransformer
        """
        import os
        import json

        # Load config
        config_path = os.path.join(load_directory, "chess_config.json")
        with open(config_path, "r") as f:
            config = json.load(f)

        # Create model instance
        model = cls(
            chess_vocab_size=config["chess_vocab_size"],
            base_model_name=config["base_model_name"],
            use_lora=config["use_lora"],
            pad_token_id=config["pad_token_id"],
            device_map=device_map,
            torch_dtype=torch_dtype,
        )

        # Load Llama weights (LoRA adapters if applicable)
        if config["use_lora"]:
            model.llama_model = PeftModel.from_pretrained(
                model.llama_model,
                load_directory,
                device_map=device_map,
            )

        # Load vocabulary bridge
        model.vocab_bridge = VocabBridge.from_pretrained(load_directory)

        print(f"Model loaded from {load_directory}")

        return model

    def __repr__(self) -> str:
        trainable = self.get_trainable_parameters()
        total = self.get_total_parameters()
        percentage = (trainable / total) * 100

        return (
            f"LlamaChessTransformer(\n"
            f"  base_model={self.base_model_name},\n"
            f"  chess_vocab_size={self.chess_vocab_size},\n"
            f"  use_lora={self.use_lora},\n"
            f"  total_parameters={total:,},\n"
            f"  trainable_parameters={trainable:,} ({percentage:.2f}%),\n"
            f")"
        )
