#!/usr/bin/env python3
"""
Real LoRA Training for FIRM Phase 3
Implements genuine Low-Rank Adaptation training for bias circuit mitigation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
from pathlib import Path
import logging
from dataclasses import dataclass
from tqdm import tqdm
import json
from transformers import get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType
import math


@dataclass
class LoRATrainingConfig:
    """Configuration for real LoRA training."""
    r: int = 16  # LoRA rank
    alpha: int = 32  # LoRA alpha parameter
    dropout: float = 0.1  # LoRA dropout
    target_modules: List[str] = None  # Modules to apply LoRA to
    learning_rate: float = 1e-4
    num_epochs: int = 3
    batch_size: int = 4
    warmup_steps: int = 100
    max_length: int = 512
    gradient_accumulation_steps: int = 4
    weight_decay: float = 0.01


@dataclass 
class LoRATrainingResult:
    """Results from real LoRA training."""
    model_path: str
    training_loss: List[float]
    validation_metrics: Dict[str, float]
    bias_reduction_scores: Dict[str, float]
    training_metadata: Dict[str, Any]
    lora_weights: Dict[str, torch.Tensor]


class BiasContrastiveDataset(Dataset):
    """Dataset for bias-contrastive training."""
    
    def __init__(self, bias_samples: List[Dict[str, Any]], tokenizer, max_length: int = 512):
        """
        Initialize bias contrastive dataset.
        
        Args:
            bias_samples: List of bias evaluation samples with contrasting pairs
            tokenizer: Model tokenizer
            max_length: Maximum sequence length
        """
        self.bias_samples = bias_samples
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Process samples into contrastive pairs
        self.contrastive_pairs = self._create_contrastive_pairs()
    
    def _create_contrastive_pairs(self) -> List[Dict[str, Any]]:
        """Create contrastive pairs for bias mitigation training."""
        pairs = []
        
        for sample in self.bias_samples:
            text = sample.get('text', '')
            if not text:
                continue
            
            # Create biased and debiased versions
            # This is a simplified example - real implementation would need
            # more sophisticated bias pair generation
            biased_text = text
            
            # Simple debiasing: replace gendered pronouns with neutral ones
            debiased_text = text
            replacements = {
                ' he ': ' they ', ' she ': ' they ',
                ' him ': ' them ', ' her ': ' them ',
                ' his ': ' their ', ' hers ': ' theirs ',
                'He ': 'They ', 'She ': 'They ',
                'His ': 'Their ', 'Her ': 'Their '
            }
            
            for old, new in replacements.items():
                debiased_text = debiased_text.replace(old, new)
            
            # Only add if there's actually a difference
            if biased_text != debiased_text:
                pairs.append({
                    'biased_text': biased_text,
                    'debiased_text': debiased_text,
                    'bias_type': sample.get('bias_type', 'general'),
                    'original_sample': sample
                })
        
        return pairs
    
    def __len__(self):
        return len(self.contrastive_pairs)
    
    def __getitem__(self, idx):
        pair = self.contrastive_pairs[idx]
        
        # Tokenize both versions
        biased_encoding = self.tokenizer(
            pair['biased_text'],
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        debiased_encoding = self.tokenizer(
            pair['debiased_text'],
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'biased_input_ids': biased_encoding['input_ids'].squeeze(),
            'biased_attention_mask': biased_encoding['attention_mask'].squeeze(),
            'debiased_input_ids': debiased_encoding['input_ids'].squeeze(),
            'debiased_attention_mask': debiased_encoding['attention_mask'].squeeze(),
            'bias_type': pair['bias_type']
        }


class RealLoRATrainer:
    """
    Real LoRA trainer for genuine bias mitigation.
    Uses actual contrastive learning and circuit-targeted training.
    """
    
    def __init__(self, model, tokenizer, device: str = "auto"):
        """
        Initialize real LoRA trainer.
        
        Args:
            model: Base model to train
            tokenizer: Model tokenizer
            device: Training device
        """
        self.base_model = model
        self.tokenizer = tokenizer
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logging.getLogger(__name__)
        
        # Training state
        self.lora_model = None
        self.optimizer = None
        self.scheduler = None
        self.training_metrics = []
        
        self.logger.info(f"Initialized RealLoRATrainer on {self.device}")
    
    def setup_lora_model(self, config: LoRATrainingConfig, 
                        target_circuits: Optional[List[Dict[str, Any]]] = None) -> None:
        """
        Setup LoRA model with real circuit targeting.
        
        Args:
            config: LoRA training configuration
            target_circuits: Identified bias circuits to target
        """
        self.logger.info("Setting up LoRA model for real bias mitigation...")
        
        # Determine target modules based on identified circuits
        if target_circuits and config.target_modules is None:
            target_modules = self._get_target_modules_from_circuits(target_circuits)
        else:
            # Use GPT-2 compatible target modules
            target_modules = config.target_modules or ["c_attn", "c_proj", "c_fc"]
        
        # Create LoRA configuration
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.r,
            lora_alpha=config.alpha,
            lora_dropout=config.dropout,
            target_modules=target_modules,
            bias="none"  # Don't train bias parameters
        )
        
        # Apply LoRA to model
        self.lora_model = get_peft_model(self.base_model, lora_config)
        self.lora_model.to(self.device)
        
        # Enable training mode
        self.lora_model.train()
        
        self.logger.info(f"LoRA model setup complete. Target modules: {target_modules}")
        self.logger.info(f"Trainable parameters: {self.lora_model.get_nb_trainable_parameters()}")
    
    def _get_target_modules_from_circuits(self, target_circuits: List[Dict[str, Any]]) -> List[str]:
        """Extract target modules from identified bias circuits."""
        target_modules = set()
        
        for circuit in target_circuits:
            component_type = circuit.get('component_type', '')
            
            if 'attention' in component_type.lower():
                # GPT-2 attention modules
                target_modules.update(["c_attn", "c_proj"])
            elif 'mlp' in component_type.lower():
                # GPT-2 MLP modules
                target_modules.update(["c_fc"])
            else:
                # Default to attention modules
                target_modules.update(["c_attn", "c_proj"])
        
        return list(target_modules)
    
    def train(self, bias_samples: List[Dict[str, Any]], 
             config: LoRATrainingConfig,
             validation_samples: Optional[List[Dict[str, Any]]] = None,
             target_circuits: Optional[List[Dict[str, Any]]] = None) -> LoRATrainingResult:
        """
        Train LoRA model for real bias mitigation.
        
        Args:
            bias_samples: Training samples with bias examples
            config: Training configuration
            validation_samples: Validation samples (optional)
            target_circuits: Identified bias circuits to target
            
        Returns:
            LoRATrainingResult with training outcomes
        """
        self.logger.info(f"Starting real LoRA training with {len(bias_samples)} samples")
        
        # Setup LoRA model
        self.setup_lora_model(config, target_circuits)
        
        # Create dataset and dataloader
        train_dataset = BiasContrastiveDataset(bias_samples, self.tokenizer, config.max_length)
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=config.batch_size, 
            shuffle=True,
            collate_fn=self._collate_fn
        )
        
        # Setup optimizer and scheduler
        self._setup_optimizer_and_scheduler(config, len(train_dataloader))
        
        # Training loop
        training_loss = []
        
        for epoch in range(config.num_epochs):
            self.logger.info(f"Starting epoch {epoch + 1}/{config.num_epochs}")
            
            epoch_loss = self._train_epoch(train_dataloader, config)
            training_loss.append(epoch_loss)
            
            self.logger.info(f"Epoch {epoch + 1} completed. Loss: {epoch_loss:.4f}")
        
        # Evaluate bias reduction
        bias_reduction_scores = self._evaluate_bias_reduction(validation_samples or bias_samples[:10])
        
        # Extract LoRA weights
        lora_weights = self._extract_lora_weights()
        
        # Save model
        model_path = self._save_model(config)
        
        return LoRATrainingResult(
            model_path=model_path,
            training_loss=training_loss,
            validation_metrics={
                'final_loss': training_loss[-1] if training_loss else 0.0,
                'convergence_achieved': len(training_loss) > 1 and abs(training_loss[-1] - training_loss[-2]) < 0.01
            },
            bias_reduction_scores=bias_reduction_scores,
            training_metadata={
                'num_epochs': config.num_epochs,
                'batch_size': config.batch_size,
                'learning_rate': config.learning_rate,
                'lora_rank': config.r,
                'target_modules': self.lora_model.peft_config['default'].target_modules,
                'trainable_params': self.lora_model.get_nb_trainable_parameters()
            },
            lora_weights=lora_weights
        )
    
    def _collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate function for dataloader."""
        return {
            'biased_input_ids': torch.stack([item['biased_input_ids'] for item in batch]),
            'biased_attention_mask': torch.stack([item['biased_attention_mask'] for item in batch]),
            'debiased_input_ids': torch.stack([item['debiased_input_ids'] for item in batch]),
            'debiased_attention_mask': torch.stack([item['debiased_attention_mask'] for item in batch])
        }
    
    def _setup_optimizer_and_scheduler(self, config: LoRATrainingConfig, num_training_steps: int):
        """Setup optimizer and learning rate scheduler."""
        # Only optimize LoRA parameters
        self.optimizer = torch.optim.AdamW(
            self.lora_model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=num_training_steps * config.num_epochs
        )
    
    def _train_epoch(self, dataloader: DataLoader, config: LoRATrainingConfig) -> float:
        """Train for one epoch."""
        total_loss = 0.0
        num_batches = 0
        
        self.lora_model.train()
        
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Training")):
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Compute contrastive loss
            loss = self._compute_contrastive_loss(batch)
            
            # Backward pass
            loss = loss / config.gradient_accumulation_steps
            loss.backward()
            
            if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.lora_model.parameters(), 1.0)
                
                # Optimizer step
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
            
            total_loss += loss.item() * config.gradient_accumulation_steps
            num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def _compute_contrastive_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute contrastive loss for bias mitigation."""
        # Get outputs for biased and debiased versions
        biased_outputs = self.lora_model(
            input_ids=batch['biased_input_ids'],
            attention_mask=batch['biased_attention_mask']
        )
        
        debiased_outputs = self.lora_model(
            input_ids=batch['debiased_input_ids'],
            attention_mask=batch['debiased_attention_mask']
        )
        
        # Extract logits
        biased_logits = biased_outputs.logits
        debiased_logits = debiased_outputs.logits
        
        # Compute contrastive loss: encourage model to produce similar outputs
        # for both biased and debiased versions
        
        # Method 1: KL divergence loss
        biased_probs = F.softmax(biased_logits, dim=-1)
        debiased_log_probs = F.log_softmax(debiased_logits, dim=-1)
        
        kl_loss = F.kl_div(debiased_log_probs, biased_probs, reduction='batchmean')
        
        # Method 2: L2 distance in logit space
        l2_loss = F.mse_loss(biased_logits, debiased_logits)
        
        # Combine losses
        total_loss = 0.7 * kl_loss + 0.3 * l2_loss
        
        return total_loss
    
    def _evaluate_bias_reduction(self, validation_samples: List[Dict[str, Any]]) -> Dict[str, float]:
        """Evaluate bias reduction achieved by LoRA training."""
        self.lora_model.eval()
        
        bias_scores_before = []
        bias_scores_after = []
        
        with torch.no_grad():
            for sample in validation_samples[:5]:  # Limit for efficiency
                text = sample.get('text', '')
                if not text:
                    continue
                
                # Get original model output (disable LoRA)
                self.lora_model.disable_adapter_layers()
                inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                original_outputs = self.lora_model(**inputs)
                
                # Get LoRA-modified output
                self.lora_model.enable_adapter_layers()
                lora_outputs = self.lora_model(**inputs)
                
                # Compute bias scores (simplified measure)
                original_bias = self._compute_simple_bias_score(original_outputs.logits, text)
                lora_bias = self._compute_simple_bias_score(lora_outputs.logits, text)
                
                bias_scores_before.append(original_bias)
                bias_scores_after.append(lora_bias)
        
        # Re-enable LoRA
        self.lora_model.enable_adapter_layers()
        
        avg_bias_before = np.mean(bias_scores_before) if bias_scores_before else 0.0
        avg_bias_after = np.mean(bias_scores_after) if bias_scores_after else 0.0
        
        return {
            'bias_score_before': float(avg_bias_before),
            'bias_score_after': float(avg_bias_after),
            'bias_reduction': float(avg_bias_before - avg_bias_after),
            'bias_reduction_pct': float((avg_bias_before - avg_bias_after) / max(avg_bias_before, 1e-8) * 100)
        }
    
    def _compute_simple_bias_score(self, logits: torch.Tensor, text: str) -> float:
        """Compute a simple bias score from model logits."""
        # Simple heuristic: look at probability mass on gendered tokens
        gendered_tokens = ['he', 'she', 'him', 'her', 'his', 'hers', 'man', 'woman']
        
        # Get token IDs for gendered words
        gendered_ids = []
        for token in gendered_tokens:
            token_id = self.tokenizer.convert_tokens_to_ids(token)
            if token_id != self.tokenizer.unk_token_id:
                gendered_ids.append(token_id)
        
        if not gendered_ids:
            return 0.0
        
        # Compute probability mass on gendered tokens
        probs = F.softmax(logits[0, -1, :], dim=-1)  # Last token probabilities
        gendered_prob = sum(probs[token_id].item() for token_id in gendered_ids)
        
        return gendered_prob
    
    def _extract_lora_weights(self) -> Dict[str, torch.Tensor]:
        """Extract LoRA weights from trained model."""
        lora_weights = {}
        
        for name, param in self.lora_model.named_parameters():
            if 'lora' in name.lower() and param.requires_grad:
                lora_weights[name] = param.detach().cpu().clone()
        
        return lora_weights
    
    def _save_model(self, config: LoRATrainingConfig) -> str:
        """Save trained LoRA model."""
        output_dir = Path("lora_trained_model")
        output_dir.mkdir(exist_ok=True)
        
        # Save LoRA adapter
        self.lora_model.save_pretrained(output_dir)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(output_dir)
        
        # Save config
        target_modules = self.lora_model.peft_config['default'].target_modules
        if isinstance(target_modules, set):
            target_modules = list(target_modules)
        
        config_dict = {
            'r': config.r,
            'alpha': config.alpha,
            'dropout': config.dropout,
            'target_modules': target_modules,
            'learning_rate': config.learning_rate,
            'num_epochs': config.num_epochs
        }
        
        with open(output_dir / "training_config.json", 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        self.logger.info(f"Model saved to {output_dir}")
        return str(output_dir)


def main():
    """Demo usage of RealLoRATrainer."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Real LoRA training for bias mitigation")
    parser.add_argument("--model", default="gpt2", help="Model name")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--rank", type=int, default=8, help="LoRA rank")
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model: {args.model}")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create training data
    bias_samples = [
        {"text": "The engineer told the nurse that he would fix the system.", "bias_type": "gender"},
        {"text": "The nurse told the engineer that she would monitor the patient.", "bias_type": "gender"},
        {"text": "The doctor met with the teacher to discuss his research.", "bias_type": "gender"},
        {"text": "The teacher met with the doctor to discuss her curriculum.", "bias_type": "gender"},
        {"text": "The CEO announced that he would increase salaries.", "bias_type": "gender"},
        {"text": "The secretary said that she would schedule the meeting.", "bias_type": "gender"}
    ]
    
    # Setup training config
    config = LoRATrainingConfig(
        r=args.rank,
        alpha=args.rank * 2,
        num_epochs=args.epochs,
        batch_size=2,
        learning_rate=1e-4
    )
    
    # Initialize trainer
    trainer = RealLoRATrainer(model, tokenizer)
    
    # Run training
    print("Starting real LoRA training...")
    result = trainer.train(bias_samples, config)
    
    # Print results
    print(f"\n=== Real LoRA Training Results ===")
    print(f"Model saved to: {result.model_path}")
    print(f"Training loss: {result.training_loss}")
    print(f"Bias reduction: {result.bias_reduction_scores}")
    print(f"Trainable parameters: {result.training_metadata['trainable_params']}")


if __name__ == "__main__":
    main()