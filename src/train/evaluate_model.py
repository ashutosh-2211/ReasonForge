"""
Model Evaluation Script
========================

Evaluates fine-tuned LoRA model on validation/test datasets.
Computes metrics like perplexity, loss, and generates sample outputs.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time

import torch
from datasets import load_from_disk, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer
from unsloth import FastLanguageModel
import numpy as np

from utils import (
    S3Handler,
    get_gpu_info,
    print_gpu_memory,
    format_time
)


class ModelEvaluator:
    """Evaluates fine-tuned models on various metrics."""

    def __init__(
        self,
        model_path: str,
        is_adapter: bool = True,
        base_model_name: str = "unsloth/Qwen2.5-Coder-3B-Instruct",
        max_seq_length: int = 2048,
        load_in_4bit: bool = True,
    ):
        """
        Initialize model evaluator.

        Args:
            model_path: Path to model or adapter
            is_adapter: Whether model_path is a LoRA adapter
            base_model_name: Base model name (needed if loading adapter)
            max_seq_length: Maximum sequence length
            load_in_4bit: Use 4-bit quantization
        """
        self.model_path = Path(model_path)
        self.is_adapter = is_adapter
        self.base_model_name = base_model_name
        self.max_seq_length = max_seq_length
        self.load_in_4bit = load_in_4bit

        self.model = None
        self.tokenizer = None
        self.s3_handler = S3Handler()

        print(f"✓ Initialized ModelEvaluator")
        print(f"  Model Path: {model_path}")
        print(f"  Is Adapter: {is_adapter}")

    def load_model(self):
        """Load model for evaluation."""
        print(f"\n🔄 Loading model for evaluation...")
        print_gpu_memory()

        if self.is_adapter:
            # Load base model with adapter
            print(f"Loading base model: {self.base_model_name}")
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.base_model_name,
                max_seq_length=self.max_seq_length,
                dtype=None,
                load_in_4bit=self.load_in_4bit,
            )

            # Load adapter weights
            print(f"Loading adapter from: {self.model_path}")
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(
                self.model,
                str(self.model_path)
            )
        else:
            # Load full fine-tuned model
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=str(self.model_path),
                max_seq_length=self.max_seq_length,
                dtype=None,
                load_in_4bit=self.load_in_4bit,
            )

        # Set to evaluation mode
        FastLanguageModel.for_inference(self.model)

        print(f"✓ Model loaded successfully")
        print_gpu_memory()

    def load_dataset(
        self,
        dataset_path: str,
        split: str = "validation"
    ) -> Dataset:
        """
        Load evaluation dataset.

        Args:
            dataset_path: Path to dataset (local or S3)
            split: Dataset split to use

        Returns:
            Dataset for evaluation
        """
        print(f"\n📥 Loading dataset for evaluation...")

        if dataset_path.startswith("s3://"):
            local_path = Path("./eval_cache/dataset")
            dataset_path = self.s3_handler.download_directory(
                dataset_path,
                local_path
            )

        dataset_dict = load_from_disk(str(dataset_path))

        if split not in dataset_dict:
            raise ValueError(f"Split '{split}' not found. Available: {list(dataset_dict.keys())}")

        dataset = dataset_dict[split]
        print(f"✓ Loaded {split} split: {len(dataset):,} examples")

        return dataset

    def compute_perplexity(
        self,
        dataset: Dataset,
        batch_size: int = 4,
        max_samples: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Compute perplexity on dataset.

        Perplexity measures how well the model predicts the text.
        Lower perplexity = better performance.

        Args:
            dataset: Dataset to evaluate on
            batch_size: Batch size for evaluation
            max_samples: Maximum samples to evaluate (None = all)

        Returns:
            Dictionary with perplexity metrics
        """
        print(f"\n📊 Computing perplexity...")

        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))

        self.model.eval()

        total_loss = 0.0
        total_tokens = 0
        num_batches = 0

        with torch.no_grad():
            for i in range(0, len(dataset), batch_size):
                batch = dataset[i:i + batch_size]

                # Tokenize
                inputs = self.tokenizer(
                    batch["text"],
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.max_seq_length,
                    padding=True
                )

                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}

                # Forward pass
                outputs = self.model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss

                # Accumulate
                batch_tokens = inputs["attention_mask"].sum().item()
                total_loss += loss.item() * batch_tokens
                total_tokens += batch_tokens
                num_batches += 1

                if (num_batches % 10) == 0:
                    current_ppl = np.exp(total_loss / total_tokens)
                    print(f"  Batch {num_batches}: Current PPL = {current_ppl:.2f}")

        # Compute final metrics
        avg_loss = total_loss / total_tokens
        perplexity = np.exp(avg_loss)

        metrics = {
            "perplexity": float(perplexity),
            "avg_loss": float(avg_loss),
            "total_tokens": int(total_tokens),
            "num_samples": len(dataset)
        }

        print(f"\n✓ Perplexity Metrics:")
        print(f"  - Perplexity: {perplexity:.2f}")
        print(f"  - Average Loss: {avg_loss:.4f}")
        print(f"  - Samples Evaluated: {len(dataset):,}")

        return metrics

    def generate_sample_outputs(
        self,
        dataset: Dataset,
        num_samples: int = 5,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> List[Dict[str, str]]:
        """
        Generate sample outputs from the model.

        Args:
            dataset: Dataset to sample from
            num_samples: Number of samples to generate
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter

        Returns:
            List of dictionaries with instruction, ground_truth, and generated
        """
        print(f"\n🎯 Generating sample outputs...")

        self.model.eval()
        samples = []

        # Select random samples
        indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)

        for idx in indices:
            example = dataset[int(idx)]

            # Extract instruction (before <|im_start|>assistant)
            text = example["text"]
            instruction_end = text.find("<|im_start|>assistant")

            if instruction_end == -1:
                continue

            instruction_text = text[:instruction_end] + "<|im_start|>assistant\n"
            ground_truth = example.get("response", "")

            # Generate
            inputs = self.tokenizer(
                instruction_text,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_seq_length
            )

            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract only the generated part (after instruction)
            generated_response = generated_text[len(instruction_text):]

            sample = {
                "instruction": example.get("instruction", ""),
                "ground_truth": ground_truth,
                "generated": generated_response
            }
            samples.append(sample)

            print(f"\n{'='*70}")
            print(f"Sample {len(samples)}:")
            print(f"{'='*70}")
            print(f"Instruction: {sample['instruction'][:200]}...")
            print(f"\nGround Truth: {sample['ground_truth'][:200]}...")
            print(f"\nGenerated: {sample['generated'][:200]}...")

        return samples

    def evaluate(
        self,
        dataset: Dataset,
        compute_ppl: bool = True,
        generate_samples: bool = True,
        num_samples: int = 5,
        output_file: Optional[Path] = None
    ) -> Dict:
        """
        Run full evaluation pipeline.

        Args:
            dataset: Dataset to evaluate on
            compute_ppl: Whether to compute perplexity
            generate_samples: Whether to generate sample outputs
            num_samples: Number of samples to generate
            output_file: Path to save evaluation results

        Returns:
            Dictionary with all evaluation metrics
        """
        print("\n" + "="*70)
        print("🔍 STARTING EVALUATION")
        print("="*70)

        results = {
            "model_path": str(self.model_path),
            "num_examples": len(dataset),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        # Compute perplexity
        if compute_ppl:
            ppl_metrics = self.compute_perplexity(dataset)
            results["perplexity_metrics"] = ppl_metrics

        # Generate samples
        if generate_samples:
            samples = self.generate_sample_outputs(dataset, num_samples=num_samples)
            results["sample_outputs"] = samples

        # Save results
        if output_file:
            output_file = Path(output_file)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)

            print(f"\n💾 Evaluation results saved to: {output_file}")

        print("\n" + "="*70)
        print("✅ EVALUATION COMPLETE")
        print("="*70)

        return results


def main():
    """Main execution flow for evaluation."""

    parser = argparse.ArgumentParser(description="Evaluate LoRA fine-tuned model")

    # Model arguments
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to model or adapter (local or S3)"
    )
    parser.add_argument(
        "--is-adapter",
        action="store_true",
        help="Model path is a LoRA adapter"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="unsloth/Qwen2.5-Coder-3B-Instruct",
        help="Base model name (needed for adapter)"
    )

    # Dataset arguments
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="Path to evaluation dataset"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        choices=["train", "validation", "test"],
        help="Dataset split to evaluate on"
    )

    # Evaluation arguments
    parser.add_argument(
        "--compute-perplexity",
        action="store_true",
        default=True,
        help="Compute perplexity metrics"
    )
    parser.add_argument(
        "--generate-samples",
        action="store_true",
        default=True,
        help="Generate sample outputs"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of samples to generate"
    )
    parser.add_argument(
        "--max-eval-samples",
        type=int,
        default=None,
        help="Maximum samples for perplexity computation"
    )

    # Output arguments
    parser.add_argument(
        "--output-file",
        type=str,
        default="./eval_results/evaluation.json",
        help="Path to save evaluation results"
    )

    args = parser.parse_args()

    # Print GPU info
    gpu_info = get_gpu_info()
    print("\n" + "=" * 70)
    print("🎮 GPU INFORMATION")
    print("=" * 70)
    for key, value in gpu_info.items():
        print(f"  {key}: {value}")
    print("=" * 70)

    # Initialize evaluator
    evaluator = ModelEvaluator(
        model_path=args.model_path,
        is_adapter=args.is_adapter,
        base_model_name=args.base_model,
    )

    # Load model
    evaluator.load_model()

    # Load dataset
    dataset = evaluator.load_dataset(
        dataset_path=args.dataset_path,
        split=args.split
    )

    # Run evaluation
    results = evaluator.evaluate(
        dataset=dataset,
        compute_ppl=args.compute_perplexity,
        generate_samples=args.generate_samples,
        num_samples=args.num_samples,
        output_file=args.output_file
    )

    # Print summary
    print("\n📊 EVALUATION SUMMARY:")
    if "perplexity_metrics" in results:
        ppl = results["perplexity_metrics"]["perplexity"]
        print(f"  - Perplexity: {ppl:.2f}")
    if "sample_outputs" in results:
        print(f"  - Generated {len(results['sample_outputs'])} samples")


if __name__ == "__main__":
    main()
