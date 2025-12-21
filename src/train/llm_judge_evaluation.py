"""
LLM-as-a-Judge Evaluation: Base vs Fine-tuned Model
====================================================

Compares base model vs fine-tuned model using GPT-5-mini as an objective judge.
- Generates outputs from both models (blind labels A/B)
- Uses GPT-5-mini to judge which output is better
- Provides statistical analysis of improvement

Usage:
    export OPENAI_API_KEY="your-api-key"

    uv run llm_judge_evaluation.py \
        --base-model "unsloth/Qwen3-4B" \
        --finetuned-model "./checkpoints/final_model" \
        --dataset-path "./data_cache/processed_reasoning" \
        --num-samples 50 \
        --output-file "./eval_results/judge_evaluation.json" \
        --openai-api-key "$OPENAI_API_KEY"
"""
import unsloth
import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional
import random

import torch
import numpy as np
from datasets import load_from_disk, Dataset
from unsloth import FastLanguageModel
from openai import OpenAI
from tqdm import tqdm


class LLMJudgeEvaluator:
    """Evaluate models using GPT-5-mini as a judge."""

    def __init__(
        self,
        base_model_name: str = "unsloth/Qwen3-4B",
        max_seq_length: int = 2048,
        openai_api_key: Optional[str] = None,
    ):
        self.base_model_name = base_model_name
        self.max_seq_length = max_seq_length

        # Initialize OpenAI client
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY env var or pass --openai-api-key")

        self.client = OpenAI(api_key=api_key)

        print("✓ Initialized LLMJudgeEvaluator")

    def generate_outputs(
        self,
        model_path: str,
        dataset: Dataset,
        is_adapter: bool = False,
        label: str = "Model",
    ) -> List[Dict]:
        """
        Generate outputs from a model.

        Args:
            model_path: Path to model or adapter
            dataset: Dataset to generate from
            is_adapter: Whether model_path is an adapter
            label: Label for this model (A or B)

        Returns:
            List of dictionaries with instruction, ground_truth, and generated output
        """
        print(f"\n{'='*70}")
        print(f"Generating outputs from {label}")
        print(f"{'='*70}")

        # Load model
        print(f"Loading model: {model_path}")

        if is_adapter:
            # Load base model + adapter
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.base_model_name,
                max_seq_length=self.max_seq_length,
                dtype=None,
                load_in_4bit=True,
            )

            from peft import PeftModel
            model = PeftModel.from_pretrained(model, model_path)
        else:
            # Load base model directly
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_path,
                max_seq_length=self.max_seq_length,
                dtype=None,
                load_in_4bit=True,
            )

        FastLanguageModel.for_inference(model)
        print(f"✓ Model loaded")

        # Generate outputs
        results = []

        for idx in tqdm(range(len(dataset)), desc=f"Generating ({label})"):
            example = dataset[idx]
            text = example["text"]

            # Extract instruction
            instruction_end = text.find("<|im_start|>assistant")
            if instruction_end == -1:
                continue

            instruction_text = text[:instruction_end] + "<|im_start|>assistant\n"
            ground_truth = example.get("response", "")

            # Generate
            inputs = tokenizer(
                instruction_text,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_seq_length
            )

            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.3,
                    top_p=0.9,
                    do_sample=True,
                )

            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated = generated[len(instruction_text):]

            results.append({
                "instruction": example.get("instruction", ""),
                "ground_truth": ground_truth,
                "generated": generated,
                "model_label": label
            })

        # Cleanup
        del model
        del tokenizer
        torch.cuda.empty_cache()

        print(f"✓ Generated {len(results)} outputs from {label}")

        return results

    def create_judgment_prompt(
        self,
        instruction: str,
        ground_truth: str,
        output_a: str,
        output_b: str,
    ) -> str:
        """
        Create a prompt for GPT-5-mini to judge which output is better.

        Following best practices from confident-ai.com:
        - Clear evaluation criteria
        - Structured output format
        - Chain-of-thought reasoning
        """
        prompt = f"""
        You are an expert evaluator for mathematical and reasoning tasks. Your job is to determine which AI model's response is better.

                    **Task/Question:**
                    {instruction}

                    **Ground Truth Reference:**
                    {ground_truth}

                    **Model A's Response:**
                    {output_a}

                    **Model B's Response:**
                    {output_b}

                    **Evaluation Criteria:**
                    1. **Correctness**: Does the response arrive at the correct answer?
                    2. **Reasoning Quality**: Is the step-by-step reasoning sound and clear?
                    3. **Completeness**: Does it address all aspects of the question?
                    4. **Mathematical Accuracy**: Are calculations and formulas correct?
                    5. **Clarity**: Is the explanation easy to follow?

                    **Instructions:**
                    1. First, analyze both responses against each criterion
                    2. Then, make your final judgment
                    3. Respond in JSON format with the following structure:

                    {{
                        "analysis": "Your detailed analysis comparing both responses",
                        "winner": "A" or "B" or "tie",
                        "confidence": "high" or "medium" or "low",
                        "reasoning": "Brief explanation of your decision"
                    }}

                    Provide your evaluation:
                    """

        return prompt

    def judge_with_gpt(
        self,
        comparisons: List[Dict],
        model: str = "gpt-5-mini",
    ) -> List[Dict]:
        """
        Use GPT-5-mini to judge which output is better.

        Args:
            comparisons: List of comparison dictionaries
            model: OpenAI model to use for judging

        Returns:
            List of judgments
        """
        print(f"\n🤖 Judging {len(comparisons)} comparisons using {model}...")

        judgments = []

        for comparison in tqdm(comparisons, desc="GPT-5-mini Judging"):
            prompt = self.create_judgment_prompt(
                instruction=comparison["instruction"],
                ground_truth=comparison["ground_truth"],
                output_a=comparison["output_a"],
                output_b=comparison["output_b"],
            )

            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are an expert AI evaluator. Always respond in valid JSON format."},
                        {"role": "user", "content": prompt}
                    ]
                )

                # Parse JSON response
                content = response.choices[0].message.content.strip()

                # Extract JSON from markdown code blocks if present
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()

                judgment = json.loads(content)

                # Add metadata
                judgment["comparison_id"] = len(judgments)
                judgment["instruction"] = comparison["instruction"][:100] + "..."

                judgments.append(judgment)

            except Exception as e:
                print(f"\n⚠️  Error judging comparison {len(judgments)}: {e}")
                # Add fallback judgment
                judgments.append({
                    "comparison_id": len(judgments),
                    "winner": "error",
                    "confidence": "low",
                    "analysis": f"Error during judgment: {str(e)}",
                    "reasoning": "Failed to get judgment",
                    "instruction": comparison["instruction"][:100] + "..."
                })

            # Small delay to avoid rate limits
            time.sleep(0.5)

        return judgments

    def analyze_results(self, judgments: List[Dict]) -> Dict:
        """
        Analyze judgment results and compute statistics.

        Args:
            judgments: List of judgment dictionaries

        Returns:
            Dictionary with analysis results
        """
        print("\n📊 Analyzing results...")

        # Count wins
        a_wins = sum(1 for j in judgments if j["winner"] == "A")
        b_wins = sum(1 for j in judgments if j["winner"] == "B")
        ties = sum(1 for j in judgments if j["winner"] == "tie")
        errors = sum(1 for j in judgments if j["winner"] == "error")

        total_valid = len(judgments) - errors

        # Confidence breakdown
        high_conf = sum(1 for j in judgments if j.get("confidence") == "high")
        med_conf = sum(1 for j in judgments if j.get("confidence") == "medium")
        low_conf = sum(1 for j in judgments if j.get("confidence") == "low")

        # Calculate win rates
        a_win_rate = (a_wins / total_valid * 100) if total_valid > 0 else 0
        b_win_rate = (b_wins / total_valid * 100) if total_valid > 0 else 0
        tie_rate = (ties / total_valid * 100) if total_valid > 0 else 0

        analysis = {
            "total_comparisons": len(judgments),
            "valid_judgments": total_valid,
            "model_a_wins": a_wins,
            "model_b_wins": b_wins,
            "ties": ties,
            "errors": errors,
            "model_a_win_rate": round(a_win_rate, 2),
            "model_b_win_rate": round(b_win_rate, 2),
            "tie_rate": round(tie_rate, 2),
            "confidence_distribution": {
                "high": high_conf,
                "medium": med_conf,
                "low": low_conf
            }
        }

        return analysis

    def run_evaluation(
        self,
        base_model_path: str,
        finetuned_model_path: str,
        dataset: Dataset,
        batch_size: int = 5,
    ) -> Dict:
        """
        Run complete evaluation pipeline.

        Args:
            base_model_path: Path to base model
            finetuned_model_path: Path to fine-tuned model/adapter
            dataset: Dataset to evaluate on
            batch_size: Number of comparisons to judge at once

        Returns:
            Complete evaluation results
        """
        print("\n" + "="*70)
        print("LLM-AS-A-JUDGE EVALUATION")
        print("="*70)

        # Randomly assign which model gets label A vs B (blind evaluation)
        randomize = random.choice([True, False])

        if randomize:
            model_a_path = finetuned_model_path
            model_a_is_adapter = True
            model_a_name = "Fine-tuned"

            model_b_path = base_model_path
            model_b_is_adapter = False
            model_b_name = "Base"
        else:
            model_a_path = base_model_path
            model_a_is_adapter = False
            model_a_name = "Base"

            model_b_path = finetuned_model_path
            model_b_is_adapter = True
            model_b_name = "Fine-tuned"

        print(f"\n🔀 Random assignment: A={model_a_name}, B={model_b_name}")

        # Generate outputs from both models
        outputs_a = self.generate_outputs(
            model_a_path,
            dataset,
            is_adapter=model_a_is_adapter,
            label="A"
        )

        print("\n⏳ Waiting 10 seconds to free GPU...")
        time.sleep(10)

        outputs_b = self.generate_outputs(
            model_b_path,
            dataset,
            is_adapter=model_b_is_adapter,
            label="B"
        )

        # Create comparisons
        comparisons = []
        for out_a, out_b in zip(outputs_a, outputs_b):
            comparisons.append({
                "instruction": out_a["instruction"],
                "ground_truth": out_a["ground_truth"],
                "output_a": out_a["generated"],
                "output_b": out_b["generated"],
            })

        # Judge in batches
        all_judgments = []

        for i in range(0, len(comparisons), batch_size):
            batch = comparisons[i:i + batch_size]
            print(f"\nJudging batch {i//batch_size + 1}/{(len(comparisons)-1)//batch_size + 1}")

            judgments = self.judge_with_gpt(batch)
            all_judgments.extend(judgments)

        # Analyze results
        analysis = self.analyze_results(all_judgments)

        # Prepare final results
        results = {
            "metadata": {
                "base_model": base_model_path,
                "finetuned_model": finetuned_model_path,
                "num_samples": len(dataset),
                "model_a_is": model_a_name,
                "model_b_is": model_b_name,
                "judge_model": "gpt-5-mini",
            },
            "analysis": analysis,
            "judgments": all_judgments,
            "comparisons": comparisons
        }

        return results


def print_results(results: Dict):
    """Print formatted evaluation results."""
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)

    metadata = results["metadata"]
    analysis = results["analysis"]

    print(f"\nModel A: {metadata['model_a_is']}")
    print(f"Model B: {metadata['model_b_is']}")
    print(f"Judge: {metadata['judge_model']}")
    print(f"Total Comparisons: {analysis['total_comparisons']}")

    print(f"\n{'='*70}")
    print("WIN RATES")
    print(f"{'='*70}")
    print(f"Model A Wins: {analysis['model_a_wins']} ({analysis['model_a_win_rate']}%)")
    print(f"Model B Wins: {analysis['model_b_wins']} ({analysis['model_b_win_rate']}%)")
    print(f"Ties: {analysis['ties']} ({analysis['tie_rate']}%)")

    # Determine which is fine-tuned
    if metadata['model_a_is'] == 'Fine-tuned':
        ft_wins = analysis['model_a_wins']
        ft_rate = analysis['model_a_win_rate']
        base_wins = analysis['model_b_wins']
        base_rate = analysis['model_b_win_rate']
    else:
        ft_wins = analysis['model_b_wins']
        ft_rate = analysis['model_b_win_rate']
        base_wins = analysis['model_a_wins']
        base_rate = analysis['model_a_win_rate']

    print(f"\n{'='*70}")
    print("FINE-TUNING IMPACT")
    print(f"{'='*70}")
    print(f"Fine-tuned Wins: {ft_wins} ({ft_rate}%)")
    print(f"Base Model Wins: {base_wins} ({base_rate}%)")

    if ft_rate > base_rate + 10:
        verdict = "✅ SIGNIFICANT IMPROVEMENT"
    elif ft_rate > base_rate:
        verdict = "✓ Modest improvement"
    elif ft_rate == base_rate:
        verdict = "➖ No clear improvement"
    else:
        verdict = "⚠️  Base model performed better"

    print(f"\nVerdict: {verdict}")

    print(f"\n{'='*70}")
    print("CONFIDENCE DISTRIBUTION")
    print(f"{'='*70}")
    conf = analysis['confidence_distribution']
    print(f"High Confidence: {conf['high']}")
    print(f"Medium Confidence: {conf['medium']}")
    print(f"Low Confidence: {conf['low']}")

    print("="*70)


def main():
    parser = argparse.ArgumentParser(description="LLM-as-a-Judge Evaluation")

    parser.add_argument(
        "--base-model",
        type=str,
        default="unsloth/Qwen3-4B",
        help="Base model name or path"
    )
    parser.add_argument(
        "--finetuned-model",
        type=str,
        required=True,
        help="Path to fine-tuned model/adapter"
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="Path to dataset"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=50,
        help="Number of samples to evaluate"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="Batch size for GPT-5-mini judging"
    )
    parser.add_argument(
        "--openai-api-key",
        type=str,
        default=None,
        help="OpenAI API key (or set OPENAI_API_KEY env var)"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="./eval_results/judge_evaluation.json",
        help="Output file for results"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load dataset
    print(f"Loading dataset from: {args.dataset_path}")
    dataset_dict = load_from_disk(args.dataset_path)
    val_dataset = dataset_dict["validation"]

    # Sample subset
    if args.num_samples < len(val_dataset):
        indices = random.sample(range(len(val_dataset)), args.num_samples)
        val_dataset = val_dataset.select(indices)

    print(f"✓ Selected {len(val_dataset)} samples for evaluation")

    # Initialize evaluator
    evaluator = LLMJudgeEvaluator(
        base_model_name=args.base_model,
        openai_api_key=args.openai_api_key,
    )

    # Run evaluation
    results = evaluator.run_evaluation(
        base_model_path=args.base_model,
        finetuned_model_path=args.finetuned_model,
        dataset=val_dataset,
        batch_size=args.batch_size,
    )

    # Save results
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Results saved to: {output_file}")

    # Print summary
    print_results(results)


if __name__ == "__main__":
    main()
