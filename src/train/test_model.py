"""
Interactive Model Testing Script
=================================

Test your fine-tuned model with interactive prompts or batch inference.
Supports both local and S3-stored models.
"""

import argparse
import json
from pathlib import Path
from typing import Optional, List, Dict

import torch
from unsloth import FastLanguageModel

from utils import get_gpu_info, print_gpu_memory


class ModelTester:
    """Interactive testing interface for fine-tuned models."""

    def __init__(
        self,
        model_path: str,
        is_adapter: bool = True,
        base_model_name: str = "unsloth/Qwen2.5-Coder-3B-Instruct",
        max_seq_length: int = 2048,
        load_in_4bit: bool = True,
    ):
        """
        Initialize model tester.

        Args:
            model_path: Path to model or adapter
            is_adapter: Whether model_path is a LoRA adapter
            base_model_name: Base model name (if loading adapter)
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

        print(f"✓ Initialized ModelTester")

    def load_model(self):
        """Load model for inference."""
        print(f"\n🔄 Loading model...")
        print_gpu_memory()

        if self.is_adapter:
            # Load base model
            print(f"Loading base model: {self.base_model_name}")
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.base_model_name,
                max_seq_length=self.max_seq_length,
                dtype=None,
                load_in_4bit=self.load_in_4bit,
            )

            # Load adapter
            print(f"Loading adapter from: {self.model_path}")
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(
                self.model,
                str(self.model_path)
            )
        else:
            # Load full model
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=str(self.model_path),
                max_seq_length=self.max_seq_length,
                dtype=None,
                load_in_4bit=self.load_in_4bit,
            )

        # Enable fast inference
        FastLanguageModel.for_inference(self.model)

        print(f"✓ Model loaded and ready for inference")
        print_gpu_memory()

    def format_prompt(self, instruction: str, system_prompt: Optional[str] = None) -> str:
        """
        Format instruction into chat template.

        Args:
            instruction: User instruction/question
            system_prompt: System prompt (optional)

        Returns:
            Formatted prompt string
        """
        if system_prompt is None:
            system_prompt = "You are a helpful assistant that provides detailed reasoning and explanations."

        prompt = (
            f"<|im_start|>system\n"
            f"{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n"
            f"{instruction}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        return prompt

    def generate(
        self,
        instruction: str,
        system_prompt: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
    ) -> str:
        """
        Generate response for an instruction.

        Args:
            instruction: User instruction/question
            system_prompt: System prompt (optional)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            repetition_penalty: Penalty for repetition

        Returns:
            Generated response
        """
        # Format prompt
        prompt = self.format_prompt(instruction, system_prompt)

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_seq_length
        )

        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=False)

        # Extract only the assistant's response
        assistant_start = generated_text.find("<|im_start|>assistant\n")
        if assistant_start != -1:
            response = generated_text[assistant_start + len("<|im_start|>assistant\n"):]
            # Remove end token if present
            response = response.replace("<|im_end|>", "").strip()
        else:
            response = generated_text

        return response

    def interactive_mode(
        self,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
    ):
        """
        Run interactive testing mode.

        Args:
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
        """
        print("\n" + "="*70)
        print("🎯 INTERACTIVE TESTING MODE")
        print("="*70)
        print("Type your questions below. Type 'quit' or 'exit' to stop.")
        print("Type 'settings' to change generation parameters.")
        print("="*70 + "\n")

        current_max_tokens = max_new_tokens
        current_temperature = temperature

        while True:
            try:
                instruction = input("\n💬 You: ").strip()

                if not instruction:
                    continue

                if instruction.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Goodbye!")
                    break

                if instruction.lower() == 'settings':
                    print(f"\nCurrent settings:")
                    print(f"  - max_new_tokens: {current_max_tokens}")
                    print(f"  - temperature: {current_temperature}")

                    try:
                        new_tokens = input(f"New max_new_tokens [{current_max_tokens}]: ").strip()
                        if new_tokens:
                            current_max_tokens = int(new_tokens)

                        new_temp = input(f"New temperature [{current_temperature}]: ").strip()
                        if new_temp:
                            current_temperature = float(new_temp)

                        print(f"✓ Settings updated!")
                    except ValueError:
                        print("✗ Invalid input, keeping current settings")

                    continue

                print("\n🤖 Assistant: ", end="", flush=True)

                response = self.generate(
                    instruction=instruction,
                    max_new_tokens=current_max_tokens,
                    temperature=current_temperature
                )

                print(response)

            except KeyboardInterrupt:
                print("\n\n👋 Interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n✗ Error: {e}")

    def batch_inference(
        self,
        instructions: List[str],
        output_file: Optional[Path] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
    ) -> List[Dict[str, str]]:
        """
        Run batch inference on multiple instructions.

        Args:
            instructions: List of instructions
            output_file: Path to save results (optional)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            List of dictionaries with instruction and response
        """
        print(f"\n🔄 Running batch inference on {len(instructions)} instructions...")

        results = []

        for i, instruction in enumerate(instructions, 1):
            print(f"\nProcessing {i}/{len(instructions)}...")

            response = self.generate(
                instruction=instruction,
                max_new_tokens=max_new_tokens,
                temperature=temperature
            )

            result = {
                "instruction": instruction,
                "response": response
            }
            results.append(result)

            print(f"Instruction: {instruction[:100]}...")
            print(f"Response: {response[:100]}...")

        # Save results
        if output_file:
            output_file = Path(output_file)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)

            print(f"\n💾 Results saved to: {output_file}")

        print(f"\n✓ Batch inference complete!")
        return results


def main():
    """Main execution flow for model testing."""

    parser = argparse.ArgumentParser(description="Test fine-tuned LoRA model")

    # Model arguments
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to model or adapter"
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

    # Mode selection
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Run batch inference"
    )

    # Batch mode arguments
    parser.add_argument(
        "--instructions-file",
        type=str,
        help="JSON file with list of instructions for batch mode"
    )
    parser.add_argument(
        "--instruction",
        type=str,
        help="Single instruction to test"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="./test_results/inference.json",
        help="Output file for batch results"
    )

    # Generation arguments
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
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

    # Initialize tester
    tester = ModelTester(
        model_path=args.model_path,
        is_adapter=args.is_adapter,
        base_model_name=args.base_model,
    )

    # Load model
    tester.load_model()

    # Run appropriate mode
    if args.interactive:
        tester.interactive_mode(
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature
        )

    elif args.batch:
        # Load instructions
        if args.instructions_file:
            with open(args.instructions_file, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    instructions = data
                else:
                    instructions = data.get("instructions", [])
        elif args.instruction:
            instructions = [args.instruction]
        else:
            raise ValueError("Provide --instructions-file or --instruction for batch mode")

        # Run batch inference
        tester.batch_inference(
            instructions=instructions,
            output_file=args.output_file,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature
        )

    elif args.instruction:
        # Single instruction test
        print(f"\n💬 Instruction: {args.instruction}")
        response = tester.generate(
            instruction=args.instruction,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature
        )
        print(f"\n🤖 Response:\n{response}")

    else:
        print("\n⚠️  No mode selected. Use --interactive, --batch, or --instruction")
        print("Run with --help for usage information")


if __name__ == "__main__":
    main()
