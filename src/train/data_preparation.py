"""
Data Preparation Script for LoRA Fine-tuning Pipeline
======================================================

This script handles:
1. Loading dataset from HuggingFace Hub
2. Processing and formatting the dataset for training
3. Uploading processed dataset to S3
4. Creating train/validation splits

Dataset: TuringEnterprises/Turing-Open-Reasoning
Model: unsloth/Qwen3-4B
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional
import boto3
from datasets import load_dataset, Dataset, DatasetDict
from datetime import datetime
import argparse


class DatasetProcessor:
    """Handles dataset loading, processing, and S3 upload operations."""

    def __init__(
        self,
        dataset_name: str = "OpenMed/Medical-Reasoning-SFT-GPT-OSS-120B",
        s3_bucket: str = "reasonforge-datasets-dev-071909720457",
        local_cache_dir: str = "./data_cache",
        aws_region: str = "us-east-1"
    ):
        """
        Initialize dataset processor.

        Args:
            dataset_name: HuggingFace dataset identifier
            s3_bucket: S3 bucket name for dataset storage
            local_cache_dir: Local directory for caching datasets
            aws_region: AWS region for S3 bucket
        """
        self.dataset_name = dataset_name
        self.s3_bucket = s3_bucket
        self.local_cache_dir = Path(local_cache_dir)
        self.local_cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize S3 client
        self.s3_client = boto3.client('s3', region_name=aws_region)

        print(f"✓ Initialized DatasetProcessor")
        print(f"  Dataset: {dataset_name}")
        print(f"  S3 Bucket: {s3_bucket}")
        print(f"  Local Cache: {local_cache_dir}")

    def load_dataset_from_hf(self, split: Optional[str] = None) -> Dataset | DatasetDict:
        """
        Load dataset from HuggingFace Hub.

        Args:
            split: Specific split to load (e.g., 'train', 'test'). If None, loads all splits.

        Returns:
            Dataset or DatasetDict object
        """
        print(f"\n📥 Loading dataset from HuggingFace Hub: {self.dataset_name}")

        try:
            dataset = load_dataset(self.dataset_name, split=split)

            if isinstance(dataset, DatasetDict):
                print(f"✓ Loaded dataset with splits: {list(dataset.keys())}")
                for split_name, split_data in dataset.items():
                    print(f"  - {split_name}: {len(split_data)} examples")
            else:
                print(f"✓ Loaded dataset: {len(dataset)} examples")

            return dataset

        except Exception as e:
            print(f"✗ Error loading dataset: {e}")
            raise

    def format_dataset_for_training(self, dataset: Dataset) -> Dataset:
        """
        Format dataset for instruction fine-tuning.

        The Turing-Open-Reasoning dataset contains reasoning traces.
        We format them into instruction-response pairs for training.

        Args:
            dataset: Raw dataset from HuggingFace

        Returns:
            Formatted dataset ready for training
        """
        print(f"\n🔄 Formatting dataset for training...")

        def format_example(example: Dict) -> Dict:
            """
            Format a single example into instruction-following format.

            Expected format for Qwen3:
            <|im_start|>system
            You are a helpful assistant.<|im_end|>
            <|im_start|>user
            {instruction}<|im_end|>
            <|im_start|>assistant
            {response}<|im_end|>
            """
            # Check available fields in the dataset
            # Common fields in reasoning datasets: 'question', 'answer', 'reasoning', 'solution'

            # Adapt based on actual dataset structure
            if 'question' in example and 'reasoning' in example:
                instruction = example['question']
                response = example['reasoning']
            elif 'prompt' in example and 'completion' in example:
                instruction = example['prompt']
                response = example['completion']
            elif 'input' in example and 'output' in example:
                # isaiahbjork/r1-reasoning-json format
                instruction = example['input']
                response = example['output']

                # Try to parse JSON output for cleaner format
                try:
                    import json
                    output_json = json.loads(response)

                    # Check for different possible field names
                    if 'process' in output_json and 'final_answer' in output_json:
                        # Combine reasoning process with final answer
                        response = f"{output_json['process']}\n\n{output_json['final_answer']}"
                    elif 'reasoning' in output_json and 'solution' in output_json:
                        # Alternative format with reasoning and solution
                        response = f"{output_json['reasoning']}\n\n{output_json['solution']}"
                    elif 'reasoning' in output_json:
                        # Just reasoning field
                        response = output_json['reasoning']
                    # else: keep original JSON string
                except (json.JSONDecodeError, TypeError, KeyError) as e:
                    # Keep original if not valid JSON or missing fields
                    pass
            else:
                # Fallback: use first two text fields
                text_fields = [k for k, v in example.items() if isinstance(v, str)]
                instruction = example[text_fields[0]] if len(text_fields) > 0 else ""
                response = example[text_fields[1]] if len(text_fields) > 1 else ""

            # Create the formatted conversation
            formatted_text = (
                f"<|im_start|>system\n"
                f"You are a helpful assistant that provides detailed reasoning and explanations.<|im_end|>\n"
                f"<|im_start|>user\n"
                f"{instruction}<|im_end|>\n"
                f"<|im_start|>assistant\n"
                f"{response}<|im_end|>"
            )

            return {
                "text": formatted_text,
                "instruction": instruction,
                "response": response
            }

        formatted_dataset = dataset.map(
            format_example,
            desc="Formatting examples",
            remove_columns=dataset.column_names
        )

        print(f"✓ Formatted {len(formatted_dataset)} examples")
        print(f"  Sample formatted text (first 200 chars):")
        print(f"  {formatted_dataset[0]['text'][:200]}...")

        return formatted_dataset

    def create_train_val_split(
        self,
        dataset: Dataset,
        val_size: float = 0.1,
        seed: int = 42
    ) -> DatasetDict:
        """
        Create train/validation split from dataset.

        Args:
            dataset: Dataset to split
            val_size: Fraction of data to use for validation (0.0-1.0)
            seed: Random seed for reproducibility

        Returns:
            DatasetDict with 'train' and 'validation' splits
        """
        print(f"\n✂️  Creating train/validation split (val_size={val_size})...")

        split_dataset = dataset.train_test_split(
            test_size=val_size,
            seed=seed,
            shuffle=True
        )

        # Rename 'test' to 'validation'
        dataset_dict = DatasetDict({
            'train': split_dataset['train'],
            'validation': split_dataset['test']
        })

        print(f"✓ Split created:")
        print(f"  - Training: {len(dataset_dict['train'])} examples")
        print(f"  - Validation: {len(dataset_dict['validation'])} examples")

        return dataset_dict

    def save_dataset_locally(self, dataset: Dataset | DatasetDict, name: str = "processed"):
        """
        Save processed dataset to local cache.

        Args:
            dataset: Dataset to save
            name: Name for the saved dataset

        Returns:
            Path to saved dataset
        """
        save_path = self.local_cache_dir / name
        print(f"\n💾 Saving dataset locally to {save_path}...")

        dataset.save_to_disk(str(save_path))

        print(f"✓ Dataset saved successfully")
        return save_path

    def upload_to_s3(self, local_path: Path, s3_prefix: str = "datasets"):
        """
        Upload local dataset to S3 bucket.

        Args:
            local_path: Local path to dataset directory
            s3_prefix: S3 prefix (folder) for uploaded files

        Returns:
            S3 URI of uploaded dataset
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        s3_key_prefix = f"{s3_prefix}/{local_path.name}_{timestamp}"

        print(f"\n☁️  Uploading dataset to S3...")
        print(f"  Bucket: s3://{self.s3_bucket}/{s3_key_prefix}")

        uploaded_files = []

        # Walk through all files in the local directory
        for local_file in local_path.rglob("*"):
            if local_file.is_file():
                # Calculate relative path for S3 key
                relative_path = local_file.relative_to(local_path)
                s3_key = f"{s3_key_prefix}/{relative_path}"

                # Upload file
                try:
                    self.s3_client.upload_file(
                        str(local_file),
                        self.s3_bucket,
                        s3_key
                    )
                    uploaded_files.append(s3_key)
                    print(f"  ✓ Uploaded: {relative_path}")

                except Exception as e:
                    print(f"  ✗ Failed to upload {relative_path}: {e}")
                    raise

        s3_uri = f"s3://{self.s3_bucket}/{s3_key_prefix}"
        print(f"\n✓ Upload complete! Total files: {len(uploaded_files)}")
        print(f"  S3 URI: {s3_uri}")

        # Save metadata
        metadata = {
            "dataset_name": self.dataset_name,
            "s3_uri": s3_uri,
            "timestamp": timestamp,
            "files": uploaded_files
        }

        metadata_path = self.local_cache_dir / f"upload_metadata_{timestamp}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"  Metadata saved to: {metadata_path}")

        return s3_uri

    def download_from_s3(self, s3_uri: str, local_path: Optional[Path] = None) -> Path:
        """
        Download dataset from S3 to local directory.

        Args:
            s3_uri: S3 URI (s3://bucket/prefix)
            local_path: Local path to download to (optional)

        Returns:
            Path to downloaded dataset
        """
        # Parse S3 URI
        if not s3_uri.startswith("s3://"):
            raise ValueError(f"Invalid S3 URI: {s3_uri}")

        parts = s3_uri[5:].split("/", 1)
        bucket = parts[0]
        prefix = parts[1] if len(parts) > 1 else ""

        if local_path is None:
            local_path = self.local_cache_dir / Path(prefix).name

        print(f"\n⬇️  Downloading dataset from S3...")
        print(f"  From: {s3_uri}")
        print(f"  To: {local_path}")

        local_path.mkdir(parents=True, exist_ok=True)

        # List all objects with the prefix
        paginator = self.s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

        downloaded_files = 0
        for page in pages:
            if 'Contents' not in page:
                continue

            for obj in page['Contents']:
                s3_key = obj['Key']

                # Calculate local file path
                relative_path = Path(s3_key).relative_to(prefix)
                local_file = local_path / relative_path
                local_file.parent.mkdir(parents=True, exist_ok=True)

                # Download file
                try:
                    self.s3_client.download_file(bucket, s3_key, str(local_file))
                    downloaded_files += 1
                    print(f"  ✓ Downloaded: {relative_path}")

                except Exception as e:
                    print(f"  ✗ Failed to download {s3_key}: {e}")
                    raise

        print(f"\n✓ Download complete! Total files: {downloaded_files}")
        return local_path


def main():
    """Main execution flow for data preparation."""

    parser = argparse.ArgumentParser(description="Prepare dataset for LoRA fine-tuning")
    parser.add_argument(
        "--dataset",
        type=str,
        default="TuringEnterprises/Turing-Open-Reasoning",
        help="HuggingFace dataset name"
    )
    parser.add_argument(
        "--s3-bucket",
        type=str,
        default="reasonforge-datasets-dev-071909720457",
        help="S3 bucket for dataset storage"
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.1,
        help="Validation split size (0.0-1.0)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (for testing)"
    )
    parser.add_argument(
        "--skip-upload",
        action="store_true",
        help="Skip S3 upload step"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("DATA PREPARATION PIPELINE")
    print("=" * 70)

    # Initialize processor
    processor = DatasetProcessor(
        dataset_name=args.dataset,
        s3_bucket=args.s3_bucket
    )

    # Step 1: Load dataset from HuggingFace
    dataset = processor.load_dataset_from_hf()

    # Handle different dataset structures
    if isinstance(dataset, DatasetDict):
        # Use train split if available
        if 'train' in dataset:
            dataset = dataset['train']
        else:
            # Use the first available split
            dataset = dataset[list(dataset.keys())[0]]

    # Limit samples if specified (useful for testing)
    if args.max_samples:
        print(f"\n⚠️  Limiting dataset to {args.max_samples} samples for testing")
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    # Step 2: Format dataset
    formatted_dataset = processor.format_dataset_for_training(dataset)

    # Step 3: Create train/validation split
    dataset_dict = processor.create_train_val_split(
        formatted_dataset,
        val_size=args.val_size
    )

    # Step 4: Save locally
    local_path = processor.save_dataset_locally(dataset_dict, name="processed_reasoning")

    # Step 5: Upload to S3
    if not args.skip_upload:
        s3_uri = processor.upload_to_s3(local_path, s3_prefix="datasets")

        print("\n" + "=" * 70)
        print("✅ DATA PREPARATION COMPLETE!")
        print("=" * 70)
        print(f"Local Path: {local_path}")
        print(f"S3 URI: {s3_uri}")
        print(f"\nTo use this dataset in training, set:")
        print(f"  --dataset-path '{s3_uri}'")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print("✅ DATA PREPARATION COMPLETE (Local only)")
        print("=" * 70)
        print(f"Local Path: {local_path}")
        print("=" * 70)


if __name__ == "__main__":
    main()
