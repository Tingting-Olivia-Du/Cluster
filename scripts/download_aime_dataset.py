#!/usr/bin/env python3
"""
Download AIME dataset from Hugging Face and save as JSONL format.
Dataset: https://huggingface.co/datasets/di-zhang-fdu/AIME_1983_2024
"""

import json
import os
from datasets import load_dataset
from pathlib import Path

def download_aime_dataset():
    """Download AIME dataset and save as JSONL"""
    
    print("Downloading AIME dataset from Hugging Face...")
    
    try:
        # Load the dataset
        dataset = load_dataset("di-zhang-fdu/AIME_1983_2024")
        
        print(f"Dataset loaded successfully!")
        print(f"Dataset info: {dataset}")
        
        # Create output directory if it doesn't exist
        output_dir = Path("./dataset/aime_dataset")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as JSONL
        output_file = output_dir / "aime_1983_2024.jsonl"
        
        print(f"Saving dataset to {output_file}...")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for i, example in enumerate(dataset['train']):
                # Convert to JSON and write to file
                json_line = json.dumps(example, ensure_ascii=False)
                f.write(json_line + '\n')
                
                # Print progress every 100 examples
                if (i + 1) % 100 == 0:
                    print(f"Processed {i + 1} examples...")
        
        print(f"Dataset saved successfully to {output_file}")
        print(f"Total examples: {len(dataset['train'])}")
        
        # Print a few examples for verification
        print("\nFirst 3 examples:")
        for i in range(min(3, len(dataset['train']))):
            example = dataset['train'][i]
            print(f"\nExample {i+1}:")
            print(f"  ID: {example['ID']}")
            print(f"  Year: {example['Year']}")
            print(f"  Problem Number: {example['Problem Number']}")
            print(f"  Question: {example['Question'][:100]}...")
            print(f"  Answer: {example['Answer']}")
            print(f"  Part: {example['Part']}")
        
        return output_file
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return None

def main():
    """Main function"""
    print("=" * 50)
    print("AIME Dataset Downloader")
    print("=" * 50)
    
    output_file = download_aime_dataset()
    
    if output_file:
        print(f"\n✅ Success! Dataset saved to: {output_file}")
        print(f"📊 Dataset contains AIME problems from 1983-2024")
        print(f"📝 Format: JSONL (one JSON object per line)")
    else:
        print("\n❌ Failed to download dataset")

if __name__ == "__main__":
    main() 