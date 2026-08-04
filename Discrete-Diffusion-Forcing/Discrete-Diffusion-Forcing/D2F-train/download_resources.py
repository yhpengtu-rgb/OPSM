#!/usr/bin/env python3
"""
Download models and datasets for D2F training using HuggingFace mirror.
Uses HF_ENDPOINT to set Chinese mirror for faster downloads.
"""

import os
import sys

# Set HuggingFace mirror to Chinese mirror
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# Create local directories
LOCAL_DATA_DIR = os.path.join(os.path.dirname(__file__), 'local_data')
LOCAL_MODEL_DIR = os.path.join(os.path.dirname(__file__), 'local_models')
os.makedirs(LOCAL_DATA_DIR, exist_ok=True)
os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)

def download_dataset(dataset_name, save_dir):
    """Download a dataset from HuggingFace"""
    print(f"Downloading dataset: {dataset_name}")
    print(f"Mirror: {os.environ['HF_ENDPOINT']}")
    
    try:
        from datasets import load_dataset
        print("Loading dataset...")
        dataset = load_dataset(dataset_name, split="train")
        
        # Save to local directory
        dataset_save_path = os.path.join(save_dir, dataset_name.replace('/', '_'))
        os.makedirs(dataset_save_path, exist_ok=True)
        dataset.save_to_disk(dataset_save_path)
        print(f"✓ Dataset saved to: {dataset_save_path}")
        print(f"  Number of samples: {len(dataset)}")
        
        # Show first sample structure
        if len(dataset) > 0:
            print(f"  Features: {dataset.features}")
            print(f"  First sample keys: {list(dataset[0].keys())}")
        
        return dataset_save_path
    except Exception as e:
        print(f"✗ Failed to download dataset {dataset_name}: {e}")
        import traceback
        traceback.print_exc()
        return None

def download_model(model_name, save_dir):
    """Download a model from HuggingFace"""
    print(f"\nDownloading model: {model_name}")
    print(f"Mirror: {os.environ['HF_ENDPOINT']}")
    
    try:
        from transformers import AutoModel, AutoTokenizer, AutoConfig
        from huggingface_hub import snapshot_download
        
        model_save_path = os.path.join(save_dir, model_name.replace('/', '_'))
        os.makedirs(model_save_path, exist_ok=True)
        
        print("Downloading model snapshot...")
        local_path = snapshot_download(
            repo_id=model_name,
            local_dir=model_save_path,
            local_dir_use_symlinks=False
        )
        print(f"✓ Model saved to: {local_path}")
        
        # Verify the download
        config = AutoConfig.from_pretrained(local_path, trust_remote_code=True)
        print(f"  Model type: {config.model_type}")
        
        return local_path
    except Exception as e:
        print(f"✗ Failed to download model {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Download all required resources"""
    print("=" * 60)
    print("D2F Training Resource Downloader")
    print(f"Using HuggingFace Mirror: {os.environ['HF_ENDPOINT']}")
    print("=" * 60)
    
    # Datasets to download
    datasets_to_download = [
        'Lansechen/bs17k_collection_filtered_hard_maxlength600',
        'Lansechen/bs17k_collection_filtered_easy_maxlength600',
    ]
    
    # Models to download (these are large, so we'll try to download them)
    # For CPU testing, we might want to use smaller models
    models_to_download = [
        # 'GSAI-ML/LLaDA-8B-Instruct',  # 8B model - very large
        # 'Dream-org/Dream-v0-Base-7B',  # 7B model - very large
    ]
    
    downloaded_datasets = {}
    downloaded_models = {}
    
    # Download datasets
    print("\n" + "=" * 60)
    print("Downloading Datasets")
    print("=" * 60)
    
    for dataset_name in datasets_to_download:
        path = download_dataset(dataset_name, LOCAL_DATA_DIR)
        if path:
            downloaded_datasets[dataset_name] = path
    
    # Try to download models
    print("\n" + "=" * 60)
    print("Downloading Models")
    print("=" * 60)
    print("Note: Models are very large (7-8B parameters).")
    print("For CPU testing, you may want to skip this and use a small mock model.")
    
    for model_name in models_to_download:
        path = download_model(model_name, LOCAL_MODEL_DIR)
        if path:
            downloaded_models[model_name] = path
    
    # Print summary
    print("\n" + "=" * 60)
    print("Download Summary")
    print("=" * 60)
    
    print("\nDatasets:")
    for name, path in downloaded_datasets.items():
        print(f"  {name} -> {path}")
    
    print("\nModels:")
    for name, path in downloaded_models.items():
        print(f"  {name} -> {path}")
    
    # Create a small test dataset for CPU testing
    print("\n" + "=" * 60)
    print("Creating Small Test Dataset for CPU Testing")
    print("=" * 60)
    
    try:
        from datasets import Dataset
        
        # Create a small test dataset with simple math problems
        test_data = {
            'question': [
                'What is 2 + 3?',
                'A farmer has 10 apples. He gives 4 to his friend. How many apples does he have left?',
                'If a train travels at 60 km/h for 2 hours, how far does it go?',
                'What is the square root of 144?',
                'A rectangle has length 8 and width 5. What is its area?',
            ],
            'qwen7b_answer': [
                '<|begin_of_thought|>\n2 + 3 = 5\n<|end_of_thought|>\n<|begin_of_solution|>\n5\n<|end_of_solution|>',
                '<|begin_of_thought|>\nFarmer has 10 apples, gives away 4. 10 - 4 = 6\n<|end_of_thought|>\n<|begin_of_solution|>\n6\n<|end_of_solution|>',
                '<|begin_of_thought|>\nDistance = speed × time = 60 × 2 = 120 km\n<|end_of_thought|>\n<|begin_of_solution|>\n120 km\n<|end_of_solution|>',
                '<|begin_of_thought|>\n√144 = 12 because 12 × 12 = 144\n<|end_of_thought|>\n<|begin_of_solution|>\n12\n<|end_of_solution|>',
                '<|begin_of_thought|>\nArea = length × width = 8 × 5 = 40\n<|end_of_thought|>\n<|begin_of_solution|>\n40\n<|end_of_solution|>',
            ]
        }
        
        test_dataset = Dataset.from_dict(test_data)
        test_dataset_path = os.path.join(LOCAL_DATA_DIR, 'test_small_dataset')
        test_dataset.save_to_disk(test_dataset_path)
        print(f"✓ Small test dataset saved to: {test_dataset_path}")
        print(f"  Number of samples: {len(test_dataset)}")
        
    except Exception as e:
        print(f"✗ Failed to create test dataset: {e}")
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
    
    # Print usage instructions
    print("\nTo use local paths in config:")
    print("  paths.model: '/path/to/local/model'")
    print("  paths.data.bs: '/path/to/local/dataset'")
    print("\nFor CPU testing with small mock model, use:")
    print("  python test_on_policy_cpu.py")

if __name__ == '__main__':
    main()