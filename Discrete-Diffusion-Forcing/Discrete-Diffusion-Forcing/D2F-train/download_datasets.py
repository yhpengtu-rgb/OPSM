#!/usr/bin/env python3
"""
Download datasets for D2F training using HuggingFace mirror.
Uses HF_ENDPOINT to set Chinese mirror for faster downloads.
"""

import os
import sys

# Set HuggingFace mirror to Chinese mirror
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# Create local directories
LOCAL_DATA_DIR = os.path.join(os.path.dirname(__file__), 'local_data')
os.makedirs(LOCAL_DATA_DIR, exist_ok=True)

def download_dataset(dataset_name, save_dir):
    """Download a dataset from HuggingFace"""
    print(f"Downloading dataset: {dataset_name}")
    print(f"Mirror: {os.environ['HF_ENDPOINT']}")
    
    try:
        from datasets import load_dataset
        
        # Load dataset with streaming first to check size
        print("Loading dataset...")
        dataset = load_dataset(dataset_name, split="train", streaming=True)
        
        # Count samples
        count = 0
        for _ in dataset:
            count += 1
            if count % 1000 == 0:
                print(f"  Processed {count} samples...")
        
        print(f"  Total samples: {count}")
        
        # Load full dataset
        dataset = load_dataset(dataset_name, split="train")
        
        # Save to local directory
        dataset_save_path = os.path.join(save_dir, dataset_name.replace('/', '_'))
        os.makedirs(dataset_save_path, exist_ok=True)
        dataset.save_to_disk(dataset_save_path)
        print(f"✓ Dataset saved to: {dataset_save_path}")
        print(f"  Number of samples: {len(dataset)}")
        print(f"  Features: {dataset.features}")
        
        return dataset_save_path
    except Exception as e:
        print(f"✗ Failed to download dataset {dataset_name}: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_test_dataset(save_dir):
    """Create a small test dataset for quick CPU testing"""
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
                'John has 25 dollars. He buys a book for 12 dollars. How much money does he have left?',
                'What is 15% of 200?',
                'The sum of three consecutive numbers is 72. What are the numbers?',
                'A circle has a radius of 7 cm. What is its area? (Use π = 3.14)',
                'If 5 workers can build a wall in 12 days, how many days will it take 10 workers?',
            ],
            'qwen7b_answer': [
                '<|begin_of_thought|>\n2 + 3 = 5\n<|end_of_thought|>\n<|begin_of_solution|>\n5\n<|end_of_solution|>',
                '<|begin_of_thought|>\nFarmer has 10 apples, gives away 4. 10 - 4 = 6\n<|end_of_thought|>\n<|begin_of_solution|>\n6\n<|end_of_solution|>',
                '<|begin_of_thought|>\nDistance = speed × time = 60 × 2 = 120 km\n<|end_of_thought|>\n<|begin_of_solution|>\n120 km\n<|end_of_solution|>',
                '<|begin_of_thought|>\n√144 = 12 because 12 × 12 = 144\n<|end_of_thought|>\n<|begin_of_solution|>\n12\n<|end_of_solution|>',
                '<|begin_of_thought|>\nArea = length × width = 8 × 5 = 40\n<|end_of_thought|>\n<|begin_of_solution|>\n40\n<|end_of_solution|>',
                '<|begin_of_thought|>\nJohn starts with 25 dollars and spends 12. 25 - 12 = 13\n<|end_of_thought|>\n<|begin_of_solution|>\n13 dollars\n<|end_of_solution|>',
                '<|begin_of_thought|>\n15% of 200 = 0.15 × 200 = 30\n<|end_of_thought|>\n<|begin_of_solution|>\n30\n<|end_of_solution|>',
                '<|begin_of_thought|>\nLet the three consecutive numbers be x, x+1, x+2. Their sum is 3x + 3 = 72, so 3x = 69, x = 23. The numbers are 23, 24, 25.\n<|end_of_thought|>\n<|begin_of_solution|>\n23, 24, 25\n<|end_of_solution|>',
                '<|begin_of_thought|>\nArea = π × r² = 3.14 × 7² = 3.14 × 49 = 153.86 cm²\n<|end_of_thought|>\n<|begin_of_solution|>\n153.86 cm²\n<|end_of_solution|>',
                '<|begin_of_thought|>\nMore workers means fewer days. 5 workers × 12 days = 60 worker-days. 60 worker-days ÷ 10 workers = 6 days\n<|end_of_thought|>\n<|begin_of_solution|>\n6 days\n<|end_of_solution|>',
            ]
        }
        
        test_dataset = Dataset.from_dict(test_data)
        test_dataset_path = os.path.join(save_dir, 'test_small_dataset')
        test_dataset.save_to_disk(test_dataset_path)
        print(f"✓ Small test dataset saved to: {test_dataset_path}")
        print(f"  Number of samples: {len(test_dataset)}")
        
        return test_dataset_path
        
    except Exception as e:
        print(f"✗ Failed to create test dataset: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Download all required resources"""
    print("=" * 60)
    print("D2F Training Resource Downloader")
    print(f"Using HuggingFace Mirror: {os.environ['HF_ENDPOINT']}")
    print("=" * 60)
    
    # First, create test dataset for quick CPU testing
    test_dataset_path = create_test_dataset(LOCAL_DATA_DIR)
    
    # Try to download the actual datasets
    print("\n" + "=" * 60)
    print("Downloading Actual Datasets")
    print("=" * 60)
    
    datasets_to_download = [
        'Lansechen/bs17k_collection_filtered_hard_maxlength600',
        # 'Lansechen/bs17k_collection_filtered_easy_maxlength600',  # Uncomment if needed
    ]
    
    downloaded = {}
    
    for dataset_name in datasets_to_download:
        print(f"\nTrying to download: {dataset_name}")
        path = download_dataset(dataset_name, LOCAL_DATA_DIR)
        if path:
            downloaded[dataset_name] = path
        else:
            print(f"  Skipping (may not be accessible)")
    
    # Print summary
    print("\n" + "=" * 60)
    print("Download Summary")
    print("=" * 60)
    
    print(f"\nTest dataset (always available):")
    print(f"  -> {test_dataset_path}")
    
    print(f"\nDownloaded datasets:")
    for name, path in downloaded.items():
        print(f"  {name} -> {path}")
    
    # Create a README with usage instructions
    print("\n" + "=" * 60)
    print("To use local datasets in config:")
    print(f"  paths.data.bs: '{test_dataset_path}'")
    print("=" * 60)

if __name__ == '__main__':
    main()