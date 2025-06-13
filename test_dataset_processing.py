import os
from data.dataset import SailingRulesDataset
from utils.config import load_config

def main():
    # Load configuration
    config = load_config()

    # Create directories if they don't exist
    os.makedirs(config['paths']['json_data'], exist_ok=True)
    os.makedirs(config['paths']['processed_data'], exist_ok=True)

    # Process dataset
    print("Processing dataset...")
    dataset_processor = SailingRulesDataset()
    dataset = dataset_processor.process()

    # Print dataset statistics
    print("\n=== Dataset Statistics ===")
    print(f"Train set size: {len(dataset['train'])}")
    print(f"Validation set size: {len(dataset['validation'])}")
    print(f"Test set size: {len(dataset['test'])}")

    # Print a few examples from the dataset
    print("\n=== Sample Examples ===")
    for split in ['train', 'validation', 'test']:
        if len(dataset[split]) > 0:
            example = dataset[split][0]
            print(f"\nExample from {split} set:")
            print(f"Instruction: {example['instruction']}")
            if example['input']:
                print(f"Input: {example['input']}")
            print(f"Output: {example['output']}")

    print("\nDataset processing complete. Ready for training.")

if __name__ == "__main__":
    main()
