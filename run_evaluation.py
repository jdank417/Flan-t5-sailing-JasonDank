import json
from datasets import load_from_disk
from utils.evaluation import SailingRulesEvaluator
from utils.config import load_config

def main():
    # Load configuration
    config = load_config()

    # Load test dataset
    dataset = load_from_disk(config['paths']['processed_data'])
    test_dataset = dataset["test"]

    # Check if test dataset is empty
    if len(test_dataset) == 0:
        print("\n=== Warning: Empty Test Dataset ===")
        print("The test dataset is empty because 100% of data was used for training.")
        print("To evaluate the model, you need to:")
        print("1. Create a separate test dataset, or")
        print("2. Modify config.yaml to allocate some data for testing.")
        print("\nEvaluation skipped.")
        return

    # Initialize evaluator
    evaluator = SailingRulesEvaluator(config['paths']['save_model'])

    # Evaluate model
    print("Evaluating model...")
    metrics, sample_comparison = evaluator.evaluate(test_dataset)

    # Print metrics
    print("\n=== Evaluation Metrics ===")
    print(f"ROUGE-1: {metrics['rouge1']:.4f}")
    print(f"ROUGE-2: {metrics['rouge2']:.4f}")
    print(f"ROUGE-L: {metrics['rougeL']:.4f}")
    print(f"BLEU: {metrics['bleu']:.4f}")

    # Print sample comparisons
    print("\n=== Sample Comparisons ===")
    for i, sample in enumerate(sample_comparison):
        print(f"\nExample {i+1}")
        print(f"Input: {sample['input']}")
        print(f"Reference: {sample['reference']}")
        print(f"Prediction: {sample['prediction']}")

    # Save results
    with open("evaluation_results.json", "w") as f:
        json.dump({
            "metrics": metrics,
            "samples": sample_comparison
        }, f, indent=2)

    print("\nEvaluation complete. Results saved to evaluation_results.json")

if __name__ == "__main__":
    main()