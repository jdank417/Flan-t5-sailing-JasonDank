import os
from datasets import load_from_disk
from models.model import SailingRulesModel
from models.trainer import SailingRulesTrainer
from data.dataset import SailingRulesDataset
from utils.config import load_config

def main():
    # Load configuration
    config = load_config()

    # Create directories if they don't exist
    os.makedirs(config['paths']['json_data'], exist_ok=True)
    os.makedirs(config['paths']['processed_data'], exist_ok=True)
    os.makedirs(config['paths']['save_model'], exist_ok=True)

    # Process dataset if not already processed
    if not os.path.exists(config['paths']['processed_data']):
        print("Processing dataset...")
        dataset_processor = SailingRulesDataset()
        dataset = dataset_processor.process()
    else:
        print("Loading processed dataset...")
        dataset = load_from_disk(config['paths']['processed_data'])

    # Initialize model
    print("Initializing model...")
    model_handler = SailingRulesModel()
    base_model, tokenizer = model_handler.load_base_model()
    peft_model = model_handler.create_peft_model(base_model)

    # Print model trainable parameters
    trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in peft_model.parameters())
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/all_params:.2%} of {all_params:,} total)")

    # Train model
    print("Starting training...")
    trainer = SailingRulesTrainer(peft_model, tokenizer, dataset)
    trainer.train()

    print(f"Training complete. Model saved to {config['paths']['save_model']}")

if __name__ == "__main__":
    main()
