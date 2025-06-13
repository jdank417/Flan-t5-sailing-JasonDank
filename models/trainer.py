import os
import logging
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)
from utils.config import load_config

# Import wandb conditionally to avoid errors if not installed
try:
    import wandb
    wandb_available = True
except ImportError:
    wandb_available = False

class SailingRulesTrainer:
    def __init__(self, model, tokenizer, dataset, config_path="config.yaml"):
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.config = load_config(config_path)

        # Set up logging
        self._setup_logging()

        # Initialize wandb if enabled
        self._init_wandb()

    def _setup_logging(self):
        """Set up logging to file and console"""
        if 'logging' not in self.config:
            return

        log_config = self.config['logging']
        if not log_config.get('log_to_file', False):
            return

        # Get log level
        log_level_str = log_config.get('log_level', 'INFO')
        log_level = getattr(logging, log_level_str, logging.INFO)

        # Configure root logger
        log_file = log_config.get('log_file', 'training.log')
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()  # Also log to console
            ]
        )

        # Create a logger for this class
        self.logger = logging.getLogger('SailingRulesTrainer')
        self.logger.info("Logging initialized. Writing to %s", log_file)

    def _init_wandb(self):
        """Initialize Weights & Biases for experiment tracking"""
        if 'logging' not in self.config:
            return

        log_config = self.config['logging']
        if not log_config.get('use_wandb', False) or not wandb_available:
            if log_config.get('use_wandb', False) and not wandb_available:
                print("Warning: wandb is enabled in config but not installed. Run 'pip install wandb' to install it.")
            return

        # Initialize wandb
        wandb_project = log_config.get('wandb_project', 'sailing-rules-assistant')
        wandb_entity = log_config.get('wandb_entity', None)

        # Create a unique run name based on model and training parameters
        run_name = f"flan-t5-{self.config['model']['lora_r']}-epochs-{self.config['training']['epochs']}"

        # Initialize wandb
        wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=run_name,
            config={
                "model": self.config['model'],
                "training": self.config['training'],
                "data": self.config['data']
            }
        )

        self.logger.info(f"Weights & Biases initialized. Project: {wandb_project}, Run: {run_name}")
        print(f"Weights & Biases initialized. Project: {wandb_project}, Run: {run_name}")
        print(f"Track your training at: {wandb.run.get_url()}")

    def preprocess_function(self, examples):
        """Preprocess the data for training"""
        inputs = []
        targets = []

        for instruction, input_text, output in zip(
            examples["instruction"],
            examples["input"],
            examples["output"]
        ):
            if input_text:
                # Format with instruction and input
                prompt = f"Instruction: {instruction}\nInput: {input_text}"
            else:
                # Format with just instruction
                prompt = f"Instruction: {instruction}"

            inputs.append(prompt)
            targets.append(output)

        model_inputs = self.tokenizer(
            inputs,
            max_length=self.config['training']['max_input_length'],
            padding="max_length",
            truncation=True
        )

        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                targets,
                max_length=self.config['training']['max_target_length'],
                padding="max_length",
                truncation=True
            )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def prepare_dataset(self):
        """Prepare dataset for training"""
        processed_datasets = self.dataset.map(
            self.preprocess_function,
            batched=True,
            remove_columns=self.dataset["train"].column_names
        )
        return processed_datasets

    def train(self):
        """Train the model"""
        if hasattr(self, 'logger'):
            self.logger.info("Starting training preparation")
        else:
            print("Starting training preparation")

        processed_datasets = self.prepare_dataset()

        # Data collator
        data_collator = DataCollatorForSeq2Seq(
            self.tokenizer,
            model=self.model,
            return_tensors="pt"
        )

        # Check if we're running on a device that supports fp16
        import torch
        device_type = torch.device("cuda" if torch.cuda.is_available() else "cpu").type
        fp16_supported = device_type == "cuda"

        if hasattr(self, 'logger'):
            self.logger.info(f"Using device: {device_type}")
        else:
            print(f"Using device: {device_type}")

        # Ensure numeric values are properly converted
        learning_rate = float(self.config['training']['learning_rate'])
        weight_decay = float(self.config['training']['weight_decay'])

        # Calculate warmup steps from ratio if provided
        warmup_steps = self.config['training'].get('warmup_steps', 0)
        if 'warmup_ratio' in self.config['training']:
            # Calculate total steps based on dataset size, batch size, and epochs
            total_steps = (
                len(processed_datasets["train"]) 
                // (self.config['training']['batch_size'] * self.config['training']['gradient_accumulation_steps'])
                * self.config['training']['epochs']
            )
            warmup_steps = int(total_steps * float(self.config['training']['warmup_ratio']))

        # Determine if we should use wandb
        use_wandb = (
            'logging' in self.config and 
            self.config['logging'].get('use_wandb', False) and 
            wandb_available
        )

        # Set up logging callbacks
        callbacks = []

        # Add wandb callback if enabled
        if use_wandb:
            from transformers.integrations import WandbCallback
            callbacks.append(WandbCallback())

            # Log dataset info to wandb
            wandb.log({
                "train_dataset_size": len(processed_datasets["train"]),
                "validation_dataset_size": len(processed_datasets["validation"]),
                "test_dataset_size": len(processed_datasets["test"]),
            })

        # Training arguments
        training_args = Seq2SeqTrainingArguments(
            output_dir=self.config['paths']['save_model'],
            learning_rate=learning_rate,
            per_device_train_batch_size=self.config['training']['batch_size'],
            per_device_eval_batch_size=self.config['training']['batch_size'],
            weight_decay=weight_decay,
            save_total_limit=3,
            num_train_epochs=self.config['training']['epochs'],
            predict_with_generate=True,
            fp16=(self.config['training']['mixed_precision'] == "fp16") and fp16_supported,
            gradient_accumulation_steps=self.config['training']['gradient_accumulation_steps'],
            warmup_steps=warmup_steps,  # Use calculated warmup steps
            lr_scheduler_type="cosine",  # Add cosine learning rate scheduler
            # Add logging settings
            logging_dir="logs",
            logging_strategy="steps",
            logging_steps=10,
            report_to=["wandb"] if use_wandb else [],
            disable_tqdm=False,  # Explicitly enable the progress bar
        )

        if not fp16_supported and self.config['training']['mixed_precision'] == "fp16":
            warning_msg = "Warning: fp16 mixed precision is not supported on this device. Training with fp32 instead."
            if hasattr(self, 'logger'):
                self.logger.warning(warning_msg)
            else:
                print(warning_msg)

        # Check if validation set is empty
        has_validation = len(processed_datasets["validation"]) > 0

        validation_msg = "Using validation dataset" if has_validation else "Validation dataset is empty. Training without validation."
        if hasattr(self, 'logger'):
            self.logger.info(validation_msg)
        else:
            print(validation_msg)

        # Initialize trainer
        trainer_kwargs = {
            "model": self.model,
            "args": training_args,
            "train_dataset": processed_datasets["train"],
            "tokenizer": self.tokenizer,
            "data_collator": data_collator,
            "callbacks": callbacks,  # Add our callbacks
        }

        if has_validation:
            trainer_kwargs["eval_dataset"] = processed_datasets["validation"]

        trainer = Seq2SeqTrainer(**trainer_kwargs)

        # Log training start
        train_start_msg = f"Starting training with {len(processed_datasets['train'])} examples for {self.config['training']['epochs']} epochs"
        if hasattr(self, 'logger'):
            self.logger.info(train_start_msg)
        else:
            print(train_start_msg)

        # Print info about where to find logs
        print("\n" + "="*50)
        print("TRAINING STARTED")
        print("="*50)
        print(f"Training logs are being saved to: {self.config['logging'].get('log_file', 'training.log')}")
        if use_wandb:
            print(f"Track training progress in real-time at: {wandb.run.get_url()}")
        print("You can safely close this terminal - training will continue in the background.")
        print("="*50 + "\n")

        # Train
        try:
            trainer.train()

            # Log training completion
            train_complete_msg = "Training completed successfully"
            if hasattr(self, 'logger'):
                self.logger.info(train_complete_msg)
            else:
                print(train_complete_msg)

        except Exception as e:
            # Log training error
            train_error_msg = f"Training failed with error: {str(e)}"
            if hasattr(self, 'logger'):
                self.logger.error(train_error_msg)
            else:
                print(train_error_msg)
            raise

        # Save model
        save_model_msg = f"Saving model to {self.config['paths']['save_model']}"
        if hasattr(self, 'logger'):
            self.logger.info(save_model_msg)
        else:
            print(save_model_msg)

        trainer.save_model(self.config['paths']['save_model'])
        self.tokenizer.save_pretrained(self.config['paths']['save_model'])

        # Close wandb run if it was used
        if use_wandb:
            wandb.finish()

        # Final message
        final_msg = f"Training complete. Model saved to {self.config['paths']['save_model']}"
        if hasattr(self, 'logger'):
            self.logger.info(final_msg)
        else:
            print(final_msg)

        return trainer
