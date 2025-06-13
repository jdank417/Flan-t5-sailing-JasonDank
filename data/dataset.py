import os
import json
import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from utils.config import load_config

class SailingRulesDataset:
    def __init__(self, config_path="config.yaml"):
        self.config = load_config(config_path)

    def load_json_data(self):
        """
        Load training data from JSON files in the json_data folder.
        Each JSON file should contain an array of objects with the format:
        {
            "instruction": "What is the definition of 'zone' around a mark?",
            "input": "",
            "output": "The zone is the area around a mark within a distance of three hull lengths of the boat nearer to it. A boat is in the zone when any part of her hull is in that area."
        }
        """
        all_examples = []
        json_data_path = self.config["paths"]["json_data"]

        # Create the directory if it doesn't exist
        os.makedirs(json_data_path, exist_ok=True)

        # Check if directory exists and has files
        if not os.path.exists(json_data_path):
            print(f"Warning: JSON data directory {json_data_path} does not exist.")
            return all_examples

        # Iterate through all files in the json data directory
        for filename in os.listdir(json_data_path):
            if filename.lower().endswith('.json'):
                file_path = os.path.join(json_data_path, filename)
                print(f"Processing JSON file: {filename}")

                try:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        data = json.load(file)

                        # Handle both array and single object formats
                        if isinstance(data, list):
                            for item in data:
                                if self._validate_example(item):
                                    all_examples.append(item)
                        elif isinstance(data, dict):
                            if self._validate_example(data):
                                all_examples.append(data)
                        else:
                            print(f"Warning: Unexpected JSON format in {filename}")
                except json.JSONDecodeError:
                    print(f"Error: Invalid JSON format in {filename}")
                except Exception as e:
                    print(f"Error processing {filename}: {str(e)}")

        print(f"Loaded {len(all_examples)} examples from JSON files")
        return all_examples

    def _validate_example(self, example):
        """Validate that an example has the required fields"""
        required_fields = ["instruction", "input", "output"]

        if not all(field in example for field in required_fields):
            print(f"Warning: Example missing required fields: {example}")
            return False

        return True

    def process(self):
        """Process JSON data into training dataset"""
        # Load examples from JSON files
        all_examples = self.load_json_data()

        if not all_examples:
            print("No valid examples found. Please add JSON files to the data/json directory.")
            # Create an empty dataset to avoid errors
            empty_df = pd.DataFrame(columns=["instruction", "input", "output"])
            empty_dataset = Dataset.from_pandas(empty_df)
            empty_dict = DatasetDict({
                'train': empty_dataset,
                'validation': empty_dataset,
                'test': empty_dataset
            })
            return empty_dict

        # Convert to pandas DataFrame
        df = pd.DataFrame(all_examples)

        # Check if we should use 100% of data for training
        if self.config['data']['train_test_split'] == 0 and self.config['data']['validation_split'] == 0:
            # Use all data for training
            train = df
            val = pd.DataFrame(columns=df.columns)  # Empty DataFrame with same columns
            test = pd.DataFrame(columns=df.columns)  # Empty DataFrame with same columns
        else:
            # Split data normally
            train_val, test = train_test_split(
                df, test_size=self.config['data']['train_test_split'],
                random_state=self.config['data']['seed']
            )

            train, val = train_test_split(
                train_val, test_size=self.config['data']['validation_split'],
                random_state=self.config['data']['seed']
            )

        # Convert to Hugging Face datasets
        train_dataset = Dataset.from_pandas(train)
        val_dataset = Dataset.from_pandas(val)
        test_dataset = Dataset.from_pandas(test)

        dataset_dict = DatasetDict({
            'train': train_dataset,
            'validation': val_dataset,
            'test': test_dataset
        })

        # Save processed dataset
        dataset_dict.save_to_disk(self.config['paths']['processed_data'])

        # Print dataset statistics
        print("\n=== Dataset Statistics ===")
        print(f"Total examples: {len(df)}")
        print(f"Train set size: {len(train)}")
        print(f"Validation set size: {len(val)}")
        print(f"Test set size: {len(test)}")

        return dataset_dict
