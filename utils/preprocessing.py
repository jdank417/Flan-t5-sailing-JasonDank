import re
import os
import pandas as pd
from utils.config import load_config

class TextPreprocessor:
    """Class to preprocess sailing rules text documents"""

    def __init__(self, config_path="config.yaml"):
        self.config = load_config(config_path)

    def clean_text(self, text):
        """Clean and normalize text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Standardize newlines
        text = re.sub(r'\n+', '\n', text)
        return text.strip()

    def extract_rules(self, text):
        """Extract individual rules from the complete rulebook text
        This is a placeholder - implementation will depend on actual rule format
        """
        # This function would parse the sailing rules text and extract
        # individual rules with their numbers, titles, and content
        rules = []

        # Placeholder for rule extraction logic
        # Example: rules.append({"number": "10.1", "title": "On Opposite Tacks", "content": "..."})

        return rules

    def extract_definitions(self, text):
        """Extract sailing term definitions from the rulebook
        This is a placeholder - implementation will depend on actual format
        """
        definitions = {}

        # Placeholder for definition extraction logic
        # Example: definitions["leeward"] = "The side of the boat that is or, when she is head to wind, was away from the wind."

        return definitions

    def save_processed_data(self, data, filename):
        """Save processed data to file"""
        output_path = os.path.join(self.config["paths"]["processed_data"], filename)

        if isinstance(data, pd.DataFrame):
            data.to_csv(output_path, index=False)
        elif isinstance(data, dict) or isinstance(data, list):
            pd.DataFrame(data).to_csv(output_path, index=False)

        print(f"Saved processed data to {output_path}")
