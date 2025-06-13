from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from utils.config import load_config

class SailingRulesPredictor:
    def __init__(self, model_path):
        self.config = load_config()
        print(f"Loading model from {model_path}...")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def predict(self, input_text):
        """Generate prediction for input text"""
        input_ids = self.tokenizer(
            input_text,
            return_tensors="pt",
            max_length=self.config['training']['max_input_length'],
            truncation=True
        ).input_ids

        outputs = self.model.generate(
            input_ids,
            max_length=self.config['training']['max_target_length'],
            min_length=30,  # Increase from 20 to 30
            num_beams=12,   # Increase from 8 to 12
            temperature=0.2,  # Further decrease from 0.3 to 0.2
            no_repeat_ngram_size=4,  # Increase from 3 to 4
            length_penalty=1.0,  # Add length penalty to encourage longer responses
            early_stopping=True
        )

        prediction = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return prediction
