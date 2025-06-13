import numpy as np
# Import evaluate library with an alias to avoid circular imports
import evaluate as eval_lib
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from utils.config import load_config

class SailingRulesEvaluator:
    def __init__(self, model_path, config_path="config.yaml"):
        self.config = load_config(config_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.rouge = eval_lib.load("rouge")
        self.bleu = eval_lib.load("bleu")

    def prepare_test_data(self, test_dataset):
        """Prepare test data for evaluation"""
        test_samples = []
        for example in test_dataset:
            if example["input"]:
                input_text = f"Instruction: {example['instruction']}\nInput: {example['input']}"
            else:
                input_text = f"Instruction: {example['instruction']}"

            test_samples.append({
                "input": input_text,
                "reference": example["output"]
            })
        return test_samples

    def generate_predictions(self, test_samples):
        """Generate model predictions"""
        predictions = []
        references = []

        for sample in test_samples:
            input_ids = self.tokenizer(
                sample["input"],
                return_tensors="pt",
                truncation=True,
                max_length=self.config['training']['max_input_length']
            ).input_ids

            outputs = self.model.generate(
                input_ids,
                max_length=self.config['training']['max_target_length'],
                num_beams=4
            )

            prediction = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            predictions.append(prediction)
            references.append(sample["reference"])

        return predictions, references

    def compute_metrics(self, predictions, references):
        """Compute evaluation metrics"""
        # ROUGE scores
        rouge_output = self.rouge.compute(
            predictions=predictions,
            references=references,
            rouge_types=["rouge1", "rouge2", "rougeL"]
        )

        # BLEU score
        tokenized_predictions = [pred.split() for pred in predictions]
        tokenized_references = [[ref.split()] for ref in references]
        bleu_score = self.bleu.compute(
            predictions=tokenized_predictions,
            references=tokenized_references
        )

        results = {
            "rouge1": rouge_output["rouge1"].mid.fmeasure,
            "rouge2": rouge_output["rouge2"].mid.fmeasure,
            "rougeL": rouge_output["rougeL"].mid.fmeasure,
            "bleu": bleu_score["bleu"]
        }

        return results

    def evaluate(self, test_dataset):
        """Run evaluation on test dataset"""
        test_samples = self.prepare_test_data(test_dataset)
        predictions, references = self.generate_predictions(test_samples)
        metrics = self.compute_metrics(predictions, references)

        # Sample comparison
        sample_comparison = []
        for i in range(min(5, len(predictions))):
            sample_comparison.append({
                "input": test_samples[i]["input"],
                "reference": references[i],
                "prediction": predictions[i]
            })

        return metrics, sample_comparison
