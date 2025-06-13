from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from utils.config import load_config

class SailingRulesModel:
    def __init__(self, config_path="config.yaml"):
        self.config = load_config(config_path)

    def load_base_model(self):
        """Load the base Flan-T5 model"""
        model = AutoModelForSeq2SeqLM.from_pretrained(self.config['model']['base_model'])
        tokenizer = AutoTokenizer.from_pretrained(self.config['model']['base_model'])
        return model, tokenizer

    def create_peft_model(self, model):
        """Apply LoRA adapters to the model for efficient fine-tuning"""
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            inference_mode=False,
            r=self.config['model']['lora_r'],
            lora_alpha=self.config['model']['lora_alpha'],
            lora_dropout=self.config['model']['lora_dropout'],
            target_modules=["q", "v"]
        )

        model = get_peft_model(model, peft_config)
        return model

    def save_model(self, model, tokenizer, output_dir):
        """Save the fine-tuned model"""
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
