import gradio as gr
from inference.predictor import SailingRulesPredictor
from utils.config import load_config

def main():
    config = load_config()
    predictor = SailingRulesPredictor(config['paths']['save_model'])

    def predict(question, context=""):
        """Process user input and return model prediction"""
        if context:
            input_text = f"Instruction: {question}\nInput: {context}"
        else:
            input_text = f"Instruction: {question}"

        answer = predictor.predict(input_text)
        return answer

    # Create Gradio interface
    demo = gr.Interface(
        fn=predict,
        inputs=[
            gr.Textbox(label="Question about Sailing Rules"),
            gr.Textbox(label="Context (optional)", placeholder="Describe the sailing scenario if relevant")
        ],
        outputs=gr.Textbox(label="Answer"),
        title="Sailing Rules Assistant (2025-2028)",
        description="Ask questions about the Racing Rules of Sailing (2025-2028 edition)",
        examples=[
            ["What are the basic right-of-way rules between boats?"],
            ["When is a boat clear ahead?"],
            ["What should I do if two boats are approaching on port and starboard tack?", "I'm on port tack approaching the windward mark."]
        ]
    )

    # Launch app
    demo.launch(server_name=config['inference']['host'], server_port=config['inference']['port'])

if __name__ == "__main__":
    main()
