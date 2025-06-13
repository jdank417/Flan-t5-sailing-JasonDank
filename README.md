# Sailing Rules Assistant

A fine-tuned Flan-T5 model for answering questions about sailing rules.

## Project Overview

This project fine-tunes a Flan-T5 model on sailing rules data to create an assistant that can answer questions about sailing rules and interpret racing scenarios.

## Data Format

The model is trained on JSON data in the following format:

```json
{
  "instruction": "What is the definition of 'zone' around a mark?",
  "input": "",
  "output": "The zone is the area around a mark within a distance of three hull lengths of the boat nearer to it. A boat is in the zone when any part of her hull is in that area."
}
```

Each training example consists of:
- `instruction`: The question or instruction
- `input`: Optional context or scenario description (can be empty string)
- `output`: The expected answer or response

## Directory Structure

- `config.yaml`: Configuration file for model, training, and paths
- `data/`: Directory containing dataset-related code and data
  - `dataset.py`: Code for processing JSON data into training datasets
  - `json/`: Directory for JSON training data files
  - `processed/`: Directory for processed datasets
- `models/`: Directory containing model-related code
  - `model.py`: Code for loading and creating the model
  - `trainer.py`: Code for training the model
- `inference/`: Directory containing inference-related code
  - `app.py`: Gradio app for interactive testing
  - `predictor.py`: Code for making predictions with the trained model
- `utils/`: Directory containing utility functions
- `train.py`: Script for training the model
- `evaluate.py`: Script for evaluating the model
- `test_dataset_processing.py`: Script for testing dataset processing

## Setup and Usage

1. **Install dependencies**:
   ```
   pip install -r requirements.txt
   ```

2. **Prepare your data**:
   - Create JSON files with training examples in the format described above
   - Place the JSON files in the `data/json/` directory

3. **Process the dataset**:
   ```
   python test_dataset_processing.py
   ```

4. **Train the model**:
   ```
   python train.py
   ```

5. **Evaluate the model**:
   ```
   python evaluate.py
   ```

6. **Interactive testing**:
   ```
   python -m inference.app
   ```

## Monitoring Training Progress

This project includes two methods for monitoring training progress:

### File Logging

Training progress is automatically logged to a file, which allows you to monitor progress even if your terminal session times out or disconnects:

- Logs are saved to `training.log` by default
- You can view logs in real-time using: `tail -f training.log`
- Log files contain detailed information about training progress, including steps, loss values, and evaluation metrics

### Weights & Biases Integration

For more advanced monitoring, the project integrates with [Weights & Biases](https://wandb.ai/):

1. **Setup**:
   - Create a free account at [wandb.ai](https://wandb.ai/) if you don't have one
   - Login to wandb: `wandb login`

2. **Configuration**:
   - Enable wandb in `config.yaml` by setting `logging.use_wandb` to `true`
   - Set your wandb username or team name in `logging.wandb_entity` (optional)
   - Customize the project name in `logging.wandb_project` if desired

3. **Benefits**:
   - Real-time monitoring of training metrics through a web interface
   - Automatic tracking of model parameters and performance
   - Visualizations of training progress
   - Experiment comparison
   - Training can be monitored from any device with internet access

With these monitoring tools, you can safely start training and close your terminal - the training will continue in the background and you can monitor progress through logs or wandb.

## Configuration

You can modify the `config.yaml` file to adjust:
- Model parameters (base model, LoRA settings)
- Training parameters (epochs, batch size, learning rate)
- Data parameters (train/test split, validation split)
- Paths for data and models
- Logging settings (file logging, wandb integration)

## License

[Your license information here]
