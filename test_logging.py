import os
import logging
from utils.config import load_config

def test_file_logging():
    """Test file logging configuration"""
    config = load_config()
    
    if 'logging' not in config:
        print("No logging configuration found in config.yaml")
        return
        
    log_config = config['logging']
    if not log_config.get('log_to_file', False):
        print("File logging is disabled in config.yaml")
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
    
    # Create a logger for this test
    logger = logging.getLogger('TestLogging')
    
    # Log some test messages
    logger.info("File logging test - INFO message")
    logger.warning("File logging test - WARNING message")
    logger.error("File logging test - ERROR message")
    
    print(f"Test log messages written to {log_file}")
    print(f"You can view them with: cat {log_file}")

def test_wandb_integration():
    """Test wandb integration"""
    config = load_config()
    
    if 'logging' not in config:
        print("No logging configuration found in config.yaml")
        return
        
    log_config = config['logging']
    if not log_config.get('use_wandb', False):
        print("Wandb is disabled in config.yaml")
        return
    
    try:
        import wandb
        wandb_available = True
    except ImportError:
        print("Wandb is not installed. Run 'pip install wandb' to install it.")
        return
    
    # Initialize wandb
    wandb_project = log_config.get('wandb_project', 'sailing-rules-assistant')
    wandb_entity = log_config.get('wandb_entity', None)
    
    # Create a unique run name
    run_name = "test-logging-run"
    
    print(f"Initializing wandb with project: {wandb_project}, entity: {wandb_entity}, run: {run_name}")
    
    # Initialize wandb
    wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        name=run_name,
        config={
            "test": True,
            "purpose": "Testing wandb integration"
        }
    )
    
    # Log some test metrics
    wandb.log({"test_metric": 0.5, "test_accuracy": 0.85})
    
    print(f"Test metrics logged to wandb. View at: {wandb.run.get_url()}")
    
    # Finish the run
    wandb.finish()

if __name__ == "__main__":
    print("Testing logging configuration...")
    test_file_logging()
    print("\nTesting wandb integration...")
    test_wandb_integration()
    print("\nLogging tests complete.")