import argparse
import yaml
import os
import models
import preprocessing
import datasets
import utils
import evaluation
from settings import Settings

def parse_args():
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    return parser.parse_args()

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def train(settings: Settings):
    # Add your training code here
    pass


def main():
    args = parse_args()
    settings = Settings(args.config)
    
    # Add your training code here
    print("Training with the following settings:")
    print(settings)
    
    train(settings)
    evaluation.Evaluate(settings)
    
if __name__ == "__main__":
    main()