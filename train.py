import argparse
import yaml
import os
import models
from preprocessing import Embedding
from datasets import Dataset
import utils
from evaluation import Evaluation
import torch
from torch.optim import *
from settings import Settings
from datasets import Dataset

def parse_args():
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument('--config', type=str, required=True, help='Path to the config file', default=r'C:\Users\Marta\Desktop\Dipl_code\Diplomski\experiments\default_train.yml')    
    parser.add_argument('--model_name', type=str, required=True, help='Name of the file used for saving model weights', default='bilstm_crf')    
    return parser.parse_args()

def define_optimizer(model, name, lr):
    if name == 'adam':
        return Adam(model.parameters(), lr=lr)
    elif name == 'adamw':
        return AdamW(model.parameters(), lr=lr)
    elif name == 'sgd':
        return SGD(model.parameters(), lr=lr) #, momentum=0.9)
    # Add more optimizers here if nessesary
    else:
        raise ValueError(f"Optimizer {name} not supported")

def train(model_name, model_args, settings_args, eval):
    # TODO: napisati kod u datasets, preprocessing, ealuation i prepraviti stvari u models
    dataset = Dataset(settings_args['dataset'], Settings.DATA_PATH)
    num_tags = dataset.num_tags
    enc_tag = dataset.enc_tag # ne znam sta je tocno ovo
    train_dataset, valid_dataset, test_dataset = dataset.load_data()

    # Create a model
    embedding = Embedding(settings_args['embedding'], dataset.tokenizer) #TODO: prepraviti argumente ako treba
    model = models.BiRNN_CRF(num_tags, model_args, embedding)

    num_epochs = model_args['epochs']
    device = model_args['device']
    optimizer = define_optimizer(model, model_args['optimizer'], model_args['lr'])

    model.to(device)
    #TODO: add early stopping
    data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=model_args['batch_size'], shuffle=True)
    valid_data_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=model_args['batch_size'], shuffle=False)

    for epoch in range(num_epochs):
        model.train()
        final_loss = 0
        for data in data_loader: # tqdm(data_loader, total=len(data_loader)):
            for k, v in data.items(): 
                data[k] = v.to(device)
            optimizer.zero_grad()
            loss = model(**data)
            loss.backward()
            optimizer.step()
            final_loss += loss.item()
        train_loss = final_loss / len(data_loader)
        torch.cuda.empty_cache()

        valid_loss = eval.evaluate(valid_data_loader, model, device) 
        torch.cuda.empty_cache()

        print(f"Train Loss ({epoch}/{num_epochs}) = {train_loss}")
        print(f"Validation Loss ({epoch}/{num_epochs}) = {valid_loss}")

        eval.evaluate(test_dataset, model, device,enc_tag)
        torch.save(model.state_dict(), Settings.MODEL_PATH +f"{model_name}_{epoch}.bin")


def main():
    args = parse_args()
    model_name = args.model_name
    model_args, settings_args = Settings(args.config)
    
    # Add your training code here
    print("Training with the following settings:")
    print("Model Args:", model_args)
    print("Settings Args:", settings_args)
    
    eval = Evaluation(settings_args)
    train(model_name, model_args, settings_args, eval)
    eval.evaluate(model_args, settings_args)
    
if __name__ == "__main__":
    main()