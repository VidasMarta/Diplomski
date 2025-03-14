import argparse
import copy
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
from torch.nn.utils import clip_grad_norm_

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
    
def train_one_epoch(model, data_loader, optimizer, device, max_grad_norm):
    model.train()
    final_loss = 0
    for data in data_loader: # tqdm(data_loader, total=len(data_loader)):
        for k, v in data.items(): 
            data[k] = v.to(device)
        optimizer.zero_grad()
        loss = model(**data)
        loss.backward()
        if max_grad_norm is not None:
            clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        final_loss += loss.item()
    return final_loss / len(data_loader)

def validate(model, data_loader, device):
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        final_loss = 0
        for data in data_loader: # tqdm(data_loader, total=len(data_loader)):
            for k, v in data.items():
                data[k] = v.to(device)
            loss = model(**data)
            final_loss += loss.item()
    return final_loss / len(data_loader)


def train(model_name, model_args, settings_args, dataset, train_dataset, valid_dataset):
    num_tags = dataset.num_tags
    max_grad_norm = model_args['max_grad_norm']
    # Create a model
    embedding = Embedding(settings_args['embedding'], dataset.tokenizer) #TODO: prepraviti argumente ako treba
    model = models.BiRNN_CRF(num_tags, model_args, embedding)

    num_epochs = model_args['epochs']
    device = model_args['device']
    optimizer = define_optimizer(model, model_args['optimizer'], model_args['lr'])

    model.to(device)
    data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=model_args['batch_size'], shuffle=True)
    valid_data_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=model_args['batch_size'], shuffle=False)

    #Initialize Variables for EarlyStopping
    best_loss = float('inf')
    best_model_weights = None
    patience = model_args['early_stopping']
    epochs_no_improve = 0
    min_delta = model_args['min_delta']

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, data_loader, optimizer, device, max_grad_norm)
        torch.cuda.empty_cache()

        # Validation
        val_loss = validate(model, valid_data_loader, device)
        torch.cuda.empty_cache()

        # Early stopping
        if val_loss + min_delta < best_loss:
            best_loss = val_loss
            epochs_no_improve = 0
            best_model_weights = copy.deepcopy(model.state_dict())  # Deep copy here      
            print(f"Validation loss improved to {best_loss:.4f}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

        print(f"Train Loss ({epoch}/{num_epochs}) = {train_loss}")
        print(f"Validation Loss ({epoch}/{num_epochs}) = {val_loss}")

    # Save the best model
    best_model = models.BiRNN_CRF(num_tags, model_args, embedding)
    best_model.load_state_dict(best_model_weights)
    torch.save(best_model.state_dict(), Settings.MODEL_PATH +f"{model_name}_best.bin")


def main():
    args = parse_args()
    model_name = args.model_name
    model_args, settings_args = Settings(args.config)
    
    # Add your training code here
    print("Training with the following settings:")
    print("Model Args:", model_args)
    print("Settings Args:", settings_args)
    
    eval = Evaluation(settings_args)
    # TODO: napisati kod u datasets, preprocessing, evaluation i prepraviti stvari u models
    dataset = Dataset(settings_args['dataset'], Settings.DATA_PATH)
    enc_tag = dataset.enc_tag # ne znam sta je tocno ovo
    train_dataset, valid_dataset, test_dataset = dataset.load_data()
    train(model_name, model_args, settings_args, dataset, train_dataset, valid_dataset)

    #Evaluate on test set
    model = torch.load(Settings.MODEL_PATH + f"{model_name}_best.bin")
    eval.evaluate(test_dataset, model, model_args['device'], enc_tag)
    
if __name__ == "__main__":
    main()