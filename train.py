import argparse
import copy
import yaml
import os
import models
from preprocessing import Embedding
from datasets import Dataset
import utils.dataset_converter as dataset_converter
from evaluation import Evaluation
import torch
from torch.optim import *
import settings
from datasets import *
from torch.nn.utils import clip_grad_norm_
import numpy as np
from utils.logger import LossLogger

def parse_args():
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument('--config', type=str, required=False, help='Path to the config file', default='/home/martavidas/Documents/FER/Diplomski/Diplomski/experiments/default_train.yml')    
    parser.add_argument('--model_name', type=str, required=False, help='Name of the file used for saving model weights', default='bilstm_crf')    
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
    
def train_one_epoch(model, data_loader, embeddings_model, optimizer, device, max_grad_norm):
    model.train()
    final_loss = 0
    for tokens, tags, att_mask in data_loader: # tqdm(data_loader, total=len(data_loader)):
        optimizer.zero_grad()

        batch_embeddings, batch_attention_masks = embeddings_model.get_embedding(tokens, att_mask)
        batch_embeddings = batch_embeddings.to(device)
        batch_attention_masks = batch_attention_masks.to(device)

        loss = model(batch_embeddings, tags, batch_attention_masks)
        loss.backward()
        if max_grad_norm is not None:
            clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()
        final_loss += loss.item()
    return final_loss / len(data_loader)

def validate(model, data_loader, embeddings_model, device):
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        final_loss = 0
        for tokens, tags, att_mask in data_loader: # tqdm(data_loader, total=len(data_loader)):
            batch_embeddings, batch_attention_masks = embeddings_model.get_embedding(tokens, att_mask)
            batch_embeddings = batch_embeddings.to(device)
            batch_attention_masks = batch_attention_masks.to(device)

            loss = model(batch_embeddings, tags, batch_attention_masks)
            final_loss += loss.item()
    return final_loss / len(data_loader)


def train(model_name, model_args, num_tags, train_dataset, valid_dataset, embeddings_model, device):
    max_grad_norm = model_args['max_grad_norm']
    # Create models
    model = models.BiRNN_CRF(num_tags, model_args, embeddings_model.embedding_dim)
    best_model = models.BiRNN_CRF(num_tags, model_args, embeddings_model.embedding_dim)

    num_epochs = model_args['epochs']
    optimizer = define_optimizer(model, model_args['optimizer'], model_args['learning_rate'])

    model.to(device)
    batch_size = model_args['batch_size']
    data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    valid_data_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size)

    #Initialize Variables for EarlyStopping
    best_loss = float('inf')
    best_model_weights = None
    patience = int(model_args['early_stopping'])
    epochs_no_improve = 0
    min_delta = float(model_args['min_delta'])

    loss_logger = LossLogger(os.path.join(settings.LOG_PATH, model_name))

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, data_loader, embeddings_model, optimizer, device, max_grad_norm)
        torch.cuda.empty_cache()

        # Validation
        val_loss = validate(model, valid_data_loader, embeddings_model, device)
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
        #TODO dodati da se sprema u neki output file 
        print(f"Train Loss ({epoch+1}/{num_epochs}) = {train_loss}")
        print(f"Validation Loss ({epoch+1}/{num_epochs}) = {val_loss}")

    # Save the best model
    best_model.load_state_dict(best_model_weights)
    torch.save(best_model.state_dict(), settings.MODEL_PATH +f"/{model_name}_best.bin")


def main():
    args = parse_args()
    model_name = args.model_name
    model_args, settings_args = settings.Settings(args.config)
    
    # Add your training code here
    print("Training with the following settings:")
    print("Model Args:", model_args)
    print("Settings Args:", settings_args)

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    
    # TODO: napisati kod za CNN charachter embedding u preprocessing
    dataset_loader = DatasetLoader("ncbi_disease_json", settings.DATA_PATH)
    total_tags, (tokens_train, tags_train), (tokens_val, tags_val), (tokens_test, tags_test) = dataset_loader.load_data()
    num_tags = len(total_tags)

    max_len = get_max_len(tokens_train, tokens_val, tokens_test)
    train_data = Dataset(tokens_train, tags_train, max_len)
    val_data = Dataset(tokens_val, tags_val, max_len)
    test_data = Dataset(tokens_test, tags_test, max_len)

    embeddings_model = Embedding.create(settings_args['embedding'],
                                settings.EMBEDDINGS_PATH, 
                                dataset_loader.dataset_name, max_len)
    #train(model_name, model_args, num_tags, train_data, val_data, embedding_model, device)

    #Evaluate on test set

    best_model_weights = torch.load(settings.MODEL_PATH + f"{model_name}_best.bin")
    best_model = models.BiRNN_CRF(num_tags, model_args, embeddings_model.embedding_dim) 
    best_model.load_state_dict(best_model_weights)
    eval = Evaluation(total_tags)
    test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=model_args['batch_size'])
    eval.evaluate(test_data_loader, best_model, device, embeddings_model)
    
if __name__ == "__main__":
    main()
    #python train.py --model_name='probni'