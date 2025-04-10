import argparse
import copy
import itertools
import os
import models
from preprocessing import Embedding, CharEmbeddingCNN
from datasets import Dataset
import utils.dataset_converter as dataset_converter
from evaluation import Evaluation
import torch
from torch.optim import *
import settings
from datasets import *
from torch.nn.utils import clip_grad_norm_
import numpy as np
from utils.logger import Logger

def parse_args():
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument('--config', type=str, required=False, help='Path to the config file', default='/lustre/home/mvidas/Diplomski/experiments/default_train.yml')    
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
    
def train_one_epoch(model, data_loader, word_embeddings_model, char_embeddings, optimizer, device, max_grad_norm):
    model.train()
    final_loss = 0
    for (tokens, tags, att_mask), char_embedding in zip(data_loader, char_embeddings or itertools.repeat(None)): # tqdm(data_loader, total=len(data_loader)):
        optimizer.zero_grad()
        
        batch_embeddings = word_embeddings_model.get_embedding(tokens, att_mask)
        batch_embeddings = batch_embeddings.to(device)
        batch_attention_masks = att_mask.to(device)
        if char_embedding != None:
            batch_char_embedding = char_embedding.to(device)
        else:
            batch_char_embedding = None
        batch_tags = tags.to(device)

        loss = model(batch_embeddings, batch_tags, batch_attention_masks, batch_char_embedding)
        loss.backward()
        if max_grad_norm is not None:
            clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()
        final_loss += loss.item()
    return final_loss / len(data_loader)

def train(model_name, model_args, num_tags, train_data_loader, valid_data_loader, word_embeddings_model, char_emb, text_train, text_val, max_len, batch_size, device, num_to_tag, eval, logger):
    print("Started training")
    max_grad_norm = model_args['max_grad_norm']
    # Create models
    model = models.BiRNN_CRF(num_tags, model_args, word_embeddings_model.embedding_dim, model_args['char_embedding_dim'])
    best_model = models.BiRNN_CRF(num_tags, model_args, word_embeddings_model.embedding_dim, model_args['char_embedding_dim'])

    num_epochs = model_args['epochs']
    optimizer = define_optimizer(model, model_args['optimizer'], model_args['learning_rate'])
    model.to(device)

    #Initialize Variables for EarlyStopping
    best_loss = float('inf')
    best_model_weights = None
    patience = int(model_args['early_stopping'])
    epochs_no_improve = 0
    min_delta = float(model_args['min_delta'])

    for epoch in range(num_epochs):
        if char_emb is not None:
            train_char_embeddings = char_emb.batch_cnn_embedding_generator(text_train, max_len, batch_size)
            val_char_embeddings = char_emb.batch_cnn_embedding_generator(text_val, max_len, batch_size)
        train_loss = train_one_epoch(model, train_data_loader, word_embeddings_model, train_char_embeddings, optimizer, device, max_grad_norm)
        logger.log_train_loss(epoch+1, train_loss)
        #torch.cuda.empty_cache()

        # Validation
        val_loss = eval.evaluate(valid_data_loader, model, device, word_embeddings_model, val_char_embeddings, num_to_tag, logger, epoch+1)
        #torch.cuda.empty_cache()

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
        
        if epoch%10 == 0:
            print(f"Train Loss ({epoch+1}/{num_epochs}) = {train_loss}")
            print(f"Validation Loss ({epoch+1}/{num_epochs}) = {val_loss}")

    # Save the best model
    best_model.load_state_dict(best_model_weights)
    torch.save(best_model.state_dict(), settings.MODEL_PATH +f"/{model_name}_best.bin")


def main(): #TODO: dodati za reproducility (https://pytorch.org/docs/stable/notes/randomness.html)
    args = parse_args()
    model_name = args.model_name
    model_args, settings_args = settings.Settings(args.config)
    
    print("Training with the following settings:")
    print("Model Args:", model_args)
    print("Settings Args:", settings_args)

    logger = Logger(os.path.join(settings.LOG_PATH, model_name), model_args, settings_args)

    if torch.cuda.is_available():
        device = 'cuda'
        print("Using cuda")
    else:
        device = 'cpu'
        print("Using cpu")
    
    dataset_loader = DatasetLoader(settings_args['dataset'], settings.DATA_PATH)
    tag_to_num, (text_train, tags_train), (text_val, tags_val), (text_test, tags_test) = dataset_loader.load_data()
    num_tags = len(tag_to_num) #number of tags used (e.g. 3 for O, B-entity, I-entity)
    num_to_tag = dict((v,k) for k,v in tag_to_num.items()) 
    
    max_len = model_args['max_length'] 
    word_embedding = settings_args['word_embedding']
    word_embeddings_model = Embedding.create(word_embedding, dataset_loader.dataset_name, max_len) 

    eval = Evaluation(word_embeddings_model.tokenizer, settings_args["tagging_scheme"])

    #if no char cnn is used
    char_emb = None
    train_char_embeddings = None
    val_char_embeddings = None
    test_char_embeddings = None
    model_args['char_embedding_dim'] = None

    batch_size = model_args['batch_size']
    if settings_args['char_cnn_embedding']:
        vocab = settings_args['cnn_vocab']
        char_emb_size = settings_args['cnn_embedding_dim']
        model_args['char_embedding_dim'] = char_emb_size
        char_kernel_size = settings_args['cnn_embedding_kernel_size']
        max_word_len = settings_args['cnn_max_word_len']
        char_emb = CharEmbeddingCNN(vocab, char_emb_size, char_kernel_size, max_word_len)
        test_char_embeddings = char_emb.batch_cnn_embedding_generator(text_test, max_len, batch_size)
        
    
    tokens_train_padded, tags_train_padded, attention_masks_train = word_embeddings_model.tokenize_and_pad_text(text_train, tags_train)
    train_data = Dataset(tokens_train_padded, tags_train_padded, attention_masks_train)
    tokens_val_padded, tags_val_padded, attention_masks_val = word_embeddings_model.tokenize_and_pad_text(text_val, tags_val)
    val_data = Dataset(tokens_val_padded, tags_val_padded, attention_masks_val)

    train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
    valid_data_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)

    train(model_name, model_args, num_tags, train_data_loader, valid_data_loader, 
        word_embeddings_model, char_emb, text_train, text_val, max_len, batch_size, 
        device, num_to_tag, eval, logger)

    #Evaluate on test set
    tokens_test_padded, tags_test_padded, attention_masks_test = word_embeddings_model.tokenize_and_pad_text(text_test, tags_test)
    test_data = Dataset(tokens_test_padded, tags_test_padded, attention_masks_test)
    test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

    best_model_weights = torch.load(settings.MODEL_PATH + f"/{model_name}_best.bin")
    best_model = models.BiRNN_CRF(num_tags, model_args, word_embeddings_model.embedding_dim, model_args['char_embedding_dim']) 
    best_model.load_state_dict(best_model_weights)
    print("Testing")
    eval.evaluate(test_data_loader, best_model, device, word_embeddings_model, test_char_embeddings, num_to_tag, logger)
    
if __name__ == "__main__":
    main()
    #python train.py --model_name='probni'