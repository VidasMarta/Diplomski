import os
import random
import numpy as np
import optuna
import torch
from datasets import Dataset, DatasetLoader
from evaluation import Evaluation
from models import ft_bb_BiRNN_CRF
from preprocessing import CharEmbeddingCNN, Embedding
import settings
from utils import trainer
from optuna.visualization import plot_optimization_history, plot_param_importances
import joblib

DATASET_NAME = "bc5cdr_json" # "ncbi_disease_json" or "bc5cdr_json"
MODEL_NAME = "D2_hyper_param_tuning_elmo_all" #D1 or D2

def train_model(model_args):    
    # Load datasets for train and test
    dataset_loader = DatasetLoader(DATASET_NAME, settings.DATA_PATH)
    tag_to_num, (text_train, tags_train), (text_val, tags_val), (_, _) = dataset_loader.load_data()
    num_to_tag = dict((v,k) for k,v in tag_to_num.items())
    word_embeddings_model = Embedding.create(model_args['word_embedding'], dataset_loader.dataset_name, model_args['max_len']) 
    eval = Evaluation(word_embeddings_model, "IOB1")
    
    tokens_train_padded, tags_train_padded, attention_masks_train, crf_mask_train = word_embeddings_model.tokenize_and_pad_text(text_train, tags_train)
    train_data = Dataset(tokens_train_padded, tags_train_padded, attention_masks_train, crf_mask_train)
    tokens_val_padded, tags_val_padded, attention_masks_val, crf_mask_val = word_embeddings_model.tokenize_and_pad_text(text_val, tags_val)
    val_data = Dataset(tokens_val_padded, tags_val_padded, attention_masks_val, crf_mask_val)
    train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=model_args['batch_size'])
    valid_data_loader = torch.utils.data.DataLoader(val_data, batch_size=model_args['batch_size'])

    if model_args['bert_finetuning']:
        return trainer.Finetuning_Trainer(MODEL_NAME, model_args, len(tag_to_num), train_data_loader, valid_data_loader, word_embeddings_model, model_args['char_emb'], 
                                    text_train, text_val,  model_args['max_len'], model_args['batch_size'],  model_args['device'], num_to_tag, eval, None).train(False)
    else: 
        return trainer.Normal_Trainer(MODEL_NAME, model_args, len(tag_to_num), train_data_loader, valid_data_loader, word_embeddings_model, model_args['char_emb'], 
                                      text_train, text_val, model_args['max_len'], model_args['batch_size'], model_args['device'], num_to_tag, eval, None).train(False)

def objective(trial): 
    # Hyperparameters to tune
    model_args = {}
    model_args['hidden_size'] = trial.suggest_categorical('hidden_size', [256, 512, 768])
    model_args['lr'] = trial.suggest_float('lr', 5e-4, 1.5e-3, log=True) #trenutni lr na 1e-3
    #model_args['ft_lr'] = trial.suggest_float('ft_lr', 1e-5, 3e-5, log=True) #trenutni ft_lr na 2e-5
    model_args['optimizer'] = "adamw" #trial.suggest_categorical("optimizer", ["adam", "adamw"])
    model_args['dropout'] = trial.suggest_uniform("dropout", 0.15, 0.45)

    model_args['attention'] = True #trial.suggest_categorical("attention", [False, True])
    if model_args['attention']:
        model_args['att_num_of_heads'] = 4 #trial.suggest_categorical("att_num_of_heads", [4, 8, 16])
    model_args['char_cnn_embedding'] = True #trial.suggest_categorical("char_cnn_embedding", [False, True])
    if model_args['char_cnn_embedding']:
        model_args['char_embedding_dim'] = 256 #trial.suggest_categorical("char_embedding_dim", [128, 256])
        feature_size = 128 #trial.suggest_categorical("feature_size", [128, 256])  
        vocab = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
        max_word_len = 20
        model_args['char_emb'] = CharEmbeddingCNN(vocab, model_args['char_embedding_dim'], feature_size, max_word_len)
    else:
        model_args['char_emb'] = None
        model_args['char_embedding_dim'] = None

    # Other hyperparameters
    model_args['use_crf'] = True
    model_args['batch_size'] = 32
    model_args['num_layers'] = 1
    model_args['cell'] = trial.suggest_categorical('cell', ['lstm', 'gru'])
    model_args['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_args['max_len'] = 256
    model_args['loss'] = 'CRF'
    model_args['epochs'] = 15 #tako da kraće traje treniranje
    model_args['max_grad_norm'] = 5.0
    model_args['early_stopping'] = 5
    model_args['word_embedding'] = "bioELMo"
    model_args['bert_finetuning'] = False

    return train_model(model_args) 


def main():        
    study = optuna.create_study(direction='maximize') 
    study.optimize(objective, n_trials=70)

    print("Best Hyperparameters:", study.best_params)

def set_seed(seed: int = 42): ##za reproducility, izvor: https://medium.com/we-talk-data/how-to-set-random-seeds-in-pytorch-and-tensorflow-89c5f8e80ce4
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")
    
if __name__ == "__main__":
    set_seed()
    main()
    #plot_graphs()