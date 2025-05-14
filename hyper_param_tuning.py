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

DATASET_NAME = "ncbi_disease_json" # or "bc5cdr_json"
MODEL_NAME = "D1_hyper_param_tuning"
MAX_LEN = 256
BATCH_SIZE = 32
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_dataset_loaders(text_train, tags_train, text_val, tags_val, word_embeddings_model):    
    # Load datasets for train and test
    tokens_train_padded, tags_train_padded, attention_masks_train, crf_mask_train = word_embeddings_model.tokenize_and_pad_text(text_train, tags_train)
    train_data = Dataset(tokens_train_padded, tags_train_padded, attention_masks_train, crf_mask_train)
    tokens_val_padded, tags_val_padded, attention_masks_val, crf_mask_val = word_embeddings_model.tokenize_and_pad_text(text_val, tags_val)
    val_data = Dataset(tokens_val_padded, tags_val_padded, attention_masks_val, crf_mask_val)
    train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE)
    valid_data_loader = torch.utils.data.DataLoader(val_data, batch_size=BATCH_SIZE)

    return train_data_loader, valid_data_loader

def objective(trial): #TODO otkomentirati ostale za hiperparam za tuning, trenutno samo 2 tunam, ovo je testna verzija
    # Hyperparameters to tune
    model_args = {}
    model_args['hidden_size'] = trial.suggest_categorical('hidden_size', [256, 512, 768])
    model_args['lr'] = trial.suggest_loguniform('lr', 5e-4, 1.5e-3) #trenutni lr na 1e-3
    model_args['ft_lr'] = 2e-5 #trial.suggest_loguniform('ft_lr', 1e-5, 3e-5) #trenutni ft_lr na 2e-5
    model_args['optimizer'] = "adam" #trial.suggest_categorical("optimizer", ["adam", "adamw"])
    model_args['dropout'] = 0.3 #trial.suggest_uniform("dropout", 0.15, 0.45)
    model_args['attention'] = False #trial.suggest_categorical("attention", [False, True])
    if model_args['attention']:
        model_args['att_num_of_heads'] = trial.suggest_categorical("att_num_of_heads", [4, 8, 16])
    model_args['char_cnn_embedding'] = False #trial.suggest_categorical("char_cnn_embedding", [False, True])
    if model_args['char_cnn_embedding']:
        model_args['cnn_embedding_dim'] = trial.suggest_categorical("cnn_embedding_dim", [128, 256])
        feature_size = trial.suggest_categorical("feature_size", [128, 256])  
        vocab = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
        max_word_len = 20
        model_args['char_emb'] = CharEmbeddingCNN(vocab, model_args['cnn_embedding_dim'], feature_size, max_word_len)
    else:
        model_args['char_emb'] = None
        model_args['cnn_embedding_dim'] = None

    # Other hyperparameters
    model_args['use_crf'] = True
    model_args['loss'] = 'CRF'
    model_args['epochs'] = 5 #tako da kraće traje treniranje
    model_args['max_grad_norm'] = 5.0
    
    dataset_loader = DatasetLoader(DATASET_NAME, settings.DATA_PATH)
    tag_to_num, (text_train, tags_train), (text_val, tags_val), (_, _) = dataset_loader.load_data()
    num_to_tag = dict((v,k) for k,v in tag_to_num.items())
    word_embeddings_model = Embedding.create("bioBERT", dataset_loader.dataset_name, MAX_LEN) 
    eval = Evaluation(word_embeddings_model, "IOB1")
    train_data_loader, valid_data_loader = get_dataset_loaders(dataset_loader, text_train, tags_train, text_val, tags_val, word_embeddings_model)

    f1_valid = trainer.Hyperparam_tuning_trainer(MODEL_NAME, model_args, len(tag_to_num), train_data_loader, valid_data_loader, word_embeddings_model, char_emb, text_train, text_val, MAX_LEN,
                                BATCH_SIZE, DEVICE, num_to_tag, eval).train()

    return f1_valid 

def main():
    study = optuna.create_study(direction='maximize') #gledat će f1 na val skupu pa treba maksimizirati
    study.optimize(objective, n_trials=50) #staviti na 50 ili 100
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