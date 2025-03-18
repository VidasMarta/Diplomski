import json
import os
import settings
import torch
from torch.nn.utils.rnn import pad_sequence
import itertools

MAX_LEN = 256

class DatasetLoader:
    def __init__(self, dataset_name, dataset_path):
        self.folder_path = os.path.join(dataset_path, dataset_name)
        self.dataset_name = dataset_name
    
    def _load_file(self, file_name): 
        file_path = os.path.join(self.folder_path, file_name)
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                tokens, tags = [], []
                for line in f:
                    line = json.loads(line)
                    tokens.append(line["tokens"])
                    tags.append(line["tags"])
            return tokens, tags # list(itertools.chain(*tokens)), list(itertools.chain(*tags))
        else:
            print(f"File {file_path} not found")

    def load_data(self):
        ''' 
        Load the dataset
        Returns:    
            total_tags: List of all tags in the dataset and their encoding
            (tokens_train, tags_train): List of training data split into tokens and tags
            (tokens_val, tags_val): List of validation data split into tokens and tags
            (tokens_test, tags_test): List of testing data split into tokens and tags  
        '''
        total_tags = {}
        tokens_train, tokens_val, tokens_test = [], [], []
        tags_train, tags_val, tags_test = [], [], []

        tags_file = os.path.join(self.folder_path, "label.json")        
        train_file = os.path.join(self.folder_path, "train.json")
        val_file = os.path.join(self.folder_path, "devel.json")
        test_file = os.path.join(self.folder_path, "test.json")
        
        with open(tags_file, "r", encoding="utf-8") as f:
                total_tags = json.load(f)
        self.num_tags = len(total_tags)
        tokens_train, tags_train = self._load_file(train_file)
        tokens_val, tags_val =self._load_file(val_file)
        tokens_test, tags_test = self._load_file(test_file)

        return total_tags, (tokens_train, tags_train), (tokens_val, tags_val), (tokens_test, tags_test)


class Dataset:
    def __init__(self, tokens, tags, max_len=100):
        self.tokens = tokens
        self.tags = tags
        self.max_len = max_len
        self.pad_token = "PAD"
        self.pad_tag = -1 

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        token_list = self.tokens[idx]
        tag_list = self.tags[idx]

        # Pad tokens with string padding
        if len(token_list) < self.max_len:
            token_list += [self.pad_token] * (self.max_len - len(token_list))
            tag_list += [self.pad_tag] * (self.max_len - len(tag_list))  # Pad tags with -1 (or any other padding tag)
        
        # Truncate if needed (in case your max_len is smaller than the actual sentence length)
        token_list = token_list[:self.max_len]
        tag_list = tag_list[:self.max_len]

        # Generate attention mask: 1 for real tokens, 0 for padding tokens
        attention_mask = [1 if token != self.pad_token else 0 for token in token_list]

        # Return token list, tag list (as tensor), and attention mask
        return token_list, torch.tensor(tag_list), torch.tensor(attention_mask)

def get_max_len(tokens_train, tokens_val, tokens_test):
    train_max_len = max(len(sublist) for sublist in tokens_train)
    val_max_len = max(len(sublist) for sublist in tokens_val)
    test_max_len = max(len(sublist) for sublist in tokens_test)
    max_len = max(train_max_len, val_max_len, test_max_len)
    return min(max_len, MAX_LEN)
 
# Example usage
if __name__ == "__main__":
    dataset_loader = DatasetLoader("ncbi_disease_json", settings.DATA_PATH)
    total_tags, (tokens_train, tags_train), (tokens_val, tags_val), (tokens_test, tags_test) = dataset_loader.load_data()
    max_len = get_max_len(tokens_train, tokens_val, tokens_test)
    train_data = Dataset(tokens_train, tags_train, max_len)
    val_data = Dataset(tokens_val, tags_val, max_len)
    test_data = Dataset(tokens_test, tags_test, max_len)
    data_loader = torch.utils.data.DataLoader(train_data, batch_size=1) #, collate_fn=collate_fn)
    i = 0
    for tokens, tags, att_mask in data_loader:
        print(tokens)
        print(tags)
        print(att_mask)
        if i == 0:
            break


    