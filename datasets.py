import json
import os
from preprocessing import Embedding
import settings
import torch

MAX_LEN = 256

class DatasetLoader:
    def __init__(self, dataset_name, dataset_path):
        self.folder_path = os.path.join(dataset_path, dataset_name)
        self.dataset_name = dataset_name
    
    def _load_file(self, file_name): 
        file_path = os.path.join(self.folder_path, file_name)
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                text, tags = [], []
                for line in f:
                    line = json.loads(line)
                    text.append(line["tokens"])
                    tags.append(line["tags"])
            return text, tags # list(itertools.chain(*text)), list(itertools.chain(*tags))
        else:
            print(f"File {file_path} not found")

    def load_data(self):
        ''' 
        Load the dataset
        Returns:    
            total_tags: List of all tags in the dataset and their encoding
            (text_train, tags_train): List of training data split into text and tags
            (text_val, tags_val): List of validation data split into text and tags
            (text_test, tags_test): List of testing data split into text and tags  
        '''
        total_tags = {}
        text_train, text_val, text_test = [], [], []
        tags_train, tags_val, tags_test = [], [], []

        tags_file = os.path.join(self.folder_path, "label.json")        
        train_file = os.path.join(self.folder_path, "train.json")
        val_file = os.path.join(self.folder_path, "devel.json")
        test_file = os.path.join(self.folder_path, "test.json")
        
        with open(tags_file, "r", encoding="utf-8") as f:
                total_tags = json.load(f)
        self.num_tags = len(total_tags)
        text_train, tags_train = self._load_file(train_file)
        text_val, tags_val =self._load_file(val_file)
        text_test, tags_test = self._load_file(test_file)

        return total_tags, (text_train, tags_train), (text_val, tags_val), (text_test, tags_test)


class Dataset:
    def __init__(self, tokens, tags, attention_masks):
        self.tokens = tokens
        self.tags = tags
        self.attention_masks = attention_masks


    def __len__(self):
        return len(self.tokens) 
    

    def __getitem__(self, idx):
        return (
            self.tokens[idx], 
            self.tags[idx],  
            self.attention_masks[idx]
        )

def get_max_len(text_train): #, text_val, text_test):
    max_len = max(len(sublist) for sublist in text_train)#train_max_len = max(len(sublist) for sublist in text_train)
    #val_max_len = max(len(sublist) for sublist in text_val)
    #test_max_len = max(len(sublist) for sublist in text_test)
    #max_len = max(train_max_len, val_max_len, test_max_len)
    return min(max_len, MAX_LEN)

'''
def collate_fn(batch):
    tokens, tags, att_masks = zip(*batch)

    # Convert to tensors
    tokens = torch.stack(tokens)  # Already padded from tokenizer
    att_masks = torch.stack(att_masks)  # Already padded from tokenizer
    
    # Pad tags manually (set PAD value to -1)
    tags = pad_sequence(tags, batch_first=True, padding_value=-1)

    return tokens, tags, att_masks'''
 
# Example usage
if __name__ == "__main__":
    dataset_loader = DatasetLoader("ncbi_disease_json", settings.DATA_PATH)
    total_tags, (text_train, tags_train), (text_val, tags_val), (text_test, tags_test) = dataset_loader.load_data()
    
    max_len = get_max_len(text_train, text_val, text_test)

    embeddings_model = Embedding.create('bioELMo', dataset_loader.dataset_name, max_len) #bioBERT
    
    tokens_train_padded, tags_train_padded, attention_masks = embeddings_model.tokenize_and_pad_text(text_train, tags_train)
    train_dataset = Dataset(tokens_train_padded, tags_train_padded, attention_masks)


    data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32) #, collate_fn=collate_fn)
    i = 0
    for text, tags, att_masks in data_loader:
        print(f"Batch size: {text.shape[0]}")
        print(f"Tokens shape: {text.shape}")
        print(f"Tags shape: {tags.shape}")
        print(f"Attention masks shape: {att_masks.shape}")
        embs = embeddings_model.get_embedding(text, att_masks)
        print(embs.shape)
        #print(embs)
        if i == 0:
            break


    