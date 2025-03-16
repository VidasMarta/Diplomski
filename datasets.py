import json
import os
from sklearn import preprocessing

class Dataset:
    def __init__(self, dataset_name, dataset_path):
        self.folder_path = os.path.join(dataset_path, dataset_name)
        self.num_tags = 0

    def load_file(self, file_name): 
        file_path = os.path.join(self.folder_path, file_name)
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                tokens, tags = [], []
                for line in lines:
                    line = json.loads(line)
                    tokens.append(line["tokens"])
                    tags.append(line["tags"])
            return tokens, tags
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
        val_file = os.path.join(self.folder_path, "valid.json")
        test_file = os.path.join(self.folder_path, "test.json")
        
        with open(tags_file, "r", encoding="utf-8") as f:
                total_tags = json.load(f)
        self.num_tags = len(total_tags)
        tokens_train, tags_train = self.load_file(train_file)
        tokens_val, tags_val =self.load_file(val_file)
        tokens_test, tags_test = self.load_file(test_file)

        return total_tags, (tokens_train, tags_train), (tokens_val, tags_val), (tokens_test, tags_test)

# Example usage
if __name__ == "__main__":
    dataset = Dataset("ncbi_disease_json", r"C:\Users\Marta\Desktop\Dipl_code\Diplomski\data")
    total_tags, (tokens_train, tags_train), (tokens_val, tags_val), (tokens_test, tags_test) = dataset.load_data()
    print(total_tags)

    