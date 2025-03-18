import json
import os
from sklearn import preprocessing


# Datasets in tsv files are available at https://github.com/cambridgeltl/MTL-Bioinformatics-2016 
class Dataset_Converter:
    '''
    Class for converting tsv dataset files to json format with encoded tags.   
    Args:
        dataset_name: Name of the dataset
        dataset_path: Path to the dataset folder
    '''
    def __init__(self, dataset_name, dataset_path):
        self.folder_path = os.path.join(dataset_path, dataset_name)
        self.enc_tag =  preprocessing.LabelEncoder() 

    def load_file(self, file_name): 
        sentences,tags=[],[]
        sentence,tag=[],[]

        json_list = []
        json_format = dict()

        file_path = os.path.join(self.folder_path, file_name)
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                for line in lines: 
                    line = line.strip()
                    if len(line) == 0:
                        if sentence==[] and tag==[]:
                            continue
                        sentences.append(sentence)
                        tag = self.enc_tag.transform(tag).tolist()
                        tags.append(tag)
                        json_format["tags"] = tag
                        json_format["tokens"] = sentence
                        json_list.append(json_format)
                        json_format = dict()
                        sentence,tag=[],[]
                    else:
                        word,tag_line = line.split("\t")
                        sentence.append(word)
                        tag.append(tag_line)


            return sentences, tags, json_list
        else:
            print(f"File {file_path} not found")

    def load_data(self):
        ''' 
        Load the dataset, splits sentences and tags, encodes tags.
        Returns:    
            total_tags: List of all tags in the dataset
            (text_train, tags_train): Training data
            (text_val, tags_val): Validation data
            (text_test, tags_test): Testing data        
        '''
        total_tags = []
        text_train, text_val, text_test = [], [], []
        tags_train, tags_val, tags_test = [], [], []

        tags_file = os.path.join(self.folder_path, "classes.txt")        
        train_file = os.path.join(self.folder_path, "train.tsv")
        val_file = os.path.join(self.folder_path, "devel.tsv")
        test_file = os.path.join(self.folder_path, "test.tsv")
        
        with open(tags_file, "r", encoding="utf-8") as f:
                for line in f.readlines():
                    total_tags.append(line.strip())
        self.enc_tag.fit(list(total_tags))
        text_train, tags_train, json_train = self.load_file(train_file)
        text_val, tags_val, json_val =self.load_file(val_file)
        text_test, tags_test, json_test = self.load_file(test_file)

        return total_tags, (text_train, tags_train, json_train), (text_val, tags_val, json_val), (text_test, tags_test, json_test)

# Example usage
if __name__ == "__main__":
    dataset = Dataset_Converter("ncbi_kaggle", r"C:\Users\Marta\Desktop\Dipl_code\Diplomski\data")
    total_tags, (text_train, tags_train, json_train), (text_val, tags_val, json_val), (text_test, tags_test, json_test) = dataset.load_data()
    print(f"Total tags: {total_tags}")
    print(f"Training data: {len(text_train)} sentences")
    print(f"Validation data: {len(text_val)} sentences")
    print(f"Testing data: {len(text_test)} sentences")

    with open(dataset.folder_path + "\\train.json", "w") as f:
        for item in json_train:
            f.write(json.dumps(item) + "\n")

    with open(dataset.folder_path + "\\devel.json", "w") as f:
        for item in json_val:
            f.write(json.dumps(item) + "\n")

    with open(dataset.folder_path + "\\test.json", "w") as f:    
        for item in json_test:
            f.write(json.dumps(item) + "\n")
    