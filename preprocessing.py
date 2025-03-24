import nltk
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from allennlp.modules.elmo import Elmo, batch_to_ids
from abc import ABC, abstractmethod
from torch.nn.utils.rnn import pad_sequence
import os
import settings

#TODO dodati CNN charachter based embeddings
class Embedding(ABC):
    def __init__(self, embedding_model_name, dataset_name, max_len=256):
        self.dataset_name = dataset_name
        self.MAX_LEN = max_len
        self.embedding_model_name = embedding_model_name
        self.embedding_dim = None

    @staticmethod
    def create(embedding_model_name, dataset_name, max_len=256):
        '''
        Factory method to create an embedding model
        Args:
            embedding_model_name: Name of the embedding model
            embeddings_path: Path to save embeddings
            dataset_name: Name of the dataset
            max_len: Maximum length of the input sequence
        Returns:
            Embedding model instance or raises ValueError if the embedding model is not supported
        '''
        if embedding_model_name == 'bioBERT':
            return Embedding_bioBERT(embedding_model_name, dataset_name, max_len)
        elif embedding_model_name == 'bioELMo':
            return Embedding_bioELMo(embedding_model_name, dataset_name, max_len)
        else:
            raise ValueError(f"Embedding {embedding_model_name} not supported, bioBERT and bioELMo are.")
        
    @abstractmethod
    def get_embedding(self, tokens):
        pass

    @abstractmethod
    def tokenize_and_pad_text(self, text, tags):
        pass
    
class Embedding_bioBERT(Embedding):
    def __init__(self, embedding_model_name, dataset_name, max_len=256):
        super(Embedding_bioBERT, self).__init__(embedding_model_name, dataset_name, max_len)
        self.max_len = max_len
        self.embedding_dim = 768  # Dimensionality of BioBERT embeddings
        model_name = "dmis-lab/biobert-base-cased-v1.1"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)

    def tokenize_and_pad_text(self, text, tags):
        # Convert list of token lists into full sentences
        sentences_list = [["".join(word) for word in line] for line in text]

        # Tokenize with padding and truncation
        tokenized_output = self.tokenizer(
            sentences_list, 
            padding="max_length", 
            truncation=True, 
            max_length=self.max_len,  
            return_tensors="pt", 
            is_split_into_words=True,
            return_token_type_ids = False,
            add_special_tokens=False
        )

        # Initialize list for padded tags
        padded_tags = []
        #Bert's tokenizer can split words into subwords, continuous words are marked with ## at the beginning of a token

        # For each sequence, align tags with tokenized tokens
        for line_tags, line_tokens in zip(tags, tokenized_output["input_ids"]):
            new_tags = []
            word_tag_index = 0
            
            # Tokenize the current sentence and break it into words and subwords
            tokens = self.tokenizer.tokenize(self.tokenizer.decode(line_tokens))

            # Loop through each token in the tokenized sentence
            for token, token_id in zip(tokens, line_tokens):
                # If the token ID corresponds to a real word (i.e., not padding)
                if token_id != self.tokenizer.pad_token_id:
                    # Check if this token corresponds to the first token of a word
                    if token.startswith("##"):
                        # Subword token -> keep the same tag as the word
                        new_tags.append(new_tags[-1])
                    else:
                        # First subword of a word -> assign a new tag
                        new_tags.append(line_tags[word_tag_index])
                        word_tag_index += 1  # Move to the next word tag
                else:
                    # Use -1 for padding tokens
                    new_tags.append(-1)
            
            padded_tags.append(new_tags)

        tags_tensor = torch.tensor(padded_tags, dtype=torch.long)

        return tokenized_output["input_ids"], tags_tensor, tokenized_output["attention_mask"]


    def get_embedding(self, token_list, attention_masks): 
        '''
        Get BioBERT embeddings using precomputed attention masks from dataloading.
        Args:
            tokens: List of tokenized sequences
        Returns: 
            embeddings_list: List of BioBERT embeddings for a batch
        '''
        self.bert.eval()
    
        with torch.no_grad():
            outputs = self.bert(token_list, attention_mask=attention_masks)
    
        return outputs.last_hidden_state

        # Save embeddings and attention masks
        #np.save(f"{self.embeddings_path}\\{self.dataset_name}\\_BioBERT_embeddings.npy", embeddings_list)
        #np.save(f"{self.embeddings_path}\\{self.dataset_name}\\_BioBERT_attention_masks.npy", attention_masks_list)
        #print(f"Processed {len(embeddings_list)} sentences.  Embeddings and attention masks saved!")


class Embedding_bioELMo(Embedding):
    def __init__(self, embedding_model_name, dataset_name, max_len=256):
        super(Embedding_bioELMo, self).__init__(embedding_model_name, dataset_name, max_len)
        # skinute tezine i options file dostupne na https://github.com/Andy-jqa/bioelmo?tab=readme-ov-file
        set_up_path = os.path.join(settings.EMBEDDINGS_PATH, "bioELMo_setup")
        options_file = os.path.join(set_up_path, "biomed_elmo_options.json")
        weight_file = os.path.join(set_up_path, "biomed_elmo_weights.hdf5")
        self.elmo = Elmo(options_file, weight_file, num_output_representations=1, dropout=0)
        self.embedding_dim = 1024  # Dimensionality of BioELMo embeddings
        self.max_len = max_len
        
    def tokenize_and_pad_text(self, text, tags): #elmos tokenisation is words -> only padding required
        sentences_list = [["".join(word) for word in line] for line in text]
        tokenized_text = batch_to_ids(sentences_list)

        tensor_tags = [torch.tensor(t) for t in tags]        

        tokens_padded = pad_sequence(tokenized_text, batch_first=True, padding_value=0)
        tags_padded = pad_sequence(tensor_tags, batch_first=True, padding_value=-1)      

        padding_mask = torch.where(tags_padded != -1, 1, 0)  

        return tokens_padded, tags_padded, padding_mask  

    def get_embedding(self, tokens, attention_masks):  
        '''
        Get BioELMo embeddings for a list of tokens
        Attention masks are not used for ELMo embedding
        Args:
            tokens: List of tokens
        Returns: 
            embeddings_list: List of BioELMo embeddings
        '''
        self.elmo.eval() 
        with torch.no_grad():
            elmo_output = self.elmo(tokens)

            embeddings = elmo_output["elmo_representations"][0] # Removing the first dimension since it is 1
            mask = elmo_output["mask"]

            #print("emb ", embeddings.shape)  # torch.Size([32, 123, 1024])
            #print(mask.shape)  # torch.Size([32, 123])
            
            embeddings = embeddings * mask.unsqueeze(-1)  # Apply mask along token dimension

            return embeddings 
        
        # Save embeddings
        #np.save(f"{self.embeddings_path}\\{self.dataset_name}\\_BioELMo_embeddings.npy", embeddings_list)
        #np.save(f"{self.embeddings_path}\\{self.dataset_name}\\_BioELMo_attention_masks.npy", attention_masks_list)
        #print(f"Processed {len(embeddings_list)} sentences. Embeddings and attention masks saved!")
