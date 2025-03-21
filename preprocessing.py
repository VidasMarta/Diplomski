import nltk
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
#from allennlp import Elmo
from abc import ABC, abstractmethod
from torch.nn.utils.rnn import pad_sequence

#TODO dodati CNN charachter based embeddings
#TODO dodati druge varijante labelinga (npr. BIOES)
class Embedding(ABC):
    def __init__(self, embedding_model_name, embeddings_path, dataset_name, max_len=256):
        self.dataset_name = dataset_name
        self.embeddings_path = embeddings_path
        self.MAX_LEN = max_len
        self.embedding_model_name = embedding_model_name
        self.embedding_dim = None

    @staticmethod
    def create(embedding_model_name, embeddings_path, dataset_name, max_len=256):
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
            return Embedding_bioBERT(embedding_model_name, embeddings_path, dataset_name, max_len)
        elif embedding_model_name == 'bioELMo':
            pass #return Embedding_bioELMo(embedding_model_name, embeddings_path, dataset_name, max_len)
        else:
            raise ValueError(f"Embedding {embedding_model_name} not supported")
        
    @abstractmethod
    def get_embedding(self, tokens):
        pass

    @abstractmethod
    def tokenize_and_pad_text(self, text, tags):
        pass
    
class Embedding_bioBERT(Embedding):
    def __init__(self, embedding_model_name, embeddings_path, dataset_name, max_len=256):
        super(Embedding_bioBERT, self).__init__(embedding_model_name, embeddings_path, dataset_name, max_len)
        self.max_len = max_len
        self.embedding_dim = 768  # Dimensionality of BioBERT embeddings
        model_name = "dmis-lab/biobert-base-cased-v1.1"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)

    def tokenize_and_pad_text(self, text, tags):
        # Convert list of token lists into full sentences
        sentences_list = []
        for line in text:
            sentences_list.append(["".join(word) for word in line])

        # Tokenize with padding and truncation
        tokenized_output = self.tokenizer(
            sentences_list, 
            padding="max_length", 
            truncation=True, 
            max_length=self.max_len,  
            return_tensors="pt", 
            is_split_into_words=True,
            return_token_type_ids = False
        )
        tags = pad_sequence([torch.tensor(tag, dtype=torch.long) for tag in tags], batch_first=True, padding_value=-1)
        tags = torch.nn.functional.pad(tags, (0, self.max_len - tags.shape[1]), value=-1)
        return tokenized_output["input_ids"], tags, tokenized_output["attention_mask"]


    def get_embedding(self, token_list, attention_masks): 
        '''
        Get BioBERT embeddings using precomputed attention masks from dataloading.
        Args:
            tokens: List of tokenized sequences
        Returns: 
            embeddings_list: List of BioBERT embeddings for a batch
            attention_masks_list: List of attention masks for a batch
        '''
        self.bert.eval()
    
        with torch.no_grad():
            outputs = self.bert(token_list, attention_mask=attention_masks)
    
        return outputs.last_hidden_state

        # Save embeddings and attention masks
        #np.save(f"{self.embeddings_path}\\{self.dataset_name}\\_BioBERT_embeddings.npy", embeddings_list)
        #np.save(f"{self.embeddings_path}\\{self.dataset_name}\\_BioBERT_attention_masks.npy", attention_masks_list)
        #print(f"Processed {len(embeddings_list)} sentences.  Embeddings and attention masks saved!")

'''
class Embedding_bioELMo(Embedding):
    def __init__(self, embedding_model_name, embeddings_path, dataset_name, max_len=256):
        super(Embedding_bioELMo, self).__init__(embedding_model_name, embeddings_path, dataset_name, max_len)
        # Load BioELMo Model
        #TODO provjetiti jel ovo radi, a mozda i fiksno skinuti tezine i options file dostupne na https://github.com/Andy-jqa/bioelmo?tab=readme-ov-file
        options_file = "https://allennlp.s3.amazonaws.com/models/elmo/biomed_elmo_options.json"
        weight_file = "https://allennlp.s3.amazonaws.com/models/elmo/biomed_elmo_weights.hdf5"
        self.elmo = Elmo(options_file, weight_file, num_output_representations=1, dropout=0)
        self.embedding_dim = 1024  # Dimensionality of BioELMo embeddings
        self.max_len = max_len
        
    def tokenize_and_pad_text(self, text):
    #TODO dodati padding tu
        return text''' #elmos tokenisation is words

    #def get_embedding(self, tokens, attention_masks):  
'''
        Get BioELMo embeddings and attention masks tensors with dimensions (batch_size, max_len, embedding_dim) and (batch_size, max_len)  for a list of tokens
        Attention masks are used to differentiate between real tokens and padding tokens.
        Args:
            tokens: List of tokens
        Returns: 
            embeddings_list: List of BioELMo embeddings
            attention_masks_list: List of attention masks
'''
'''
        self.elmo.eval() 
        # Store sentence embeddings
        embeddings_list = []
        attention_masks_list = []
        sentences = []

        # Process each line in dataset
        for token, attn_mask in zip(tokens, attention_masks):  
            # Split into multiple sentences
            sentences.append(" ".join(token))
            attention_masks_list.append(attention_mask)
            
        # Tokenize each sentence
        tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]
            
        # Convert to BioELMo embeddings
        character_ids = self.elmo.batch_to_ids(tokenized_sentences)  # Convert tokens to ELMo format
        with torch.no_grad():
            embeddings = self.elmo(character_ids)["elmo_representations"][0]  # Get token-level embeddings

        #return embeddings_list #, sentence_list
        # Save embeddings
        #np.save(f"{self.embeddings_path}\\{self.dataset_name}\\_BioELMo_embeddings.npy", embeddings_list)
        #np.save(f"{self.embeddings_path}\\{self.dataset_name}\\_BioELMo_attention_masks.npy", attention_masks_list)
        #print(f"Processed {len(embeddings_list)} sentences. Embeddings and attention masks saved!")
        return torch.stack(embeddings_list), torch.stack(attention_masks_list) '''