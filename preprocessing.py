import nltk
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
#from allennlp import Elmo
from abc import ABC, abstractmethod

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
    
class Embedding_bioBERT(Embedding):
    def __init__(self, embedding_model_name, embeddings_path, dataset_name, max_len=256):
        super(Embedding_bioBERT, self).__init__(embedding_model_name, embeddings_path, dataset_name, max_len)
        self.embedding_dim = 768  # Dimensionality of BioBERT embeddings
        model_name = "dmis-lab/biobert-base-cased-v1.1"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)

    def get_embedding(self, tokens, attention_masks): 
        '''
        Get BioBERT embeddings using precomputed attention masks from the dataset.
        Args:
            tokens: List of tokenized sequences
            attention_masks: Precomputed attention masks from data loading
        Returns: 
            embeddings_list: List of BioBERT embeddings
            attention_masks_list: List of attention masks for a batch
        '''
        self.bert.eval()
        embeddings_list = []
        attention_masks_list = []

        # Process each line in the dataset
        for token, attn_mask in zip(tokens, attention_masks):             
            sentence = " ".join(token)

            # Tokenize using BioBERT's tokenizer
            inputs = self.tokenizer(sentence, return_tensors="pt", padding="max_length", truncation=True, max_length=self.MAX_LEN)
            input_ids = inputs["input_ids"]

            #Ensure `attention_mask` from dataset matches the model's expected format
            attn_mask = torch.reshape(attn_mask, [1, -1]) # Ensure correct shape: (1, max_len)

            # Get BioBERT embeddings
            with torch.no_grad():
                outputs = self.bert(input_ids, attention_mask=attn_mask)

            # Extract last hidden state
            hidden_states = outputs.last_hidden_state
            bioBERT_embeddings = hidden_states.squeeze(0)

            embeddings_list.append(bioBERT_embeddings)
            attention_masks_list.append(attn_mask)

        # Save embeddings and attention masks
        #np.save(f"{self.embeddings_path}\\{self.dataset_name}\\_BioBERT_embeddings.npy", embeddings_list)
        #np.save(f"{self.embeddings_path}\\{self.dataset_name}\\_BioBERT_attention_masks.npy", attention_masks_list)
        #print(f"Processed {len(embeddings_list)} sentences.  Embeddings and attention masks saved!")
        return torch.stack(embeddings_list), torch.stack(attention_masks_list)

'''
class Embedding_bioELMo(Embedding):
    def __init__(self, embedding_model_name, embeddings_path, dataset_name, max_len=256):
        super(Embedding_bioELMo, self).__init__(embedding_model_name, embeddings_path, dataset_name, max_len)
        # Load BioELMo Model
        #TODO provjetiti jel ovo radi, a mozda i fiksno skinuti tezine i options file dostupne na https://github.com/Andy-jqa/bioelmo?tab=readme-ov-file
        options_file = "https://allennlp.s3.amazonaws.com/models/elmo/biomed_elmo_options.json"
        weight_file = "https://allennlp.s3.amazonaws.com/models/elmo/biomed_elmo_weights.hdf5"
        self.elmo = Elmo(options_file, weight_file, num_output_representations=1, dropout=0)
        self.embedding_dim = 1024  # Dimensionality of BioELMo embeddings'''

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