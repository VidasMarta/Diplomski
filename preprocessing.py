import nltk
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from allennlp.modules.elmo import Elmo
from abc import ABC, abstractmethod

#TODO dodati CNN charachter based embeddings
class Embedding(ABC):
    def __init__(self, embedding_model_name, embeddings_path, dataset_name, max_len=256):
        self.dataset_name = dataset_name
        self.embeddings_path = embeddings_path
        self.MAX_LEN = max_len
        self.embedding_model_name = embedding_model_name
        self.embedding_dim = None

    @staticmethod
    def create(embedding_model_name, embeddings_path, dataset_name, max_len=256):
        if embedding_model_name == 'bioBERT':
            return Embedding_bioBERT(embedding_model_name, embeddings_path, dataset_name, max_len)
        elif embedding_model_name == 'bioELMo':
            return Embedding_bioELMo(embedding_model_name, embeddings_path, dataset_name, max_len)
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

    def get_embedding(self, tokens): 
        '''
        Get BioBERT embeddings and attention masks tensors with dimensions (batch_size, max_len, embedding_dim)  for a list of tokens
        Attention masks are used to differentiate between real tokens and padding tokens.
        Args:
            tokens: List of tokens
        Returns: 
            embeddings_list: List of BioBERT embeddings
            attention_masks_list: List of attention masks
        '''
        self.bert.eval()
        embeddings_list = []
        attention_masks_list = []

        # Process each line in the dataset
        for token in tokens:             
            sentence = " ".join(token)
            # Tokenize using BioBERT's tokenizer
            inputs = self.tokenizer(sentence, return_tensors="pt", padding="max_length", truncation=True, max_length=self.MAX_LEN)
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            # Get BioBERT embeddings
            with torch.no_grad():
                outputs = self.bert(input_ids, attention_mask=attention_mask)

            # Extract last hidden state
            hidden_states = outputs.last_hidden_state
            bioBERT_embeddings = hidden_states.squeeze(0)

            embeddings_list.append(bioBERT_embeddings)
            attention_masks_list.append(attention_mask)

        # Save embeddings and attention masks
        #np.save(f"{self.embeddings_path}\\{self.dataset_name}\\_BioBERT_embeddings.npy", embeddings_list)
        #np.save(f"{self.embeddings_path}\\{self.dataset_name}\\_BioBERT_attention_masks.npy", attention_masks_list)
        #print(f"Processed {len(embeddings_list)} sentences.  Embeddings and attention masks saved!")
        return torch.stack(embeddings_list), torch.stack(attention_masks_list)


class Embedding_bioELMo(Embedding):
    def __init__(self, embedding_model_name, embeddings_path, dataset_name, max_len=256):
        super(Embedding_bioELMo, self).__init__(embedding_model_name, embeddings_path, dataset_name, max_len)
        # Load BioELMo Model
        options_file = "https://allennlp.s3.amazonaws.com/models/elmo/biomed_elmo_options.json"
        weight_file = "https://allennlp.s3.amazonaws.com/models/elmo/biomed_elmo_weights.hdf5"
        self.elmo = Elmo(options_file, weight_file, num_output_representations=1, dropout=0)
        self.embedding_dim = 1024  # Dimensionality of BioELMo embeddings

    def get_embedding(self, tokens):  
        '''
        Get BioELMo embeddings and attention masks tensors with dimensions (batch_size, max_len, embedding_dim)  for a list of tokens
        Attention masks are used to differentiate between real tokens and padding tokens.
        Args:
            tokens: List of tokens
        Returns: 
            embeddings_list: List of BioELMo embeddings
            attention_masks_list: List of attention masks
        '''
        self.elmo.eval() 
        # Store sentence embeddings
        embeddings_list = []
        attention_masks_list = []
        sentences = []

        # Process each line in dataset
        for token in tokens:
            # Split into multiple sentences
            sentences.append(" ".join(token))
            
        # Tokenize each sentence
        tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]
            
        # Convert to BioELMo embeddings
        character_ids = self.elmo.batch_to_ids(tokenized_sentences)  # Convert tokens to ELMo format
        with torch.no_grad():
            embeddings = self.elmo(character_ids)["elmo_representations"][0]  # Get token-level embeddings

        for embedding in embeddings:
            if embedding.shape[0] > self.MAX_LEN:
                embedding = embedding[:self.MAX_LEN]
                attention_mask = np.ones(self.MAX_LEN)
            else:
                padding = torch.zeros((self.MAX_LEN - embedding.shape[0], self.embedding_dim))
                embedding = torch.cat((embedding, padding), dim=0)
                attention_mask = np.concatenate([np.ones(embedding.shape[0]), np.zeros(self.MAX_LEN - embedding.shape[0])])
 
            embeddings_list.append(embedding.numpy())
            attention_masks_list.append(attention_mask)

        #return embeddings_list #, sentence_list
        # Save embeddings
        #np.save(f"{self.embeddings_path}\\{self.dataset_name}\\_BioELMo_embeddings.npy", embeddings_list)
        #np.save(f"{self.embeddings_path}\\{self.dataset_name}\\_BioELMo_attention_masks.npy", attention_masks_list)
        #print(f"Processed {len(embeddings_list)} sentences. Embeddings and attention masks saved!")
        return torch.stack(embeddings_list), torch.stack(attention_masks_list) 