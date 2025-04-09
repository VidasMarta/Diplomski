import torch
from transformers import AutoTokenizer, AutoModel
from allennlp.modules.elmo import Elmo, batch_to_ids
from abc import ABC, abstractmethod
import os
import settings
import torch.nn as nn

class Embedding(ABC): #For word embeddings
    def __init__(self, embedding_model_name, dataset_name, max_len=256):
        self.dataset_name = dataset_name
        self.max_len = max_len
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
    
class Embedding_bioBERT(Embedding): #TODO: dodati i tezine za large (https://github.com/naver/biobert-pretrained)
    def __init__(self, embedding_model_name, dataset_name, max_len=256):
        super(Embedding_bioBERT, self).__init__(embedding_model_name, dataset_name, max_len)
        self.max_len = max_len
        self.embedding_dim = 768  # Dimensionality of BioBERT embeddings
        bioBERT_setup_path = os.path.join(settings.EMBEDDINGS_PATH, "bioBERT_setup") 
        self.tokenizer = AutoTokenizer.from_pretrained(bioBERT_setup_path)
        self.bert = AutoModel.from_pretrained(bioBERT_setup_path)

    def tokenize_and_pad_text(self, text, tags):
        # Convert list of token lists into full sentences
        sentences_list = [["".join(word) for word in line] for line in text]
        all_input_ids = []
        all_padded_tags = []
        all_attention_masks = []

        for words, tags in zip(sentences_list, tags):
            word_piece_ids = [] #ids of tokenized words
            aligned_tags = [] 

            word_piece_ids.append(self.tokenizer.cls_token_id) #add cls token in the beginning of a sentence
            aligned_tags.append(-1)  # tag for CLS is padding -1

            for word, tag in zip(words, tags):
                tokens = self.tokenizer.tokenize(word)
                token_ids = self.tokenizer.convert_tokens_to_ids(tokens)

                if len(word_piece_ids) + len(token_ids) + 1 >= self.max_len: #make sure that len of current sentence embedding (+1 for sep) is not begger than max_len 
                    break 
                    
                word_piece_ids.extend(token_ids) 
                aligned_tags.extend([tag] * len(token_ids))  # Repeat tag for each subword

            
            word_piece_ids.append(self.tokenizer.sep_token_id) #add sep token in the beginning of a sentence
            aligned_tags.append(-1)  #tag for SEP is padding

            # Add padding to match max_len
            padding_length = self.max_len - len(word_piece_ids)
            word_piece_ids.extend([self.tokenizer.pad_token_id] * padding_length)
            aligned_tags.extend([-1] * padding_length)
            
            #compute attention_mask
            attention_mask = [1 if idx != self.tokenizer.pad_token_id else 0 for idx in word_piece_ids]

            all_input_ids.append(torch.tensor(word_piece_ids))
            all_padded_tags.append(torch.tensor(aligned_tags))
            all_attention_masks.append(torch.tensor(attention_mask))

        return torch.stack(all_input_ids), torch.stack(all_padded_tags), torch.stack(all_attention_masks)


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

    
    def _pad_or_truncate(self, seq, pad, pad_value):
        if seq.size(0) < self.max_len:
            return nn.functional.pad(input=seq, pad=pad, value=pad_value)
        else:
            return seq[:self.max_len]

        
    def tokenize_and_pad_text(self, text, tags): 
        sentences_list = [["".join(word) for word in line] for line in text]
        tokenized_text = batch_to_ids(sentences_list)

        tensor_tags = [torch.tensor(t) for t in tags] 

        tokens_padded = torch.stack([self._pad_or_truncate(seq, (0, 0, 0, self.max_len - seq.shape[0]), 0) for seq in tokenized_text])    
        tags_padded = torch.stack([self._pad_or_truncate(seq, (0, self.max_len - seq.shape[0]), -1) for seq in tensor_tags])         

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


class CharEmbeddingCNN(nn.Module): #For char embeddings
    def __init__(self, vocab, emb_size,  kernel_size, max_word_length): #, args, number_of_classes):
        super(CharEmbeddingCNN, self).__init__()
        self.vocab = vocab
        self.vocab += "<UNK>"
        self.max_word_length = max_word_length

        self.seq = nn.Sequential(
            nn.Conv1d(in_channels=len(self.vocab), out_channels=emb_size, kernel_size=kernel_size, bias=False),
            nn.Tanh(),
            nn.MaxPool1d(kernel_size=max_word_length-kernel_size+1)
        )

    def forward(self, x):
        return self.seq(x).squeeze()
   
    # preraden kod s https://www.kaggle.com/code/anubhavchhabra/character-level-word-embeddings-using-1d-cnn
    def batch_cnn_embedding_generator(self, text, max_sentence_length, batch_size):                 
        char_to_idx_map = {char: idx for idx, char in enumerate(self.vocab)}
        unk_index = len(self.vocab) - 1 

        ohe_characters = torch.eye(n=len(self.vocab))

        for i in range(0, len(text), batch_size):
            batch_sentences = text[i:i + batch_size]
            batch_embeddings = []
            for words in batch_sentences:
                ohe_words = torch.empty(size=(0, len(self.vocab), self.max_word_length))
                for word in words:
                    idx_representation = [char_to_idx_map.get(char, unk_index) for char in word] 
                    ohe_representation = ohe_characters[idx_representation].T # Shape: (vocab_size, word_length)
                    padded_ohe_representation = nn.functional.pad(input=ohe_representation, pad=(0, self.max_word_length-len(word)))
                    ohe_words = torch.cat((ohe_words, padded_ohe_representation.unsqueeze(dim=0))) #Shape: (num_words, vocab_size, max_word_length)

                if len(ohe_words) > max_sentence_length:
                    ohe_words = ohe_words[:max_sentence_length]
                elif 0 < len(ohe_words) < max_sentence_length:
                    ohe_words = torch.cat((
                        ohe_words, 
                        torch.zeros((max_sentence_length - len(ohe_words), len(self.vocab), self.max_word_length)))
                    )
                elif len(ohe_words) == 0:
                    ohe_words = torch.zeros(max_sentence_length, len(self.vocab))

                embedding = self.forward(ohe_words)
                batch_embeddings.append(embedding) 

            batch_embeddings = torch.stack(batch_embeddings)

            yield batch_embeddings