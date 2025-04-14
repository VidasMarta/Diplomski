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
        self.crf_attention_mask = None

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
        
    def _pad_or_truncate(self, seq, pad, pad_value):
        if seq.size(0) < self.max_len:
            return nn.functional.pad(input=seq, pad=pad, value=pad_value)
        else:
            return seq[:self.max_len]
        
    @abstractmethod
    def get_embedding(self, tokens):
        pass

    @abstractmethod
    def tokenize_and_pad_text(self, text, tags):
        pass

    @abstractmethod
    def get_relevant_tags(self, tags, num_to_tag_dict):
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
        all_input_ids = []
        all_padded_tags = []
        all_attention_masks = []
        all_word_level_masks = [] #needed for crf

        for sentence, sentence_tags in zip(text, tags):
            encoding = self.tokenizer(
                sentence,
                is_split_into_words=True,
                truncation=True,
                padding='max_length',
                max_length=self.max_len,
                return_tensors='pt',
                return_attention_mask=True,
            )

            input_ids = encoding["input_ids"][0]
            attention_mask = encoding["attention_mask"][0]
            word_ids = encoding.word_ids(batch_index=0)

            # Create word-level mask: 1 for first subword of each word
            word_level_mask = []
            prev_word_id = None
            for word_id in word_ids:
                if word_id is None or word_id == prev_word_id:
                    word_level_mask.append(0)
                else:
                    word_level_mask.append(1)
                    prev_word_id = word_id

            # Pad word-level mask to max_len
            word_level_mask = torch.tensor(
                self._pad_or_truncate(torch.tensor(word_level_mask), (0, self.max_len - len(word_level_mask)), 0)
            )

            # add -1 tag for cls and sep tokens and pad tags
            new_tags = []
            new_tags.append(-1)
            new_tags.extend(sentence_tags)
            new_tags.append(-1)
            new_tags = torch.tensor(new_tags)
            padded_tags = self._pad_or_truncate(new_tags, (0, self.max_len - new_tags.shape[0]), -1)

            all_input_ids.append(input_ids)
            all_attention_masks.append(attention_mask)
            all_padded_tags.append(padded_tags)
            all_word_level_masks.append(word_level_mask)

        return torch.stack(all_input_ids), torch.stack(all_padded_tags), torch.stack(all_attention_masks), torch.stack(all_word_level_masks)


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

    def get_relevant_tags(self, tags, num_to_tag_dict, word_level_masks):
        all_relevant_tags = []
        for tag_seq, mask in zip(tags, word_level_masks):
            relevant_tags = []
            for tag, is_first_subword in zip(tag_seq, mask):
                if is_first_subword and tag != -1:
                    relevant_tags.append(num_to_tag_dict[int(tag)])
            all_relevant_tags.append(relevant_tags)
        return all_relevant_tags

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

        
    def tokenize_and_pad_text(self, text, tags): 
        sentences_list = [["".join(word) for word in line] for line in text]
        tokenized_text = batch_to_ids(sentences_list)

        tensor_tags = [torch.tensor(t) for t in tags] 

        tokens_padded = torch.stack([self._pad_or_truncate(seq, (0, 0, 0, self.max_len - seq.shape[0]), 0) for seq in tokenized_text])  #TODO  
        tags_padded = torch.stack([self._pad_or_truncate(seq, (0, self.max_len - seq.shape[0]), -1) for seq in tensor_tags])         

        padding_mask = torch.where(tags_padded != -1, 1, 0)  

        self.crf_attention_mask = padding_mask
        return tokens_padded, tags_padded, None, padding_mask #embedding doesn't need attention mask

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

    def get_relevant_tags(self, tags, num_to_tag_dict, mask):
        return [[num_to_tag_dict[int(tag)] for tag in seq if int(tag) != -1] for seq in tags]


class CharEmbeddingCNN(nn.Module): #For char embeddings
    def __init__(self, vocab, emb_size, feature_size, max_word_length, kernel_sizes = [7, 3], dropout=0.1): #, args, number_of_classes):
        super(CharEmbeddingCNN, self).__init__()
       
        self.vocab = vocab if "<UNK>" in vocab else vocab + ["<UNK>"]
        self.vocab_size = len(self.vocab)
        self.char_to_idx = {char: idx for idx, char in enumerate(self.vocab)}
        self.unk_idx = self.char_to_idx["<UNK>"]
        self.max_word_length = max_word_length
        self.embedding_dim = emb_size

        # Define deep CNN layers (inspired by https://github.com/ahmedbesbes/character-based-cnn/blob/master/src/model.py)
        self.dropout_input = nn.Dropout2d(dropout)

        self.conv1 = nn.Sequential(
            nn.Conv1d(self.vocab_size, feature_size, kernel_size=kernel_sizes[0]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(feature_size, feature_size, kernel_size=kernel_sizes[0]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(feature_size, feature_size, kernel_size=kernel_sizes[1]),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(feature_size, feature_size, kernel_size=kernel_sizes[1]),
            nn.ReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(feature_size, feature_size, kernel_size=kernel_sizes[1]),
            nn.ReLU()
        )
        self.conv6 = nn.Sequential(
            nn.Conv1d(feature_size, feature_size, kernel_size=kernel_sizes[1]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3)
        )

        with torch.no_grad():
            dummy = torch.zeros(1, self.vocab_size, self.max_word_length)
            out = self._forward_conv(dummy)
            self.flatten_dim = out.view(1, -1).shape[1]

        self.fc = nn.Linear(self.flatten_dim, emb_size)

    def _forward_conv(self, x):
        x = self.dropout_input(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
    
    def _word_to_ohe(self, word): # make oh representation of a word and pad to max_word_len
        idxs = [self.char_to_idx.get(c, self.unk_idx) for c in word[:self.max_word_length]] #get oh encoding for each char in word
        one_hot = torch.eye(self.vocab_size)[idxs].T  # make 0/1 matrix with shape (vocab_size, len)
        padded = nn.functional.pad(one_hot, (0, self.max_word_length - one_hot.shape[1]))  # (vocab_size, max_word_length)
        return padded
   
    # preraden kod s https://www.kaggle.com/code/anubhavchhabra/character-level-word-embeddings-using-1d-cnn
    def batch_cnn_embedding_generator(self, text, batch_size):                 
        for i in range(0, len(text), batch_size):
            batch_sentences = text[i:i + batch_size]
            max_sent_len = max(len(sent) for sent in batch_sentences)

            word_tensors = []
            for sentence in batch_sentences:
                sent_tensors = []
                for word in sentence[:max_sent_len]:
                    ohe = self._word_to_ohe(word).unsqueeze(0)
                    sent_tensors.append(ohe)
                while len(sent_tensors) < max_sent_len:
                    sent_tensors.append(torch.zeros((1, self.vocab_size, self.max_word_length)))
                word_tensors.append(torch.cat(sent_tensors, dim=0))  # (max_sent_len, vocab_size, max_word_length)

            batch_tensor = torch.stack(word_tensors)  # (batch_size, max_sent_len, vocab_size, max_word_length)
            batch_tensor = batch_tensor.view(-1, self.vocab_size, self.max_word_length)  # (B * L, V, W)
            embeddings = self.forward(batch_tensor)  # (B * L, emb_size)
            embeddings = embeddings.view(len(batch_sentences), max_sent_len, -1)  # (B, L, emb_size)
            yield embeddings

from models import BiRNN_CRF
# Example usage
if __name__ == "__main__":
    # Dummy tag mapping
    tag2num = {'O': 0, 'B-Disease': 1, 'I-Disease': 2}
    num2tag = {v: k for k, v in tag2num.items()}

    # Dummy data: two tokenized sentences and corresponding tags
    sentences = [
        ["The", "patient", "was", "diagnosed", "with", "pneumonia", "."],
        ["He", "has", "diabetes", "mellitus"]
    ]

    tags = [
        [0, 0, 0, 0, 0, 1, 0],   # "pneumonia" is a disease
        [0, 0, 1, 2]             # "diabetes mellitus" is a multi-token entity
    ]

    # Create the embedding instance
    embedder = Embedding.create("bioBERT", "dummy_dataset", max_len=16)

    # Get token ids, tag ids, attention masks, and word-level masks
    input_ids, padded_tags, attention_masks, word_level_mask = embedder.tokenize_and_pad_text(sentences, tags)

    # Get embeddings from BioBERT
    embeddings = embedder.get_embedding(input_ids, attention_masks)

    # Verify output shapes
    print(f"Input IDs shape: {input_ids.shape}")            # [batch, max_len]
    print(f"Padded Tags shape: {padded_tags.shape}")        # [batch, max_len]
    print(f"Attention Mask shape: {attention_masks.shape}") # [batch, max_len]
    print(f"Embeddings shape: {embeddings.shape}")          # [batch, max_len, 768]
    print(f"Word-level mask: {word_level_mask.shape}")      # [batch, max_len]


    # Convert back predicted tags for evaluation/debugging
    decoded_tags = embedder.get_relevant_tags(padded_tags, num2tag, word_level_mask)
    print(f"Decoded tags: {decoded_tags}")
    
    model_args = dict()
    model_args['cell'] = 'lstm'
    model_args['hidden_size'] = 512
    model_args['num_layers'] = 1
    model_args['dropout'] = 0.3
    model_args['use_crf'] = True
    model_args['loss'] = "CRF"

    model = BiRNN_CRF(3, model_args, embedder.embedding_dim)
    logits = model(embeddings, padded_tags, attention_masks)
    preds = model.predict(embeddings, attention_masks)

    pred_tags = embedder.get_relevant_tags(preds, num2tag, word_level_mask) #[[num2tag[int(tag)] for tag in seq ] for seq in preds]

    print(f"Predicted tags: {pred_tags}")

