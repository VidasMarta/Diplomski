import torch
import torch.nn as nn
from TorchCRF import CRF
import math

class BiRNN_CRF(nn.Module):
    # TODO: proučiti multitask segment binarne klasifikacije (focal ili dice loss)
    def __init__(self, num_tag, model_args, word_embedding_dim, char_embedding_dim = None): 
        super(BiRNN_CRF, self).__init__()
        self.num_tag = num_tag

        self.cell = model_args['cell']
        self.hidden_size = model_args['hidden_size']
        self.num_layers = model_args['num_layers']   
        self.dropout = model_args['dropout']
        self.use_crf = model_args['use_crf']
        self.criterion = model_args['loss']
        self.attention = model_args['attention']
            
        self.embedding_dim = word_embedding_dim + char_embedding_dim if char_embedding_dim != None else word_embedding_dim

        if self.cell == 'lstm':
            self.rnn = nn.LSTM(self.embedding_dim, self.hidden_size, num_layers=self.num_layers, bidirectional=True, batch_first=True)
        elif self.cell == 'gru':    
            self.rnn = nn.GRU(self.embedding_dim, self.hidden_size, num_layers=self.num_layers, bidirectional=True, batch_first=True)
        else:       
            raise ValueError(f"Cell {self.cell} not supported")

        self.dropout_tag = nn.Dropout(self.dropout)

        if self.attention:
            self.attention_layer = nn.MultiheadAttention(self.hidden_size*2, model_args['att_num_of_heads'], batch_first=True) # *2 because of bidirectional
        
        self.hidden2tag_tag = nn.Linear(self.hidden_size*2, self.num_tag) # *2 because of bidirectional

        if self.use_crf:
            self.crf_tag = CRF(self.num_tag)
        else:
            if self.criterion == 'cross_entropy':
                self.criterion = nn.CrossEntropyLoss(ignore_index=-1) #to ignore padding in loss computation
            #mozda dodati jos neke loss funkcije
            else:
                raise ValueError(f"Loss {model_args['loss']} not supported")

    
    # Return the loss only, does not decode tags
    def forward(self, word_embedding, target_tag, attention_mask, char_embedding = None): 
        '''
        Forward pass of the model, computes the loss 
        Args:
            embedding: Embedding tensor with dimensions (batch_size, max_len, embedding_dim)
            target_tag: Target tag tensor with dimensions (batch_size, max_len)
            attention_mask: Attention masks tensor with dimensions (batch_size, max_len)

        Returns:
            loss: Loss value of crf or cross entropy, if crf is used, the loss is token mean
        '''
        mask = attention_mask.bool()
        if char_embedding != None:
            embedding = torch.cat((word_embedding, char_embedding), dim=-1) #spojiti embeddinge
        else:
            embedding = word_embedding
        h, _ = self.rnn(embedding)
        o_tag = self.dropout_tag(h)

        if self.attention:
            padding_mask = torch.where(mask == True, False, True) #key_padding_mask expects True on indexes that should be ignored
            attention_output, _ = self.attention_layer(o_tag, o_tag, o_tag, key_padding_mask = padding_mask)  
            tag = self.hidden2tag_tag(attention_output)
        else:
            tag = self.hidden2tag_tag(o_tag)

        if self.use_crf:
            loss = -self.crf_tag.forward(tag, target_tag, mask).mean()
        else:  
            loss = self.criterion(tag.view(-1, self.num_tag), target_tag.view(-1))
        
        return loss

    def predict(self, word_embedding, attention_mask, char_embedding = None): 
        '''
        Predict the most likely tag sequence
        Args:
            embedding: Embedding tensor with dimensions (batch_size, max_len, embedding_dim)
            attention_masks: Attention masks tensor with dimensions (batch_size, max_len)
        Returns:    
            tag: Predicted tag tensor with dimensions (batch_size, max_len)
        '''
        mask = attention_mask.bool()
        if char_embedding != None:
            embedding = torch.cat((word_embedding, char_embedding), dim=-1) #spojiti embeddinge
        else:
            embedding = word_embedding
        h, _ = self.rnn(embedding)
        o_tag = self.dropout_tag(h)

        if self.attention:
            padding_mask = torch.where(mask == True, False, True)
            attention_output, _ = self.attention_layer(o_tag, o_tag, o_tag, key_padding_mask = padding_mask)  
            tag = self.hidden2tag_tag(attention_output)
        else:
            tag = self.hidden2tag_tag(o_tag)
            
        if self.use_crf:
            tags = self.crf_tag.viterbi_decode(tag, mask)
            tag = [[torch.tensor(t) for t in tag] for tag in tags]
        else:
            tag = torch.argmax(tag, dim=-1)

        return tag

