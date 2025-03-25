import torch
import torch.nn as nn
from TorchCRF import CRF

class BiRNN_CRF(nn.Module):
    def __init__(self, num_tag, model_args, embedding_dim): #TODO: embedding_dim more biti zbroj dimenzija word i char embeddinga
        super(BiRNN_CRF, self).__init__()
        self.num_tag = num_tag

        self.cell = model_args['cell']
        self.hidden_size = model_args['hidden_size']
        self.num_layers = model_args['num_layers']   
        self.dropout = model_args['dropout']
        self.use_crf = model_args['use_crf']
        self.criterion = model_args['loss']
        self.embedding_dim = embedding_dim

        if self.cell == 'lstm':
            self.rnn = nn.LSTM(self.embedding_dim, self.hidden_size, num_layers=self.num_layers, bidirectional=True, batch_first=True)
        elif self.cell == 'gru':    
            self.rnn = nn.GRU(self.embedding_dim, self.hidden_size, num_layers=self.num_layers, bidirectional=True, batch_first=True)
        else:       
            raise ValueError(f"Cell {self.cell} not supported")

        self.dropout_tag = nn.Dropout(self.dropout)
        
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
    def forward(self, embedding, target_tag, attention_masks): #TODO: add char_embedding
        '''
        Forward pass of the model, computes the loss 
        Args:
            embedding: Embedding tensor with dimensions (batch_size, max_len, embedding_dim)
            target_tag: Target tag tensor with dimensions (batch_size, max_len)
            attention_masks: Attention masks tensor with dimensions (batch_size, max_len)

        Returns:
            loss: Loss value of crf or cross entropy, if crf is used, the loss is token mean
        '''
        #embedding = torch.cat((word_embedding, char_embedding), dim=1) #spojiti embeddinge
        h, _ = self.rnn(embedding)

        o_tag = self.dropout_tag(h)
        tag = self.hidden2tag_tag(o_tag)

        if self.crf_tag:
            mask = torch.squeeze(attention_masks, -2).bool() #has to be in shape (batch_size, sequence_size)
            loss = -self.crf_tag.forward(tag, target_tag, mask).mean()
        else:  
            loss = self.criterion(tag.view(-1, self.num_tag), target_tag.view(-1))
        
        return loss

    def predict(self, embedding, attention_masks): #TODO: add char_embedding
        '''
        Predict the most likely tag sequence
        Args:
            embedding: Embedding tensor with dimensions (batch_size, max_len, embedding_dim)
            attention_masks: Attention masks tensor with dimensions (batch_size, max_len)
        Returns:    
            tag: Predicted tag tensor with dimensions (batch_size, max_len)
        '''
        #embedding = torch.cat((word_embedding, char_embedding), dim=1) #spojiti embeddinge
        h, _ = self.rnn(embedding)

        o_tag = self.dropout_tag(h)
        tag = self.hidden2tag_tag(o_tag)
        
        if self.crf_tag:
            mask = torch.squeeze(attention_masks, -2).bool()
            tag = self.crf_tag.viterbi_decode(tag, mask)
        else:
            tag = torch.argmax(tag, dim=-1)

        return tag

        
