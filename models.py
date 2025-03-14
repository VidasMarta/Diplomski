import torch
import transformers
import torch.nn as nn
from torchcrf import CRF

class BiRNN_CRF(nn.Module):
    def __init__(self, num_tag, model_args, embedding):
        super(BiRNN_CRF, self).__init__()
        self.num_tag = num_tag

        self.cell = model_args['cell']
        self.hidden_size = model_args['hidden_size']
        self.num_layers = model_args['num_layers']   
        self.dropout = model_args['dropout']
        self.use_crf = model_args['use_crf']
        self.criterion = model_args['loss']
        self.embedding = embedding

        #self.bert = transformers.BertModel.from_pretrained(embedding_model_path, return_dict=False)
        if self.cell == 'lstm':
            self.rnn = nn.LSTM(self.embedding.embedding_dim, self.hidden_size, num_layers=self.num_layers, bidirectional=True, batch_first=True)
        elif self.cell == 'gru':    
            self.rnn = nn.GRU(self.embedding.embedding_dim, self.hidden_size, num_layers=self.num_layers, bidirectional=True, batch_first=True)
        else:       
            raise ValueError(f"Cell {self.cell} not supported")

        self.dropout_tag = nn.Dropout(self.dropout)
        
        self.hidden2tag_tag = nn.Linear(self.hidden_size, self.num_tag)

        if self.use_crf:
            self.crf_tag = CRF(self.num_tag, batch_first=True)
        else:
            if self.criterion == 'cross_entropy':
                self.criterion = nn.CrossEntropyLoss()
            #mozda dodati jos neke loss funkcije
            else:
                raise ValueError(f"Loss {model_args['loss']} not supported")
    
    # Return the loss only, does not decode tags
    def forward(self, ids, mask, token_type_ids, target_tag):
        #TODO: embedding nadopuni argumente
        x, _ = self.embedding() #self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)
        h, _ = self.bilstm(x)

        o_tag = self.dropout_tag(h)
        tag = self.hidden2tag_tag(o_tag)

        if self.crf_tag:
            mask = torch.where(mask == 1, True, False)
            loss_tag = -self.crf_tag(tag, target_tag, mask=mask, reduction='token_mean')
            loss = loss_tag
        else:  
            loss = self.criterion(tag.view(-1, self.num_tag), target_tag.view(-1))
        
        return loss

    # Encodes the tags, does not return loss
    def encode(self, ids, mask, token_type_ids, target_tag):
        #TODO: embedding nadopuni argumente
        x, _ = self.embedding() # self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)
        h, _ = self.rnn(x)

        o_tag = self.dropout_tag(h)
        tag = self.hidden2tag_tag(o_tag)
        
        if self.crf_tag:
            mask = torch.where(mask == 1, True, False)
            tag = self.crf_tag.decode(tag, mask=mask)

        return tag

        
