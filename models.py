import torch
import torch.nn as nn
from TorchCRF import CRF

class BiRNN_CRF(nn.Module):
    #TODO: dodati attention mechanism (https://medium.com/@eugenesh4work/attention-mechanism-for-lstm-used-in-a-sequence-to-sequence-task-be1d54919876)
    # i/ili multitask segment binarne klasifikacije (focal ili dice loss)
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
            # Attention layer: compute attention weights over time steps
            #This is sentence-level attention (https://medium.com/swlh/a-simple-overview-of-rnn-lstm-and-attention-mechanism-9e844763d07b similar to this, only feed into CRF not rnn-decoder)
            # TODO: proučiti (i implementirati (?)) token-level attention
            self.attention_layer = nn.Linear(2*self.hidden_size, 1)
            self.hidden2tag_tag = nn.Linear(self.hidden_size*4, self.num_tag) # *2 because of concatenation of h and context vector
        else:
            self.hidden2tag_tag = nn.Linear(self.hidden_size*2, self.num_tag) # *2 because of bidirectional

        if self.use_crf:
            self.crf_tag = CRF(self.num_tag)
        else:
            if self.criterion == 'cross_entropy':
                self.criterion = nn.CrossEntropyLoss(ignore_index=-1) #to ignore padding in loss computation
            #mozda dodati jos neke loss funkcije
            else:
                raise ValueError(f"Loss {model_args['loss']} not supported")
            
    def _compute_sent_level_attention(self, h, mask):
        attn_scores = self.attention_layer(h).squeeze(-1)  
        attn_scores = attn_scores.masked_fill(mask == 0, -1e9)    # Mask out padded tokens
        attn_weights = nn.functional.softmax(attn_scores, dim=1)     
        context = torch.bmm(attn_weights.unsqueeze(1), h) 
        context = context.repeat(1, h.size(1), 1)  # so that each h_i has the same sentence context concatenated        
        return context

    
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
        if char_embedding != None:
            embedding = torch.cat((word_embedding, char_embedding), dim=-1) #spojiti embeddinge
        else:
            embedding = word_embedding
        h, _ = self.rnn(embedding)
        o_tag = self.dropout_tag(h)

        if self.attention:
            context = self._compute_sent_level_attention(o_tag, attention_mask)  
            combined = torch.cat([o_tag, context], dim=-1)
            tag = self.hidden2tag_tag(combined)
        else:
            tag = self.hidden2tag_tag(o_tag)

        if self.use_crf:
            mask = attention_mask.bool()
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
        if char_embedding != None:
            embedding = torch.cat((word_embedding, char_embedding), dim=-1) #spojiti embeddinge
        else:
            embedding = word_embedding
        h, _ = self.rnn(embedding)
        o_tag = self.dropout_tag(h)

        if self.attention:
            context = self._compute_sent_level_attention(o_tag, attention_mask)  
            combined = torch.cat([o_tag, context], dim=-1)
            tag = self.hidden2tag_tag(combined)
        else:
            tag = self.hidden2tag_tag(o_tag)
            
        if self.use_crf:
            mask = attention_mask.bool()
            tags = self.crf_tag.viterbi_decode(tag, mask)
            tag = [[torch.tensor(t) for t in tag] for tag in tags]
        else:
            tag = torch.argmax(tag, dim=-1)

        return tag
