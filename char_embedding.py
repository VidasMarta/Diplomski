import torch.nn as nn
import torch
from torch.nn.functional import pad

from datasets import DatasetLoader, get_max_len
import settings

#preuzeto s: https://github.com/ahmedbesbes/character-based-cnn/tree/master
class CharEmbeddingCNN(nn.Module):
    def __init__(self, input_len, embed_size, kernel_size, max_length): #, args, number_of_classes):
        super(CharEmbeddingCNN, self).__init__()

        self.seq = nn.Sequential(
            nn.Conv1d(in_channels=input_len, out_channels=embed_size, kernel_size=kernel_size, bias=False),
            nn.Tanh(),
            nn.MaxPool1d(kernel_size=max_length-kernel_size+1)
        )

    def forward(self, x):
        return self.seq(x).squeeze()


        '''

        # define conv layers

        self.dropout_input = nn.Dropout2d(args.dropout_input)

        self.conv1 = nn.Sequential(
            nn.Conv1d(
                args.number_of_characters + len(args.extra_characters),
                256,
                kernel_size=7,
                padding=0,
            ),
            nn.ReLU(),
            nn.MaxPool1d(3),
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=7, padding=0), nn.ReLU(), nn.MaxPool1d(3)
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, padding=0), nn.ReLU()
        )

        self.conv4 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, padding=0), nn.ReLU()
        )

        self.conv5 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, padding=0), nn.ReLU()
        )

        self.conv6 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, padding=0), nn.ReLU(), nn.MaxPool1d(3)
        )

        # compute the  output shape after forwarding an input to the conv layers

        input_shape = (
            128,
            args.max_length,
            args.number_of_characters + len(args.extra_characters),
        )
        self.output_dimension = self._get_conv_output(input_shape)

        # define linear layers

        self.fc1 = nn.Sequential(
            nn.Linear(self.output_dimension, 1024), nn.ReLU(), nn.Dropout(0.5)
        )

        self.fc2 = nn.Sequential(nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(0.5))

        self.fc3 = nn.Linear(1024, number_of_classes)

        # initialize weights

        self._create_weights()

    # utility private functions

    def _create_weights(self, mean=0.0, std=0.05):
        for module in self.modules():
            if isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
                module.weight.data.normal_(mean, std)

    def _get_conv_output(self, shape):
        x = torch.rand(shape)
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = x.view(x.size(0), -1)
        output_dimension = x.size(1)
        return output_dimension

    # forward

    def forward(self, x):
        x = self.dropout_input(x)
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x'''

    
#doraden kod s https://www.kaggle.com/code/anubhavchhabra/character-level-word-embeddings-using-1d-cnn
def get_batched_cnn_embedding(text, vocab, batch_size, emb_size,  kernel_size, max_sentence_length, max_word_length):    
    vocab += "<UNK>"

    model = CharEmbeddingCNN(len(vocab), emb_size, kernel_size, max_word_length)
        
    char_to_idx_map = {char: idx for idx, char in enumerate(vocab)}
    unk_index = len(vocab) - 1 

    ohe_characters = torch.eye(n=len(vocab))

    all_batches = []

    for i in range(0, len(text), batch_size):
        batch_sentences = text[i:i + batch_size]
        batch_embeddings = []
        for words in batch_sentences:
            ohe_words = torch.empty(size=(0, len(vocab), max_word_length))
            for word in words:
                idx_representation = [char_to_idx_map.get(char, unk_index) for char in word] 
                ohe_representation = ohe_characters[idx_representation].T # Shape: (vocab_size, word_length)
                padded_ohe_representation = pad(ohe_representation, (0, max_word_length-len(word)))
                ohe_words = torch.cat((ohe_words, padded_ohe_representation.unsqueeze(dim=0))) #Shape: (num_words, vocab_size, max_word_length)

            if len(ohe_words) > max_sentence_length:
                ohe_words = ohe_words[:max_sentence_length]
            elif 0 < len(ohe_words) < max_sentence_length:
                ohe_words = torch.cat((
                    ohe_words, 
                    torch.zeros((max_sentence_length - len(ohe_words), len(vocab), max_word_length)))
                )
            elif len(ohe_words) == 0:
                ohe_words = torch.zeros(max_sentence_length, len(vocab))

            embedding = model(ohe_words)
            batch_embeddings.append(embedding) 

        batch_embeddings = torch.stack(batch_embeddings)
        all_batches.append(batch_embeddings)

    all_batches = torch.stack(all_batches) 

    return all_batches, len(vocab)

# Example usage
if __name__ == "__main__":
    dataset_loader = DatasetLoader("ncbi_disease_json", settings.DATA_PATH)
    total_tags, (text_train, tags_train), (text_val, tags_val), (text_test, tags_test) = dataset_loader.load_data()
    
    max_sentence_len = get_max_len(text_train, text_val, text_test)
    max_word_len = 7 #TODO: Ovo je za sad proizvoljno, istraziti ima li neka praksa
    vocab="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"

    train_ohe_batches = get_batched_cnn_embedding(text_train[0:64], vocab, 32, 256, 3, max_sentence_len, max_word_len)

    for batch_embedding in train_ohe_batches:
        print(batch_embedding.shape)
        print(batch_embedding[0].shape)

        break



