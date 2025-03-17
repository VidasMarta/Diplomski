import os
import yaml
import datasets
import models
from settings import Settings
import torch
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt


# TODO: mozda koristiti seqval (https://github.com/chakki-works/seqeval/tree/master) za evaluaciju
class Evaluation:
    def __init__(self, tags):
        self.tags = tags

    def evaluate(self, data_loader, model, device, embeddings_model):
        '''
        Evaluate the model on the test set
        Args:
            data_loader: DataLoader object with test data
            model: Model object
            device: Device to run the model on
            embeddings_model: Embedding object
        '''
        if device == 'cuda' and not torch.cuda.is_available():
            device = 'cpu'
        model = model.to(device)    
        model.eval()  # Set model to evaluation mode
        all_true_tags = []
        all_pred_tags = []
        with torch.no_grad():
            final_loss = 0
            for data in data_loader: # tqdm(data_loader, total=len(data_loader)):
                batch_embeddings, batch_attention_masks = embeddings_model.get_embedding(data['tokens'])
                batch_embeddings = batch_embeddings.to(device)
                batch_attention_masks = batch_attention_masks.to(device)

                true_tags = data['tags'].to(device)
                loss = model(batch_embeddings, true_tags, batch_attention_masks)
                final_loss += loss.item()

                pred_tags = model.predict(batch_embeddings, batch_attention_masks)

                all_true_tags.extend(true_tags.cpu().numpy().flatten()) #flatten so that the batch dimension is removed
                all_pred_tags.extend(np.array(pred_tags).flatten())

            #TODO dodati da se sprema u neki output file   
            print(f"Test Loss = {final_loss / len(data_loader)}")
            print(classification_report(all_true_tags, all_pred_tags, target_labels=list(self.tags.values()), target_names=list(self.tags.keys())))
            cm = confusion_matrix(all_true_tags, all_pred_tags)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(self.tags.keys()))
            disp.plot()
            plt.show()