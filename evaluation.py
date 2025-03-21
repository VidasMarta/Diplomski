import os
import yaml
import datasets
import models
from settings import Settings
import torch
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score
import numpy as np
import matplotlib.pyplot as plt
import seqeval


# TODO: mozda koristiti seqval (https://github.com/chakki-works/seqeval/tree/master) za evaluaciju
class Evaluation:
    def __init__(self, tags):
        self.tags = tags

    def evaluate(self, data_loader, model, device, embeddings_model, labels): #, loss_logger, log_file_name, epoch = 0, test=False):
        '''
        Evaluate the model on the test set
        Args:
            data_loader: DataLoader object with test data
            model: Model object
            device: Device to run the model on
            embeddings_model: Embedding object
            test: set True if test data evaluation, default=False
        '''
        model = model.to(device)    
        model.eval()  # Set model to evaluation mode
        all_true_tags = []
        all_pred_tags = []
        with torch.no_grad():
            final_loss = 0
            for tokens, tags, att_mask in data_loader: # tqdm(data_loader, total=len(data_loader)):
                batch_embeddings = embeddings_model.get_embedding(tokens, att_mask)
                batch_embeddings = batch_embeddings.to(device)
                batch_attention_masks = att_mask.to(device)
                batch_tags = tags.to(device)

                loss = model(batch_embeddings, batch_tags, batch_attention_masks)
                final_loss += loss.item()

                pred_tags = model.predict(batch_embeddings, batch_attention_masks)

                # Padding pred_tags to match the length of the longest sequence in the batch
                padded_pred_tags = []
                for i, tag_seq in enumerate(pred_tags):
                    # Pad each sequence to match the length of att_mask
                    padding_length = len(att_mask[i]) - len(tag_seq)
                    padded_seq = list(tag_seq) + [-1] * padding_length
                    padded_pred_tags.append(padded_seq)

                padded_pred_tags = np.array(padded_pred_tags)

                all_true_tags.extend(tags.numpy().flatten()) #flatten so that the batch dimension is removed
                all_pred_tags.extend(padded_pred_tags.flatten())  

            # save to log file
            #loss_logger.log_losses(file_name=log_file_name, epoch=epoch, loss=final_loss/len(data_loader), 
            #                       f1_score = f1_score(y_true=all_true_tags, y_pred=all_pred_tags, average='micro'))

            if 1: #test:
                print(f"Test Loss = {final_loss / len(data_loader)}")
                print(classification_report(all_true_tags, all_pred_tags)) #, labels=list(self.tags.values()), target_names=list(self.tags.keys())))
                cm = confusion_matrix(all_true_tags, all_pred_tags)
                disp = ConfusionMatrixDisplay(confusion_matrix=cm) #, display_labels=list(self.tags.keys()))
                disp.plot()
                plt.show()