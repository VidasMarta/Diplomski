import os
import seqeval.metrics
import yaml
import datasets
import models
from settings import Settings
import torch
#from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score
import numpy as np
import matplotlib.pyplot as plt
import seqeval
from itertools import chain


# TODO: mozda koristiti seqval (https://github.com/chakki-works/seqeval/tree/master) za evaluaciju
class Evaluation:
    def __init__(self, tagging_scheme = 'IOB1'):
        self.tagging_scheme = tagging_scheme

    

    def evaluate(self, data_loader, model, device, embeddings_model, num_to_tag_dict): #, loss_logger, log_file_name, epoch = 0, test=False):
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

                # Remove padding only from true tags 
                unpadded_true_tags = [[t for t in seq if int(t) != -1] for seq in tags]
                tokenss = [[t for t,m in zip(tok, mask) if m] for tok, mask in zip(tokens, att_mask)]

                for true, pred, tok in zip(unpadded_true_tags, pred_tags, tokenss):
                    if len(true) != len(pred):
                        print(f"Inconsistent: {len(true)} and {len(pred)}, tokens len: {len(tok)}")
                        print(true)
                        print(pred)

                all_true_tags.extend(unpadded_true_tags) 
                all_pred_tags.extend(pred_tags)

            # save to log file
            #loss_logger.log_losses(file_name=log_file_name, epoch=epoch, loss=final_loss/len(data_loader), 
            #                       f1_score = f1_score(y_true=all_true_tags, y_pred=all_pred_tags, average='micro'))
            
            #for seqeval tags have to be strings
            unpadded_true_string_tags = [[num_to_tag_dict[int(t)] for t in seq] for seq in all_true_tags]

            # Predicted tags already have no padding, so just map them to strings
            predicted_string_tags = [[num_to_tag_dict[int(t)] for t in seq] for seq in all_pred_tags]

            print(seqeval.metrics.classification_report(unpadded_true_string_tags, predicted_string_tags, scheme=self.tagging_scheme))
 
            print(f"Test Loss = {final_loss / len(data_loader)}")
            '''print(classification_report(all_true_tags, all_pred_tags)) #, labels=list(self.tags.values()), target_names=list(self.tags.keys())))
            cm = confusion_matrix(all_true_tags, all_pred_tags)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm) #, display_labels=list(self.tags.keys()))
            disp.plot()
            plt.show()'''