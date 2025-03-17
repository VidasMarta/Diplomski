import os
import yaml
import datasets
import models
from settings import Settings
import torch
from sklearn.metrics import classification_report


# staviti da se spremaju u neki log file metrike po epohama ili tako nesto
class Evaluation:
    def __init__(self, tags):
        self.tags = tags

    def evaluate(self, data_loader, model, device, embeddings_model):
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

                       
            print(f"Test Loss = {final_loss / len(data_loader)}")
            print(classification_report(all_true_tags, all_pred_tags, target_labels=list(self.tags.values()), target_names=list(self.tags.keys())))