import os
import json
from datetime import datetime

class Logger:
    def __init__(self, model_logs_path, model_args, settings_args):
        self.output_path = model_logs_path
        self.init_time = datetime.now()
        if not os.path.exists(self.output_path):
            os.mkdir(model_logs_path)

        #prepare files for log writing
        open(self.output_path+'/train.log',"w").close()
        open(self.output_path+'/valid.log',"w").close()
        open(self.output_path+'/test.log',"w").close()

        #write config (hyperparameters etc.)
        log_config_file = open(self.output_path+'/config.log', "w")
        json.dump(model_args, log_config_file)
        log_config_file.write("\n")
        json.dump(settings_args, log_config_file)

        log_config_file.close()


    def log_test_results(self, loss, f1_score, precision, recall, f1_score_strict, precision_strict, recall_strict):
        log_file = open(self.output_path + '/test.log',"a")
        log_file.write(f"loss: {loss}, f1_score: {f1_score}, precison: {precision}, recall:{recall}, f1_score(strict): {f1_score_strict}, precision(strict): {precision_strict}, recall(strict): {recall_strict}, time: {datetime.now() - self.init_time}\n")
        log_file.close()

    def log_val_results(self, epoch, loss, f1_score, f1_score_strict):
        log_file = open(self.output_path + '/valid.log',"a")
        log_file.write(f"epoch: {epoch}, loss: {loss}, f1_score: {f1_score}, f1_score(strict): {f1_score_strict}, time: {datetime.now() - self.init_time}\n")
        log_file.close()

    def log_train_loss(self, epoch, loss):
        log_file = open(self.output_path + '/train.log', "a")
        log_file.write(f"epoch: {epoch}, loss: {loss}, time: {datetime.now() - self.init_time}\n")
        log_file.close()


