import os

class Logger:
    def __init__(self, model_logs_path):
        self.output_path = model_logs_path
        if not os.path.exists(self.output_path):
            os.mkdir(model_logs_path)

        open(self.output_path+'/train.log',"w").close()
        open(self.output_path+'/valid.log',"w").close()
        open(self.output_path+'/test.log',"w").close()

    def log_test_results(self, loss, f1_score, precision, recall):
        log_file = open(self.output_path + '/test.log',"a")
        log_file.write(f"loss: {loss}, f1_score: {f1_score}, precison: {precision}, recall:{recall}")
        log_file.close()

    def log_val_results(self, epoch, loss, f1_score):
        log_file = open(self.output_path + '/valid.log',"a")
        log_file.write(f"epoch: {epoch}, loss: {loss}, f1_score: {f1_score}")
        log_file.close()

    def log_train_loss(self, epoch, loss):
        log_file = open(self.output_path + 'train.log', "a")
        log_file.write(f"epoch: {epoch}, loss: {loss}")
        log_file.close()


