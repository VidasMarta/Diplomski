import os

class LossLogger:
    def __init__(self, model_logs_path):
        self.output_path = model_logs_path
        if not os.path.exists(model_logs_path):
            os.mkdir(model_logs_path)

        open(model_logs_path+'/train.log',"w").close()
        open(model_logs_path+'/valid.log',"w").close()
        open(model_logs_path+'/test.log',"w").close()

    def log_losses(self, file_name, epoch, loss, f1_score):
        log_file = open(self.model_logs_path+'/'+file_name,"a")
        log_file.write(str(epoch) + ',' + str(loss) + ',' + ',' + str(f1_score) + '\n')
        log_file.close()
