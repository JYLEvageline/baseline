import time
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from Loss import NSNLLLoss

class ModelManager:
    def __init__(self, opt):
        self.opt = opt

    def build_model(self, model_type, dataset):
        u_size = dataset.u_vocab.size()
        v_size = dataset.v_vocab.size()
        t_size = dataset.t_vocab_size
        model = self.init_model(model_type, u_size, v_size, t_size)
        if self.opt['load_model']:
            self.load_model(model, model_type, self.opt['iter'])
            train_time = 0.0
            return model, train_time
        trainer = Trainer(model, self.opt, model_type)
        train_time, best_epoch = trainer.train(dataset.train_loader, dataset.valid_loader, self)
        self.load_model(model, model_type, best_epoch)
        return model, train_time

    def load_model(self, model, model_type, epoch):
        # TODO add
        pass

    def init_model(self, model_type, u_size, v_size, t_size):
        if model_type == 'ncf':
            return 1
        elif model_type == 'decoder':
            return None
        else:
            print 'unknown model type'
            return None

class Trainer:
    def __init__(self, model, opt, model_type):
        self.opt = opt
        self.train_log_file = opt['train_log_file']
        self.n_epoch = opt['n_epoch']
        self.batch_size = opt['batch_size']
        self.model_type = model_type
        self.save_gap = opt['save_gap']
        self.model = model
        self.criterion = NSNLLLoss()
        self.optimizer = opt.Adam(self.model.parameters())

    def train(self, train_data, valid_data, model_manager):
        best_hr1 = 0
        best_epoch = 0
        start = time.time()
        for epoch in xrange(self.n_epoch):
            self.train_one_epoch(train_data, epoch)
            if (epoch + 1) % self.save_gap == 0:
                valid_hr1 = self.valid_one_epoch(valid_data, epoch)
                if valid_hr1 >= best_hr1:
                    best_hr1 = valid_hr1
                    best_epoch = epoch
                    model_manager.save_model(self.model, self.model_type)
        end = time.time()
        return end - start, best_epoch

    def train_one_epoch(self, train_data, epoch):
        total_loss = 0.0
        for i, data_batch in enumerate(train_data):
            self.optimizer.zero_grad()
            session_idx = data_batch[0]
            vids = Variable(data_batch[1])
            tids = Variable(data_batch[2])
            vids_next = Variable(data_batch[3])
            tids_next = Variable(data_batch[4])
            masks = data_batch[7]
            uids = Variable(data_batch[8])
            outputs = self.model(vids, tids, vids_next, tids_next, uids, masks)
            loss = self.criterion(outputs)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.data[0]

