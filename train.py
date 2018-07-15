import re
import os
import torch
import datetime
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from utils import save, load
from config import JsonConfig


class Trainer(object):
    def __init__(self, graph, optim, lrschedule, loaded_step,
                       devices, data_device,
                       dataset, hparams):
        if isinstance(hparams, str):
            hparams = JsonConfig(hparams)
        # set members
        # append date info
        date = str(datetime.datetime.now())
        date = date[:date.rfind(":")].replace("-", "").replace(":", "").replace(" ", "_")
        self.log_dir = os.path.join(hparams.Dir.log_root, "log_" + date)
        self.checkpoints_dir = os.path.join(self.log_dir, "checkpoints")
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        # write hparams
        hparams.dump(self.log_dir)
        if not os.path.exists(self.checkpoints_dir):
            os.makedirs(self.checkpoints_dir)
        self.checkpoints_gap = hparams.Train.checkpoints_gap
        self.max_checkpoints = hparams.Train.max_checkpoints
        self.graph = graph
        self.optim = optim
        # grad operation
        self.max_grad_clip = hparams.Train.max_grad_clip
        self.max_grad_norm = hparams.Train.max_grad_norm
        # copy devices from built graph
        self.devices = devices
        self.data_device = data_device
        # tensorboard
        self.writer = SummaryWriter(log_dir=self.log_dir)
        # number of training batches
        self.batch_size = hparams.Train.batch_size
        self.data_loader = DataLoader(dataset,
                                      batch_size=self.batch_size,
                                      shuffle=True,
                                      drop_last=True)
        self.n_epoches = (hparams.Train.num_batches + len(self.data_loader) - 1) // len(self.data_loader)
        self.global_step = 0
        # lr schedule
        self.lrschedule = lrschedule
        self.loaded_step = loaded_step

        # log
        self.scalar_log_gaps = 50
        self.plot_gaps = 1000
        self.inference_gap = 100
        
    def train(self):
        # set to training state
        self.graph.train()
        self.global_step = self.loaded_step
        # begin to train
        for epoch in range(self.n_epoches):
            print("epoch", epoch)
            progress = tqdm(self.data_loader)
            for i_batch, batch in enumerate(progress):
                # update learning rate
                lr = self.lrschedule["func"](global_step=self.global_step, **self.lrschedule["args"])
                for param_group in self.optim.param_groups:
                    param_group['lr'] = lr
                self.optim.zero_grad()
                if self.global_step % self.scalar_log_gaps == 0:
                    self.writer.add_scalar("lr/lr", lr, self.global_step)
                # to device
                for k in batch:
                    batch[k] = batch[k].to(self.data_device)
                # forward phase
                z, objective = self.graph(x=batch["x"], y_onehot=batch["y"])
                
                # loss
                loss = self.graph.loss_generative(objective)
                if self.global_step % self.scalar_log_gaps == 0:
                    self.writer.add_scalar("loss/loss_generative", loss, self.global_step)

                # backward
                self.graph.zero_grad()
                self.optim.zero_grad()
                loss.backward()
                # operate grad
                if self.max_grad_clip is not None and self.max_grad_clip > 0:
                    torch.nn.utils.clip_grad_value_(self.graph.parameters(), self.max_grad_clip)
                if self.max_grad_norm is not None and self.max_grad_norm > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.graph.parameters(), self.max_grad_norm)
                    if self.global_step % self.scalar_log_gaps == 0:
                        self.writer.add_scalar("grad_norm/grad_norm", grad_norm, self.global_step)
                # step
                self.optim.step()

                # checkpoints
                if self.global_step % self.checkpoints_gap == 0 and self.global_step > 0:
                    save(global_step=self.global_step,
                         graph=self.graph,
                         optim=self.optim,
                         pkg_dir=self.checkpoints_dir,
                         is_best=True,
                         max_checkpoints=self.max_checkpoints)
                if self.global_step % self.plot_gaps == 0:
                    pass

                # inference
                if hasattr(self, "inference_gap"):
                    if self.global_step % self.inference_gap == 0:
                        img = self.graph(y_onehot=batch["y"], reverse=True)
                        for bi in range(len(img)):
                            self.writer.add_image("infer/{}".format(bi), img[bi], self.global_step)

                # global step
                self.global_step += 1

        self.writer.export_scalars_to_json(os.path.join(self.log_dir, "all_scalars.json"))
        self.writer.close()
