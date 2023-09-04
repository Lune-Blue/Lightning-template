import torch
import torch.nn as nn
import json
import numpy as np
import pytorch_lightning as pl
from packaging import version
import importlib
import warnings
import os
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed,
    Seq2SeqTrainer,
    BitsAndBytesConfig,
    LlamaTokenizer
)
import sys
sys.path.append('../')

IGNORE_INDEX = -100

class DefaultModelTrainer(pl.LightningModule):
    def __init__(self, models, args):
        super(DefaultModelTrainer, self).__init__()
        self.save_hyperparameters(args)
        
        self.args = args
        self.model = models
        
        # Initialize the loss function
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
        self.training_step_outputs = []
        self.valid_step_outputs = []
        self.test_step_outputs = []
        
    def forward(self, **kwargs):
        return self.model(**kwargs)
    
    
    def training_step(self, batch, batch_idx):
        labels = batch['labels']
        output = self.model(**batch)
        
        if output.loss is not None:
            loss = output.loss
        else:
            # output target is to shift the labels to the right 1 token
            output_target = output[:,1:-1,:].contiguous()
            label_target = labels[:,1:].contiguous()
            loss = self.loss_fn(output_target.view(-1, output_target.size(-1)), label_target.view(-1))
        self.log('train_loss', loss, logger=True, sync_dist=True)
        self.training_step_outputs.append(loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        labels = batch['labels']
        output = self.model(**batch)
        
        if output.loss is not None:
            loss = output.loss
        else:
            # output target is to shift the labels to the right 1 token
            output_target = output[:,1:-1,:].contiguous()
            label_target = labels[:,1:].contiguous()
            loss = self.loss_fn(output_target.view(-1, output_target.size(-1)), label_target.view(-1))
        self.log('val_loss', loss, logger=True, sync_dist=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        input_img = batch['input_img']
        input_ids = batch['input_ids']
        outputs = self.model.generate(input_img,input_ids, max_new_tokens=self.args.max_new_tokens, top_p=self.args.top_p, top_k=self.args.top_k, do_sample=self.args.do_sample, temperature=self.args.temperature)
        output_seq = self.model.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        decoded_input = self.model.tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)
        
        save_data = {
            "batch_idx": batch_idx,
            "predictions": output_seq,
            "inputs": decoded_input,
        }
        
        # Save data to JSON file
        with open(self.args.save_prediction_file, 'a') as file:
            json.dump(save_data, file)
            file.write('\n')  # Write newline after each JSON object for easier reading
            
        return None
    
    def on_train_epoch_end(self):
        epoch_mean = torch.stack(self.training_step_outputs).mean()
        self.log("training_epoch_mean_loss", epoch_mean)
        self.training_step_outputs.clear()
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr)
        if self.args.scheduler_type == 'ReduceLROnPlateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=self.args.scheduler_mode, factor=self.args.factor, patience=self.args.patience)
            return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}
        else: # scheduler_type == 'CyclicLR'
            scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=self.args.lr, max_lr=self.args.max_lr, step_size_up=self.opt.step_size_up, cycle_momentum=False)
            
        return [optimizer], [scheduler]