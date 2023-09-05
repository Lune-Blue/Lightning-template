import os
import sys
import json
import yaml
import argparse
import torch
import torch.nn as nn
import pytorch_lightning as pl
import wandb
import transformers

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, LearningRateFinder
from typing import Optional, Dict, Sequence, List, Union
from transformers import set_seed

from data.dataclass import ProjectArguments, ModelArguments, DataArguments, SchedulingArguments, TrainingArguments,LoRaArguments, GenerationArguments
from utils.utility import get_accelerate_model, get_best_checkpoint
from data.dataset import BaseDataModule
from model.BackboneModel import DefaultModel
from model.trainer import DefaultModelTrainer

import warnings 
warnings.filterwarnings("ignore", category=UserWarning)   


print("aa")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/train.yaml')
    parser.add_argument('--config_name', type=str, default='base')
    parser.add_argument('--use_accelerate', action='store_true', help='using accelerate model')
    args = parser.parse_args()
    return args

### Learning Rate Finder: Experimental feature
class FineTuneLearningRateFinder(LearningRateFinder):
    def __init__(self, milestones, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.milestones = milestones

    def on_fit_start(self, *args, **kwargs):
        return

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch in self.milestones or trainer.current_epoch == 0:
            self.lr_find(trainer, pl_module)

def get_ModelCheckpoint(args):
    if args.full_finetune:
        lr = args.lr
        val_checkpoint = ModelCheckpoint(
            dirpath=args.save_dir,
            monitor='val_loss',
            filename=f"lr-{lr}"+'epoch-{epoch:02d}val_loss:{val_loss:.2f}',
            save_top_k=2,
            mode='min',
        )
        return val_checkpoint
    else:
        print('Saving PEFT checkpoint...')
        lr = args.lr
        val_checkpoint = ModelCheckpoint(
            dirpath = os.path.join(args.save_dir, 'adapter'),
            monitor='val_loss',
            filename="adapter_model"+f"lr-{lr}"+'epoch-{epoch:02d}val_loss:{val_loss:.2f}',
            save_top_k=1,
            mode='min',
        )        
        return val_checkpoint
    
    
def gather_checkpoint(args):
    lr_monitor = LearningRateMonitor(logging_interval='step')
    early_stopping = EarlyStopping('val_loss')
    model_checkpoint = get_ModelCheckpoint(args)
    if args.use_lrfinder:
        learningRateFinder = FineTuneLearningRateFinder(milestones=(5, 10))
        return [lr_monitor, early_stopping, model_checkpoint, learningRateFinder]
    else:
        return [lr_monitor, early_stopping, model_checkpoint]
    


def main(args):
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)[args.config_name]
    hfparser = transformers.HfArgumentParser((
        ProjectArguments, ModelArguments, DataArguments, 
        SchedulingArguments, TrainingArguments, LoRaArguments, GenerationArguments
    ))
    project_args, model_args, data_args, schedule_args, training_args, lora_args, generation_args = hfparser.parse_dict(config)
    training_args.generation_config = transformers.GenerationConfig(**vars(generation_args))

    args = argparse.Namespace(
        **vars(project_args),**vars(model_args), **vars(data_args),
        **vars(schedule_args), **vars(training_args), **vars(lora_args),
        **vars(generation_args), **vars(args)
    )
    wandb_logger = WandbLogger(project=args.wandb_project,
                               name=args.wandb_name, save_dir=args.save_dir)
    if args.use_accelerate:
        model, tokenizer = get_accelerate_model(args, None)
        model = DefaultModelTrainer(model, args)
    else:
        model = DefaultModel(args)
        model = DefaultModelTrainer(model, args)
        tokenizer = model.model.get_tokenizer()
             
    set_seed(args.random_state)
    callback_list = gather_checkpoint(args)
    
    trainer = pl.Trainer(callbacks=callback_list, devices=args.devices, accelerator='gpu',
                         max_epochs=args.max_epochs, logger=wandb_logger, strategy=args.strategy, log_every_n_steps=1)   
    datamodule = BaseDataModule(args, tokenizer)
    print("done")
    if args.do_train:
        trainer.fit(model, datamodule=datamodule)
    if args.do_test:
        if args.use_accelerate:
            checkpoint_dir, flag = get_best_checkpoint(os.path.join(args.save_dir, 'adapter'))
        else:
            checkpoint_dir, flag = get_best_checkpoint(args.save_dir)
        if not flag:
            print("checkpoint not detected!")
            return False
        trainer.test(model, datamodule=datamodule, ckpt_path=checkpoint_dir)
        
        
        """
        *the way of loading model*
        
        # load the model from the checkpoint
        model = MyLightningModule.load_from_checkpoint("/path/to/checkpoint.ckpt")
        
        # disable randomness, dropout, etc...
        model.eval()
        
        # predict with the model
        y_hat = model(x)
        """
        


    
if __name__ == '__main__':
    args = get_args()
    main(args)