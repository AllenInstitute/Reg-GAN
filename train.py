import argparse
from contextlib import nullcontext

import wandb

import torch.cuda
from yaml import CLoader

from trainer import Cyc_Trainer
import yaml

def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream, Loader=CLoader)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='Yaml/CycleGan.yaml', help='Path to the config file.')
    opts = parser.parse_args()
    config = get_config(opts.config)
    
    if config['name'] == 'CycleGan':
        trainer = Cyc_Trainer(config)
    else:
        raise ValueError(f'{config["name"]} not supported')

    if torch.cuda.is_available():
        trainer = trainer.cuda()

    if config['use_wandb']:
        wandb_run = wandb.init(
            project=config['wandb_project'],
            entity=config['wandb_entity'],
            config=config
        )
    else:
        wandb_run = nullcontext()

    with wandb_run:
        trainer.train()
    
    



###################################
if __name__ == '__main__':
    main()