from __future__ import print_function
import argparse, os, json, traceback, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from PIL import Image
from copy import deepcopy

import albumentations as A
import numpy as np
import pandas as pd
from tokenizers import AddedToken
from torch.utils.data import Dataset
from transformers import Pix2StructProcessor
from transformers import DataCollatorWithPadding
from transformers import Pix2StructConfig, Pix2StructForConditionalGeneration
#from accelerate import Accelerator

#---------------------------------------------------  
#MODEL
#---------------------------------------------------

class BenetechModel(nn.Module):
    """
    The Benetech model
    """

    def __init__(self, hyperparams):
        print("initializing the Benetech model...")

        super(BenetechModel, self).__init__()
        self.hyperparams = hyperparams
        self.device = hyperparams.device
        
        backbone_config = Pix2StructConfig.from_pretrained(hyperparams.backbone_path)
        backbone_config.text_config.max_length = hyperparams.max_length
        backbone_config.text_config.is_decoder = True

        backbone_config.text_config.pad_token_id = hyperparams.pad_token_id
        backbone_config.text_config.decoder_start_token_id = hyperparams.decoder_start_token_id
        backbone_config.text_config.bos_token_id = hyperparams.bos_token_id

        # backbone_config.decoder.max_length = cfg.model.max_length

        self.backbone = Pix2StructForConditionalGeneration.from_pretrained(
            hyperparams.backbone_path,
            config=backbone_config,
        )

        # resize model embeddings
        print("resizing model embeddings...")
        print(f"tokenizer length = {hyperparams.len_tokenizer}")
        self.backbone.decoder.resize_token_embeddings(hyperparams.len_tokenizer)

        self.loss_fn = nn.CrossEntropyLoss(
            ignore_index=-100,
            reduction="mean",
        ).to(self.device)
        
        

        # self.backbone.encoder.gradient_checkpointing_enable()
        # self.backbone.decoder.gradient_checkpointing_enable()

    def forward(
            self,
            flattened_patches,
            attention_mask,
            labels,
    ):
        
        flattened_patches , attention_mask, labels = flattened_patches.to(self.device), attention_mask.to(self.device), labels.to(self.device) 
  
        outputs = self.backbone(
            flattened_patches=flattened_patches,
            attention_mask=attention_mask,
            labels=labels,
        )

        loss = outputs.loss

        return loss