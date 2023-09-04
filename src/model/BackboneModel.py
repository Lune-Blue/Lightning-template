import sys
from typing import Dict, Optional
import torch
import torch.nn as nn
import transformers
from transformers import BitsAndBytesConfig
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    set_seed, 
    Seq2SeqTrainer,
    BitsAndBytesConfig,

)

class DefaultModel(nn.Module):    
    def __init__(self, model_args):
        super(DefaultModel, self).__init__()

        self.model_args = model_args
        
        # Get Model
        if self.model_args.model_4bit: # for inference
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_args.model_name_or_path,
                quantization_config=bnb_config,
                device_map="auto"
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_args.model_name_or_path,
                torch_dtype=torch.float32
            )
        
        # Get Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_args.model_name_or_path,
            padding_side="right",
            truncation_sied=model_args.truncation_side,
            use_fast=True,
        )
        if self.tokenizer._pad_token is None:
                
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=self.tokenizer,
                model=self.model,
            )

    def forward(self, labels, **kwargs):
        if labels is None:
            return self.model(**kwargs)
        else:
            return self.model(**kwargs, labels=labels)

    def get_tokenizer(self):
        return self.tokenizer
    
    def generate(self, **kwargs):
        return self.model.generate(**kwargs)
    


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg
    
