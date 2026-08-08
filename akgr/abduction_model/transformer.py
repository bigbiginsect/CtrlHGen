import argparse

import json
import pandas as pd

import math
import torch
import torch.nn as nn

import warnings

# transformer (huggingface T5)
# from transformers import T5Tokenizer

from transformers import T5Config, T5ForConditionalGeneration
# from transformers import BartConfig, BartForConditionalGeneration
from transformers import GPT2Config, GPT2LMHeadModel
from akgr.abduction_model.t5 import myT5
# from akgr.abduction_model.Bart import myBart
# from akgr.abduction_model.gpt2 import myGPT2LM

# from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
# from transformers import DataCollatorForLanguageModeling


def create_gpt2_config(tokenizer, model_config: dict) -> GPT2Config:
    """Build the Phase A GPT-2 config without network/local-template access."""
    required = {"n_layer", "n_embd", "n_head", "n_positions", "n_ctx"}
    missing = required - set(model_config)
    if missing:
        raise ValueError(f"Missing explicit GPT-2 config keys: {sorted(missing)}")
    if int(model_config["n_embd"]) % int(model_config["n_head"]):
        raise ValueError("GPT-2 n_embd must be divisible by n_head")
    vocab_size = len(tokenizer)
    token_ids = tokenizer.get_vocab().values()
    if token_ids and max(token_ids) >= vocab_size:
        raise ValueError("Tokenizer contains an id outside len(tokenizer)")
    return GPT2Config(
        vocab_size=vocab_size,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        n_layer=int(model_config["n_layer"]),
        n_embd=int(model_config["n_embd"]),
        n_head=int(model_config["n_head"]),
        n_positions=int(model_config["n_positions"]),
        n_ctx=int(model_config["n_ctx"]),
    )


def create_reproduction_transformer(tokenizer, model_config: dict) -> GPT2LMHeadModel:
    """Create a randomly initialized, strict-config reproduction model."""
    return GPT2LMHeadModel(create_gpt2_config(tokenizer, model_config))

def create_transformer(ntoken: int, special_tokens: dict,
        model_name: str, config_model: dict):

    # Common configurations (will overwrite pretrained config)
    # https://huggingface.co/docs/transformers/v4.30.0/en/main_classes/configuration#transformers.PretrainedConfig
    common_config = {
        'vocab_size': ntoken + 1, # pad_token is negative
        'pad_token_id': special_tokens['PAD'],
        'bos_token_id': special_tokens['START'],
        'eos_token_id': special_tokens['END'],
        # https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5Config.relative_attention_max_distance
        'decoder_start_token_id': special_tokens['START']
        }
    # Also load customized configurations into it (will overwrite)
    if model_name in config_model:
        common_config.update(config_model[model_name])
        print(f'{model_name} config_model[model_name]: {config_model[model_name]}')
        print(f'common_config: {common_config}')
    else:
        confirmed = input(f'Cannot find config for {model_name} in config_model, press "Y" to continue:')
        if confirmed != 'Y':
            exit()
    # Create transformers
    if 'T5' in model_name:
        if model_name == 'T5-default':
            # https://huggingface.co/docs/transformers/model_doc/t5#training
            config = T5Config.from_pretrained(
                "t5-small",
                **common_config
            )
            transformer = T5ForConditionalGeneration(config)
        # https://huggingface.co/docs/transformers/model_doc/t5#training
        # input: two tensors input_ids=src and labels=tgt
        elif 'T5_disablepos' in model_name:
            config = T5Config.from_pretrained(
                "t5-small",
                **common_config
            )
            transformer = myT5(config)
        else:
            return None
    elif 'GPT2' in model_name:
        # Historical YAML used ``num_layers`` (not a GPT2Config field), so
        # translate it deliberately while keeping this entry point offline.
        legacy_layers = int(common_config.pop("num_layers", common_config.pop("n_layer", 12)))
        config = GPT2Config(
            n_layer=legacy_layers,
            n_embd=int(common_config.pop("n_embd", 768)),
            n_head=int(common_config.pop("n_head", 12)),
            n_positions=int(common_config.pop("n_positions", 1024)),
            n_ctx=int(common_config.pop("n_ctx", 1024)),
            **common_config,
        )
        if 'GPT2_6' in model_name:
            transformer = GPT2LMHeadModel(config)
        else:
            exit()

    # Add attributes
    transformer.model_name = model_name
    return transformer


class TransformerModel(nn.Module):
    def __init__(self, device, ntoken: int, special_tokens: dict,
        model_name: str, config_model: dict):
        super().__init__()
        self.device = device
        self.ntoken = ntoken + 1 # pad_token is negative

        self.pad_token = special_tokens['PAD']
        self.start_token = special_tokens['START']
        self.end_token = special_tokens['END']

        self.model_name = model_name
        # Common configurations (will overwrite pretrained config)
        # https://huggingface.co/docs/transformers/v4.30.0/en/main_classes/configuration#transformers.PretrainedConfig
        common_config = {
            'vocab_size': self.ntoken,
            'pad_token_id': self.pad_token,
            'bos_token_id': self.start_token,
            'eos_token_id': self.end_token,
            # https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5Config.relative_attention_max_distance
            'decoder_start_token_id': self.start_token
        }
        # Also load customized configurations into it (will overwrite)
        if model_name in config_model:
            common_config.update(config_model[model_name])
        if 'T5' in model_name:
            if model_name == 'T5-default':
                # https://huggingface.co/docs/transformers/model_doc/t5#training
                self.config = T5Config.from_pretrained(
                    "t5-small",
                    **common_config
                )
                self.transformer = T5ForConditionalGeneration(self.config)
            elif model_name == 'T5-disable-pos':
                self.config = T5Config.from_pretrained(
                    "t5-small",
                    **common_config
                )
                self.transformer = myT5(self.config)
            # https://huggingface.co/docs/transformers/model_doc/t5#training
            # input: two tensors input_ids=src and labels=tgt
            elif 'T5_disablepos' in model_name:
                self.config = T5Config.from_pretrained(
                    "t5-small",
                    **common_config
                )
                self.transformer = myT5(self.config)
            else:
                return None
        elif 'GPT2' in model_name:
            legacy_layers = int(common_config.pop("num_layers", common_config.pop("n_layer", 12)))
            self.config = GPT2Config(
                n_layer=legacy_layers,
                n_embd=int(common_config.pop("n_embd", 768)),
                n_head=int(common_config.pop("n_head", 12)),
                n_positions=int(common_config.pop("n_positions", 1024)),
                n_ctx=int(common_config.pop("n_ctx", 1024)),
                **common_config,
            )
            # if model_name == 'GPT2-disable-pos':
            #     #self.transformer = myGPT2LM(self.config)
            #     self.transformer = GPT2LMHeadModel(self.config)
            if model_name == 'GPT2_6':
                self.transformer = GPT2LMHeadModel(self.config)

    def forward(self, src, src_pad_mask, tgt):
        """
        Input:
        - src: ... + <PAD>
        - tgt: <START> + ... + <END> + <PAD>
        """
        input_ids = src
        attention_mask = src_pad_mask
        transformer_result = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=tgt
        )
        # logits: Prediction scores of the language modeling head (scores for
        #   each vocabulary token before SoftMax).
        #   of shape (B, S, V)

        logits = transformer_result.logits

        loss = transformer_result.loss
        pred_argmax = logits.argmax(2)

        return logits, loss, pred_argmax

    def generate(self, **kwargs):
        return self.transformer(**kwargs)
