import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from tools.logger import Logger
import logging


class PhiSLM:
    def __init__(self, debug=False):
        self.logger = Logger(name="Phi", level=logging.DEBUG if debug else logging.INFO)
        self.model_name = "microsoft/Phi-3-mini-4k-instruct"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Device set to use: {self.device}")

        try:
            trust_remote_code = True
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=trust_remote_code)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
                trust_remote_code=trust_remote_code,
                ).eval().to(self.device)

            self.logger.info(f"Model {self.model_name} loaded successfully.")

        except Exception as e:
            self.logger.error(f"Error loading model {self.model_name}: {e}")
            self.logger.error("Please ensure the model name is correct and you have internet access for the initial download.")
            self.logger.error("If using a large model, ensure you have sufficient GPU VRAM.")
            raise
        
        try:
            if self.tokenizer.pad_token is None:
                if self.tokenizer.eos_token is not None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token 
                elif self.tokenizer.unk_token is not None:
                    self.tokenizer.pad_token = self.tokenizer.unk_token 
                else:
                    # As a last resort, add a new padding token
                    self.logger.warning("Warning: No EOS or UNK token found, adding a new PAD token.")
                    self.tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
                    self.model.resize_token_embeddings(len(self.tokenizer))
        except Exception as e:
            self.logger.error(f"Error setting padding token: {e}")
            raise
    
    def generate_embeddings(self, inputs: list[str]):
        '''
        Obtain the vector that represents the entire input, consolidating individual token vectors into one.
        Applies mean pooling
        '''
        tokenized = self.tokenizer(inputs, padding=True, truncation=True, return_tensors="pt").to(self.device)
        self.logger.debug(f"Tokenized input shape (input_ids): {tokenized['input_ids'].shape}")
        self.logger.debug(f"Attention mask shape: {tokenized['attention_mask'].shape}")

        # Use 'with torch.no_grad():' to disable gradient calculation
        # output_hidden_states=True is essential to get the layer outputs
        with torch.no_grad():
            outputs = self.model(**tokenized, output_hidden_states=True)
        
        last_hidden_states = outputs.hidden_states[-1]
        self.logger.debug(f"Shape of last hidden states (sequence embeddings per token): {last_hidden_states.shape}")
        self.logger.debug(f"Hidden size of the model: {last_hidden_states.shape[-1]}")

        mask = tokenized['attention_mask'].unsqueeze(-1).expand_as(last_hidden_states) 
        masked_hidden_states = last_hidden_states * mask

        sum_hidden_states = masked_hidden_states.sum(dim=1)

        actual_lengths = tokenized['attention_mask'].sum(dim=1)

        actual_lengths = torch.clamp(actual_lengths, min=1)

        mean_pooled_embeddings = sum_hidden_states / actual_lengths.unsqueeze(-1)

        return mean_pooled_embeddings
    