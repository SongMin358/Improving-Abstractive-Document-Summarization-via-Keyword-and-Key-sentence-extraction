import tokenizers
import torch
from torch.utils.data import Dataset
import json
import os
import sys
sys.path.append("utils")
from utils import *


class SummaryDataset(Dataset):
    def __init__(self, split, task_type, tokenizer, data_dir, enc_max_len=1024, dec_max_len=100):
        with open(os.path.join(data_dir, f"{split}.json"), "r") as f:
            data = json.load(f)

        self.split = split
        self.collection = data
        self.task_type = task_type
        self.tokenizer = tokenizer
        self.enc_max_len = enc_max_len
        self.dec_max_len = dec_max_len
        assert task_type  in TASK_LIST, f"Invalid Task Type {task_type}"


    def __len__(self):
        return len(self.collection)

    def __getitem__(self, index):
        cur_edit_data = self.collection[index]
        document = cur_edit_data['document']
        summary = cur_edit_data['summary']
        key_words = cur_edit_data['key_words']
        key_sentence = cur_edit_data['key_sentence']

        input_text = ""
        label_text = ""

        if self.task_type == NAIVE:
            input_text += document
            label_text += summary

        elif self.task_type == KW:
            input_text += f"{document}\nKey Words: {key_words}"
            label_text += summary
        
        elif self.task_type == KS:
            input_text += f"{document}\nKey Sentence: {key_sentence}"
            label_text += summary

        elif self.task_type == KWKS:
            input_text += f"{document}\nKey Words: {key_words}\nKey Sentence: {key_sentence}"
            label_text += summary


        encoded_input = self.tokenizer(
            input_text,
            padding="max_length",
            truncation=True,
            max_length=self.enc_max_len,
            return_tensors="pt"
        )
        encoded_label = self.tokenizer(
            label_text,
            padding='max_length',
            truncation=True,
            max_length=self.dec_max_len,
            return_tensors='pt'
        )
        model_inputs = encoded_input
        model_inputs['input_ids'] = model_inputs['input_ids'].squeeze(0)
        model_inputs['attention_mask'] = model_inputs['attention_mask'].squeeze(
            0)
        model_inputs['labels'] = encoded_label['input_ids'].squeeze(0)
        
        return model_inputs 


        