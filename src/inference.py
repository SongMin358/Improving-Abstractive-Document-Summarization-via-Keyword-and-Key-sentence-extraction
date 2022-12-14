import os
import json
import argparse
from collections import Counter, defaultdict

from tqdm import tqdm
import numpy as np
import evaluate

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import sys
sys.path.append("utils")
sys.path.append("data")
# from ..EditEval.src.metrics.update_rouge import update_rouge
from dataset import SummaryDataset
from util import *

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint_path", type=str)
parser.add_argument("--task_type", type=str)

args = parser.parse_args()



device = "cuda"
finetune_weight = os.path.join(args.checkpoint_path, args.task_type)
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
finetuned_model = AutoModelForSeq2SeqLM.from_pretrained("/data/intern/summarization_with_keyword/Summurization-via-Multi-task-Learning/checkpoint/NAIVE/checkpoint-30")
finetuned_model.eval()
finetuned_model.to(device)
test_dataset = SummaryDataset("test", args.task_type, tokenizer,
                            "data", 1024, 100)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
rouge = evaluate.load('rouge')
rouge_agg = evaluate.load('rouge')
collection = []


with torch.no_grad():
    for b in tqdm(test_dataloader):


        cur_example = b
        # print(cur_example)
        input_ids = cur_example["input_ids"].to(device)
        attention_mask = cur_example["attention_mask"].to(device)
        response = finetuned_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            force_words_ids=[[28152, 12169, 4086, 35]],
            max_length=200,
            num_beams=5
        ).cpu()
        
        
        history =  tokenizer.batch_decode(cur_example["input_ids"], skip_special_tokens=True)
        label = tokenizer.batch_decode(cur_example["labels"], skip_special_tokens=True)
        pred = tokenizer.batch_decode(response, skip_special_tokens=True)
        rouge_score = rouge.compute(references=label, predictions=pred, use_aggregator=False)

        
        # print(sari_score)
        rouge_agg.add_batch(predictions=pred, references=label)
        
        for b_i in range(len(label)):
            collection.append({
                "input": history[b_i],
                "pred": pred[b_i],
                "label": label[b_i],
                "rouge_score": {k:v[b_i] for k,v in rouge_score.items()}
            })
            
       
    
# collection = sorted(collection, key=lambda x: x[1])

print(finetune_weight)


with open(os.path.join(finetune_weight,"results_agg_temp_new.json"), "w") as f:
    json.dump(
        {
        "metric":{
            "rouge": dict(rouge_agg.compute())
            },
        "prediction":collection
        }, f
    
    )
                
