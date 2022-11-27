from transformers import AutoModel, AutoTokenizer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset
import evaluate 
import sys
import numpy as np
import random
sys.path.append("utils/")
from util import split_into_sentences
import torch
from tqdm import tqdm
import json
from torch.utils.data import dataloader
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--split", type=str)


args = parser.parse_args()

split = args.split


random.seed(42)

dataset = load_dataset("cnn_dailymail", '3.0.0')
rouge = evaluate.load("rouge")
cnn_dataset = dataset[split]
print(len(cnn_dataset))
if split == "train":
    cnn_dataset = random.sample(list(cnn_dataset),4000)
else:
    cnn_dataset = random.sample(list(cnn_dataset),500)
    
tokenizer = AutoTokenizer.from_pretrained("bloomberg/KeyBART")
print(len(cnn_dataset))
model = AutoModelForSeq2SeqLM.from_pretrained("bloomberg/KeyBART")
model.eval()
device = "cuda:0"
model.to(device)
batch_size = 16


collector = []

cur_dataloader = dataloader.DataLoader(cnn_dataset, batch_size=batch_size)
key_words_collector = []
for batch in tqdm(cur_dataloader):
    input_ids = tokenizer.batch_encode_plus(batch['article'], padding="max_length", return_tensors="pt", truncation=True)['input_ids'].to(device)
    with torch.no_grad():
        output_ids = model.generate(
        input_ids=input_ids,
        max_length=40,  
        num_beams=10
    ).cpu().detach()
    decoded_seqs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    key_words = [seq.strip(";").split(";") for seq in decoded_seqs] 
    key_words_collector += key_words

for d_idx, data in enumerate(tqdm(cnn_dataset)):
    # load document and summary sentence
    doc = data['article']
    summary_sentence = data['highlights']
    
    
    # print(key_words)
        
    # find key sentence with rougeL score
    doc_sentences = split_into_sentences(doc)
    rouge_scores = rouge.compute(predictions=doc_sentences, references=[summary_sentence for _ in range(len(doc_sentences))], use_aggregator=False)["rougeL"]
    key_sentence_idx = np.argmax(rouge_scores)
    key_sentence = doc_sentences[key_sentence_idx]
    # print(key_sentence)
    
    collector.append({
        "document" : doc,
        "summary": summary_sentence,
        "key_words": key_words_collector[d_idx],
        "key_sentence": key_sentence
    })
    

with open(f"data/{split}.json", "w") as f:
    json.dump(collector, f, indent=4)