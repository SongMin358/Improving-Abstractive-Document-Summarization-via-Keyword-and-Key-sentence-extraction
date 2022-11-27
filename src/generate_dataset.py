from transformers import AutoModel, AutoTokenizer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset
import evaluate 
import sys
import numpy as np
sys.path.append("utils/")
from util import split_into_sentences
import torch
from tqdm import tqdm
import json
from torch.utils.data import dataloader
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--split", type=str)
parser.add_argument("--train_split_idx", type=int)
parser.add_argument("--train_split_num", type=int)

args = parser.parse_args()

split = args.split
train_split_idx = args.train_split_idx
train_split_num = args.train_split_num



dataset = load_dataset("cnn_dailymail", '3.0.0')
rouge = evaluate.load("rouge")
cnn_dataset = dataset[split]
print(len(cnn_dataset))
if split == "train":
    len_train = len(cnn_dataset)
    split_len = int(len_train/train_split_num)
    print(split_len)
    if train_split_idx == train_split_num-1:
        cnn_dataset = list(cnn_dataset)[train_split_idx*split_len:]
    else:
        cnn_dataset = list(cnn_dataset)[train_split_idx*split_len: (train_split_idx+1)*split_len]
tokenizer = AutoTokenizer.from_pretrained("bloomberg/KeyBART")
print(len(cnn_dataset))
model = AutoModelForSeq2SeqLM.from_pretrained("bloomberg/KeyBART")
model.eval()
device = "cuda:0"
model.to(device)
batch_size = 32


collector = []

cur_dataloader = dataloader.DataLoader(cnn_dataset, batch_size=batch_size)
key_words_collector = []
for batch in tqdm(cur_dataloader):
    input_ids = tokenizer.batch_encode_plus(batch['article'], padding="max_length", return_tensors="pt", truncation=True)['input_ids'].to(device)
    with torch.no_grad():
        output_ids = model.generate(
        input_ids=input_ids,
        do_sample=True,  # 샘플링 전략 사용
        max_length=30,  # 최대 디코딩 길이는 30
        top_k=50,  # 확률 순위가 50위 밖인 토큰은 샘플링에서 제외
        top_p=0.95  # 누적 확률이 95%인 후보집합에서만 생성
    ).cpu().detach()
    decoded_seqs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    key_words = [seq.strip(";").split(";") for seq in decoded_seqs] 
    key_words_collector += key_words

for d_idx, data in tqdm(enumerate(cnn_dataset)):
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
    
if split == "train":
    with open(f"data/{split}_{train_split_idx}.json", "w") as f:
        json.dump(collector, f, indent=4)
else:
     with open(f"data/{split}.json", "w") as f:
        json.dump(collector, f, indent=4)