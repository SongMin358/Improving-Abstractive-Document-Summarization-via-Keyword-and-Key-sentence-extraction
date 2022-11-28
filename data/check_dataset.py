import sys 
sys.path.append("data")
sys.path.append("utils")
from util import *
from dataset import SummaryDataset
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")

for task_type in TASK_LIST:
    test_dataset = SummaryDataset("test", task_type, tokenizer, "data", dec_max_len=200)
    for data in test_dataset:
        cur_data = data
        print(f"#### {task_type} ####")
        print(tokenizer.decode(cur_data['labels'], skip_special_tokens=True))
        print("="*100)
        break
        
print("Your dataset is good to go!")
