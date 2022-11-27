import sys
sys.path.append("data")
sys.path.append("utils")
from utils import *
from dataset import SummaryDataset
from transformers import AutoTokenizer, AutoModel
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import load_metric
from nltk import word_tokenize
import numpy as np
import torch
import wandb
import os
import argparse

parser = get_parser()
args = parser.parse_args()

finetune_model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"
tokenizer = AutoTokenizer.from_pretrained(args.model_name)
finetune_model.gradient_checkpointing_enable()
finetune_model = finetune_model.to(device)
columns = ["Input", "Label", "Pred"]


metric = load_metric("rouge")


def compute_metrics(eval_pred):
    predictions, labels, input_text = eval_pred
    # history_string = tokenizer.batch_decode(history, skip_special_tokens=True)
    decoded_preds = tokenizer.batch_decode(
        predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_inputs = tokenizer.batch_decode(
        input_text, skip_special_tokens=True)

    pred_string = decoded_preds
    label_string = decoded_labels
    # Rouge expects a newline after each sentence

    result = metric.compute(predictions=decoded_preds,
                            references=decoded_labels)
    # Extract a few results

    result = {k: v.mid.fmeasure for k, v in result.items()}
    # Add mean generated length
    prediction_lens = [np.count_nonzero(
        pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    wandb.log({"edit_pred": wandb.Table(
        data=list(zip(decoded_inputs, pred_string, label_string)),
        columns=["Text", "Predicted Label", "True Label"]
    )})
    return {k: round(v, 8) for k, v in result.items()}


output_dir = os.path.join(args.checkpoint_dir, args.task_type)
finetune_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    do_train=True,
    do_eval=True,
    do_predict=True,
    evaluation_strategy='steps',
    logging_strategy="steps",
    save_strategy="steps",
    eval_steps=50,
    logging_steps=10,
    per_device_train_batch_size=args.train_batch_size,
    per_device_eval_batch_size=args.val_batch_size,
    learning_rate=args.init_lr,
    weight_decay=args.weight_decay,
    num_train_epochs=args.epoch,
    max_grad_norm=0.1,
    # label_smoothing_factor=0.1,
    gradient_accumulation_steps=2,
    gradient_checkpointing=True,
    # max_steps= ,
    lr_scheduler_type='polynomial',
    # warmup_ratio= ,
    warmup_steps=args.warm_up,
    save_total_limit=1,
    fp16=True,
    seed=516,
    logging_first_step=True,
    load_best_model_at_end=True,
    predict_with_generate=True,
    prediction_loss_only=False,
    generation_max_length=100,
    generation_num_beams=5,
    metric_for_best_model='loss',
    greater_is_better=True,
    report_to='wandb',
    auto_find_batch_size=True,
    include_inputs_for_metrics=True,
)
train_dataset = SummaryDataset("train", args.task_type, tokenizer,
                            "/home/ai/hj/Edit/data", args.encoder_max_len, args.decoder_max_len)
eval_dataset = SummaryDataset("validation", args.task_type, tokenizer,
                           "/home/ai/hj/Edit/data", args.encoder_max_len, args.decoder_max_len)
test_dataset = SummaryDataset("test", args.task_type,  tokenizer,
                           "/home/ai/hj/Edit/data", args.encoder_max_len, args.decoder_max_len)

finetune_trainer = Seq2SeqTrainer(
    model=finetune_model,
    args=finetune_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics


    # preprocess_logits_for_metrics=preprocess_logits_for_metrics
)
if __name__ == "__main__":
    wandb.init(project="Edit", entity="tutoring-convei")
    wandb.run.name = f"{args.task_type}"
    finetune_trainer.train()

    # Save final weights
    finetune_trainer.save_model(os.path.join(args.checkpoint_dir, args.task_type))
