import argparse
import json
import time
import openai
from typing import List, Optional, Union
from Edit.EditEval.src.metrics.custom_metrics import *

ORIGINAL_TO_COMMENT = "ORIGINAL_TO_COMMENT"
ORIGINAL_TO_CORRECTION = "ORIGINAL_TO_CORRECTION"
ORIGINALANDCOMMENT_TO_CORRECTION = "ORIGINALANDCOMMENT_TO_CORRECTION"
ORIGINAL_TO_COMMENTANDCORRECTION = "ORIGINAL_TO_COMMENTANDCORRECTION"
TASK_LIST = ["ORIGINAL_TO_COMMENT", "ORIGINAL_TO_CORRECTION", "ORIGINALANDCOMMENT_TO_CORRECTION", "ORIGINAL_TO_COMMENTANDCORRECTION"]


def get_parser():
    parser = argparse.ArgumentParser()
    # Training hyperparameters
    parser.add_argument('--task_type',
                    default=ORIGINAL_TO_CORRECTION,
                    const=ORIGINAL_TO_CORRECTION,
                    nargs='?',
                    choices=TASK_LIST,
                    help='Type of task (default: %(default)s)')
    parser.add_argument("--include_topic_card", type=bool)
    parser.add_argument('--epoch', type=int, default=5)
    parser.add_argument('--train_batch_size', type=int, default=64)
    #parser.add_argument('--display_step',type=int, default=2000)
    parser.add_argument('--val_batch_size',type=int, default=4)
    parser.add_argument('--test_batch_size',type=int,default=1)
    # Model hyperparameters
    parser.add_argument('--model_name',type=str, default="facebook/bart-large")
    # Optimizer hyperparameters
    parser.add_argument('--init_lr',type=float, default=3e-5)
    parser.add_argument('--warm_up',type=int, default=30)
    parser.add_argument('--weight_decay',type=float, default=1e-2)
    parser.add_argument('--decay_epoch',type=int, default=0)
    parser.add_argument('--adam_beta1',type=float, default=0.9)
    parser.add_argument('--adam_beta2',type=float, default=0.999)
    parser.add_argument('--adam_eps',type=float, default=1e-12)
    parser.add_argument('--dropout_rate',type=float, default=0.1)
    # Tokenizer hyperparameters
    parser.add_argument('--encoder_max_len', type=int, default=300)
    parser.add_argument('--decoder_max_len', type=int, default=100)
    parser.add_argument('--vocab_size',type=int, default=51201)
    parser.add_argument('--eos_idx',type=int, default=51200)
    parser.add_argument('--tokenizer_name',type=str, default='RobertaTokenizer')
    parser.add_argument('--checkpoint_dir', type=str, default="/data/intern/Edit/checkpoint")
    # Checkpoint directory hyperparameters
    #parser.add_argument('--pretrained_weight_path',type=str, default='pretrained_weights')
    parser.add_argument('--best_finetune_weight_path',type=str, default='final_bart')
    # Dataset hyperparameters
    parser.add_argument('--test_output_file_name',type=str, default="test_output.txt")
    
    return parser


def compute_metrics(
    name: str = None,
    originals: Optional[List[str]] = None,
    predictions: Optional[List[str]] = None,
    targets: Optional[Union[List[List[str]], List[str]]] = None,
    original_tokens: Optional[List[List[str]]] = None,
    prediction_tokens: Optional[List[List[str]]] = None,
    target_tokens: Optional[List[List[List[str]]]] = None,
) -> dict:

    # TODO: add assertion errors
    assert name in ("update_rouge", "em", "em_diff", "sari", "gleu", "bleu", "ibleu"), \
        'Must enter a valid metric name: {"update_rouge", "em", "em_diff", "sari", "gleu", "bleu", "ibleu"}'

    if name == "update_rouge":
        metric = UpdateRougeMetric()
    elif name == "em":
        metric = ExactMatchMetric()
    elif name == "em_diff":
        metric = ExactMatchDiffMetric()
    elif name == "sari":
        metric = EasseSariMetric()
    elif name == "gleu":
        metric = GLEUMetric()
    elif name == "bleu":
        metric = iBLEUMetric(alpha=1.0)  # -> BLEU(candidate, reference)
    elif name == "ibleu":
        # iBLEU = alpha * BLEU(candidate, reference) - (1 - alpha) * BLEU(candidate, source)
        metric = iBLEUMetric()  # default: alpha=0.9

    if targets is not None:
        if isinstance(targets[0], str):
            targets = [[target] for target in targets]
    scores = metric.evaluate(originals=originals, predictions=predictions, targets=targets)
    return scores


class GPT3:
    def __init__(self, args):
        self.args = args

        self.model_name = self.args.model_name
        self.max_tokens = self.args.max_tokens
        self.temperature = float(self.args.temperature)
        self.top_p = self.args.top_p
        self.num_samples = self.args.num_samples
        self.presence_penalty = self.args.presence_penalty
        self.frequency_penalty = self.args.frequency_penalty
        self.save_output = self.args.save_output
        self.cur_idx = -1

    def login_to_openai(self, keys, cur_idx):
        openai.api_key = keys[cur_idx] 

    def set_new_key(self):
        with open("Edit/personal_info.json") as f:
            keys = json.load(f)
        self.cur_idx += 1
        self.cur_idx = self.cur_idx % len(keys)
        self.login_to_openai(keys, self.cur_idx)

    def inference(self, prompt, return_raw=False, stop_seq="\n\n", temperature=None):
        timeout_stack = 0
        if temperature is None:
            temperature = self.temperature
        while True:
            try:
                # print("ours inference")
                output = openai.Completion.create(
                    engine=self.model_name,
                    prompt=prompt,
                    temperature=temperature,
                    top_p=self.top_p,
                    n=self.num_samples,
                    presence_penalty=self.presence_penalty,
                    frequency_penalty=self.frequency_penalty,
                    max_tokens=self.max_tokens,
                    stop=stop_seq,
                    logprobs=1
                )
                break
            except Exception as e:
                # print(e)
                # print(f"time limit, stack: {timeout_stack}")
                timeout_stack += 1
                if timeout_stack >= 3:
                    print("change to another key")
                    self.set_new_key()
                    timeout_stack = 0
                time.sleep(1)

        if return_raw:
            return output
        return output['choices'][0]['text']
