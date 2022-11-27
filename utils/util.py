import spacy
import argparse

nlp = spacy.load('en_core_web_sm')

def split_into_sentences(text):
    sentences = [str(i) for i in nlp(text).sents]
    return sentences


NAIVE = "NAIVE"
KW = "KW"
KS = "KS"
KWKS = "KWKS"
TASK_LIST = ["NAIVE", "KW", "KS", "KWKS"]


def get_parser():
    parser = argparse.ArgumentParser()
    # Training hyperparameters
    parser.add_argument('--task_type',
                    default=KW,
                    const=KW,
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
    parser.add_argument('--best_finetune_weight_path',type=str, default='final_bart', required=True)
    # Dataset hyperparameters
    parser.add_argument('--test_output_file_name',type=str, default="test_output.txt")
    
    return parser
