CUDA_VISIBLE_DEVICES=$1 python src/train.py --epoch=4 --best_finetune_weight_path=/data/intern/summarization_with_keyword/Summurization-via-Multi-task-Learning/checkpoint/NAIVE/best --task_type=NAIVE
CUDA_VISIBLE_DEVICES=$1 python src/train.py --epoch=10  --init_lr=1e-3 --best_finetune_weight_path=/data/intern/summarization_with_keyword/Summurization-via-Multi-task-Learning/checkpoint/KWKS/best --task_type=KWKS