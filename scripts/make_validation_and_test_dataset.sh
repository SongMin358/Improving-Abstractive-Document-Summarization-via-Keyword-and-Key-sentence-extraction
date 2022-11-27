CUDA_VISIBLE_DEVICES=$1 python src/generate_dataset.py --split=validation --train_split_idx=0 --train_split_num=0
CUDA_VISIBLE_DEVICES=$1 python src/generate_dataset.py --split=test --train_split_idx=0 --train_split_num=0
