CUDA_VISIBLE_DEVICES=$1 python src/generate_dataset.py --split=validation 
CUDA_VISIBLE_DEVICES=$1 python src/generate_dataset.py --split=test 
