# Emotion2vec-S

## Introduction

This repository contains the implementation of Emotion2Vec-S, a self-supervised learning (SSL) model for speech emotion recognition, as presented in our paper "Steering Language Model to Stable Speech Emotion Recognition via Contextual Perception and Chain of Thought". 

## Requirements and Installation

This project follows the fairseq installation process.

### Requirements

- PyTorch version >= 1.10.0
- Python version >= 3.8

### Installation

To install fairseq and develop locally:

```bash
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install --editable ./
```

## 1. Feature Extraction

You can download the pre-trained [Emotion2vec-S model](https://drive.google.com/drive/folders/1LWWi6bahzn7fJP4fCgPleOyQ30sD_BWO?usp=drive_link) and put it in the "C2SER/Emotion2Vec-S/ckpt" folder.

Meanwhile. we have provided the pretrained checkpoints in the huggingface model hub. You can also download ckpt file from here[xxxx].

Use the `speech_feature_extraction.py` script (refer to the official EmoBox code) to extract features from audio files. The script supports parallel processing and provides the following parameters:

- `--model_path`: Path to the checkpoint file
- `--model_dir`: Path to the model
- `--dump_dir`: Directory to save extracted features
- `--device`: Device to run the model on (e.g., 'cuda:0')
- `--data`: Path to the dataset JSON file

```bash
export PYTHONPATH=/path/to/your/fairseq:$PYTHONPATH
cd examples/sb
datasets=("m3ed")  # Add dataset names to this array e.g., iempcap

for dataset in "${datasets[@]}"; do
    echo "Processing dataset: $dataset"
    python3 speech_feature_extraction.py \
        --model_path C2SER/Emotion2Vec-S/ckpt/ \
        --model_dir C2SER/Emotion2Vec-S/examples/data2vec/ \
        --dump_dir C2SER/Emotion2Vec-S/dump_${dataset}-S \
        --device cuda \
        --data C2SER/Emotion2Vec-S/${dataset}/${dataset}.json 
done

echo "All datasets processed successfully."
```

### 2. Training and testing on EmoBox using extracted features

If you want to test our model on other datasets using [EmoBox](https://github.com/emo-box/EmoBox/tree/main). There is also an example provided below, which you can modify to suit your needs:

Use k-fold cross-validation with learning rates (1e-3, 1e-4) and hidden sizes (128, 256):

```bash
cd examples/sb
data=/path/to/your/data_files
lrs=(1e-3 1e-4)               # Learning rate list
hidden_sizes=(128 256)        # Hidden size list
gpus=(0 1 2 3)                # GPU list
task_id=0
declare -A dataset_folds=(
    ["mesd"]=1
)
declare -A dataset_classes=(
    ["mesd"]=6
)
datasets=("mesd")

for dataset in "${datasets[@]}"; do
    folds=${dataset_folds[$dataset]}
    n_classes=${dataset_classes[$dataset]}

    for lr in "${lrs[@]}"; do
        for hidden_size in "${hidden_sizes[@]}"; do
            gpu=${gpus[$task_id % ${#gpus[@]}]}
            export CUDA_VISIBLE_DEVICES=$gpu
            task_number=$((task_id + 1))
            for fold in $(seq 1 $folds); do
                echo "Training fold $fold with lr=$lr, hidden_size=$hidden_size on GPU $gpu, task_number=$task_number, dataset=$dataset..."
                python3 train.py \
                    hparams/data2vec2-large_freeze.yaml \
                    --output_folder /path/to/your/${dataset}-S/fold${fold}_lr${lr}_hidden${hidden_size} \
                    --seed 1234 \
                    --batch_size 32 \
                    --lr $lr \
                    --train_annotation ${data}/${dataset}/fold_${fold}/${dataset}_train_fold_${fold}.json \
                    --test_annotation ${data}/${dataset}/fold_${fold}/${dataset}_test_fold_${fold}.json \
                    --number_of_epochs 100 \
                    --feat_dir /path/to/your/dump_${dataset}-S \
                    --label_map ${data}/${dataset}/label_map.json \
                    --device cuda \
                    --out_n_neurons ${n_classes} \
                    --hidden_size $hidden_size &
            done
            task_id=$((task_id + 1))
        done
    done
done

wait
echo "All training tasks completed."