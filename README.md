# Steering Language Model to Stable Speech Emotion Recognition via Contextual Perception and Chain of Thought
<!-- <br> <sub> The official implementation of C<sup>2</sup>SER (submit to ACL 2025) </sub> -->

## Abstract
We propose C<sup>2</sup>SER, a novel audio language model (ALM) designed to enhance the stability and accuracy of speech emotion recognition (SER) through **C**ontextual perception and **C**hain of Thought (CoT). C<sup>2</sup>SER integrates the Whisper encoder for semantic perception and Emotion2Vec-S for acoustic perception, where Emotion2Vec-S extends Emotion2Vec with semi-supervised learning to enhance emotional discrimination. Additionally, C<sup>2</sup>SER employs a CoT approach, processing SER in a step-by-step manner while leveraging speech content and speaking styles to improve recognition. To further enhance stability, C<sup>2</sup>SER introduces self-distillation from explicit CoT to implicit CoT, mitigating error accumulation and boosting recognition accuracy. Extensive experiments show that C<sup>2</sup>SER outperforms existing popular ALMs, such as Qwen2-Audio and SECap, delivering more stable and precise emotion recognition.


<p align="center">
  <img src="figs/details of CSER.drawio.jpg" width="550"/>
</p>

## Roadmap 📝

Release code and documents of 
- [x] Emo-Emilia dataset
- [x] Emotion2Vec-S model and feature extraction code
- [x] Release C<sup>2</sup>SER-LLM model and Inference pipeline

Release pretrained checkpoint of 
- [x] Emotion2Vec-S model
- [x] C<sup>2</sup>SER-LLM model
## Emo-Emilia Dataset

To better simulate real-world context, we introduce a new SER test set, **Emo-Emilia**.
Specifically, we apply the automated labeling approach to annotate Emilia, a large-scale multilingual and diverse speech generation resource with over 100,000 hours of speech data that captures a wide range of emotional contexts.
We then manually verify the accuracy of the emotion labels. Each utterance is checked by at least two experts to ensure both accuracy and reliability. The final proposed test set, Emo-Emilia, consists of 1400 test samples, with 100 samples per emotion category across seven types in both Chinese and English (700 samples per language).

Emo-Emilia is a subset of Emilia dataset, to get the complete Emo-Emilia data, please get Emilia data first. The original Emilia dataset can be accessed [here](https://emilia-dataset.github.io/Emilia-Demo-Page/).

Emo-Emilia Dataset files: `./Emo-Emilia/Emo-Emilia-ALL.jsonl`

## Emotion2Vec-S

### Introduction

This repository contains the implementation of Emotion2Vec-S, a self-supervised learning (SSL) model for speech emotion recognition, as presented in our paper "Steering Language Model to Stable Speech Emotion Recognition via Contextual Perception and Chain of Thought". 

### Requirements and Installation

This project follows the fairseq installation process.

#### Requirements

- PyTorch version >= 1.10.0
- Python version >= 3.8

#### Installation

To install fairseq and develop locally:

```bash
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install --editable ./
```

### Feature Extraction

You can download the pre-trained [Emotion2vec-S model](https://drive.google.com/drive/folders/1LWWi6bahzn7fJP4fCgPleOyQ30sD_BWO?usp=drive_link) and put it in the `./Emotion2Vec-S/ckpt` folder. 
Meanwhile，we have provided the pretrained checkpoints on the Hugging Face Model Hub. You can also download ckpt file from [here](https://huggingface.co/ASLP-lab/Emotion2Vec-S). We also provide [here](https://drive.google.com/drive/folders/12AOVJT7I9GSLJnjHa-Elc-UKgog-mZR2) the feature files for the Emo-Emilia dataset extracted using Emotion2vec-S. 

If you want to extract features using Emotion2Vec-S，you will also need to provide a `wav.scp` file and place it in the `./Emotion2Vec-S` directory. Here is an example of the `wav.scp` file:：
```pgsql
audio_name1 /path/to/audio_name1.wav
audio_name2 /path/to/audio_name2.wav
audio_name3 /path/to/audio_name3.wav
```

Next, you can directly run the following code to extract features：
```python
import torch
import os
import sys
import json
import numpy as np
import argparse
from tqdm import tqdm
import torchaudio
import torch.nn.functional as F
import fairseq
from dataclasses import dataclass

SAMPLING_RATE=16000

@dataclass
class UserDirModule:
    user_dir: str

def extract_fairseq_feature(wav_path, model, device):
    try:
        wav, sr = torchaudio.load(wav_path)
        # Convert multi-channel to mono by averaging
        if wav.size(0) > 1:
            wav = torch.mean(wav, dim=0, keepdim=True)
        if sr != SAMPLING_RATE:
            wav = torchaudio.functional.resample(wav, sr, SAMPLING_RATE)
        wav = wav[0, :].view(1, -1)
        wav = wav.to(device)
        out = model.extract_features(wav)
        return out
    except Exception as e:
        print(f"Error processing audio file {wav_path}: {e}")
        return None

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="./Emotion2Vec-S/ckpt/checkpoint.pt")
    parser.add_argument('--model_dir', type=str, default="./Emotion2Vec-S/examples/data2vec/")
    parser.add_argument('--dump_dir', type=str, default="./Emotion2Vec-S/features_frm")
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--data', type=str, default="./Emotion2Vec-S/wav.scp")
    parser.add_argument('--level', type=str, default="frame", help="frame or utterance")
    args = parser.parse_args()

    data = {}
    with open(args.data, 'r') as f:
        for line in f:
            seg_id, wav_path = line.strip().split(maxsplit=1)
            data[seg_id] = wav_path

    os.makedirs(args.dump_dir, exist_ok=True)

    seg_ids = data.keys()
    print(f'Loaded {len(seg_ids)} audio entries')
    # load models
    my_model_path = UserDirModule(args.model_dir)
    fairseq.utils.import_user_module(my_model_path)
    model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([args.model_path])
    model = model[0].to(args.device)
    
    for seg_id in tqdm(seg_ids):

        wav_path = data[seg_id]
        if not os.path.exists(wav_path):
            print(f"WARNING: {wav_path} does not exist")
            continue 
        try:
            torchaudio.load(wav_path)
        except:
            print(f'ERROR: Failed to load {wav_path}')
            continue         

        feat = extract_fairseq_feature(wav_path, model, args.device)

        if feat is not None:
            if args.level == 'frame':
                feat = feat['x'].cpu().detach().numpy()[0]
            elif args.level == 'utterance':
                feat = feat['utt_x'].cpu().detach().numpy()[0] 
            else:
                raise ValueError("Unknown level: {}".format(args.level))            

            save_path = os.path.join(args.dump_dir, f"{seg_id}.npy")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.save(save_path, feat)
            print(f"Processed: {seg_id} | Shape: {feat.shape} | Saved to: {save_path}")
        else:
            print(f"Skipped problematic file: {seg_id}")

```
Alternatively, you can adjust the code according to your needs. The code path is `./Emotion2Vec-S/speech_feature_extraction.py`. You can also use the `./Emotion2Vec-S/extract_feature.sh` script to batch process features for multiple datasets. The script supports parallel processing and offers the following parameters:

- `--model_path`: Path to the checkpoint file
- `--model_dir`: Path to the model
- `--dump_dir`: Directory to save extracted features
- `--device`: Computation device (e.g., 'cuda:0')
- `--data`: Path to the dataset scp file
- `--level`: Feature extraction level (frame-level/utterance-level)

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
```
## C<sup>2</sup>SER-LLM

### Introduction

As presented in the Roadmap，C<sup>2</sup>SER employs a CoT training approach to incentivize reasoning capability. This approach decomposes the SER task into sequential steps: first perceiving speech content and speaking style, followed by emotion inference, with the assistance of prior context. This structured method imitates human thinking and reduces the possibility of hallucinations. To further enhance stability and prevent error propagation, especially in longer thought chains, C<sup>2</sup>SER introduces self-distillation, transferring knowledge from explicit to implicit CoT.

### Installation

To install the project dependencies, use the following command:

```
cd C2SER-llm
pip install -r requirements.txt
```
### Pretrained Model


To run the code, you need to download two files. The first file is [Qwen-7B](https://huggingface.co/Qwen/Qwen2-7B). After downloading, replace the `llm_path` in `./C2SER-llm/config.yaml` with your download path. The second file is the pretrained model **C2SER_llm.pt**. We have provided the pretrained checkpoints on the Hugging Face Model Hub. You can also download ckpt file from [here](https://huggingface.co/ASLP-lab/C2SER-LLM). After downloading, replace the
`checkpoint_path` in `./C2SER-llm/infer_runtime.py` with the path to your downloaded file.

### Inference

We provide three input parameters in `./C2SER-llm/infer_runtime.py`:
- `--input_wav_path`: Path to the test WAV file.
- `--ssl_vector_path`: Path to the utterance-level feature.
- `--input_prompt`: Prompts for stage1 or stage2

After extracting the utterance-level features of the audio file using Emotion2Vec-S, you need to replace `input_wav_path` and `ssl_vector_path` in `./C2SER-llm/infer_runtime.py` with the paths to your test audio file and extracted utterance-level features, respectively. You can also control the output of Stage1 and Stage2 by adjusting `input_prompt`. The prompt information is listed in `./C2SER-llm/prompt_config.yaml`. Then, you can directly perform inference by running the following code.

```
python C2SER-llm/infer_runtime.py
```

### Results

We have provided an example result for the file `./Emotion2Vec-S/test_wav/vo_EQAST002_1_paimon_07.wav`

If you use the Stage 1 prompt: `Please describe the speaking style, content, and the speaker's emotional state of this speech.` ，the output will be:
```
说话者以缓慢的速度、高昂的语调和中等音量的声音说道：“不知道艾德林小姐有没有给我们准备好吃的点心呢。”通过分析语音特征，推测情绪为快乐，透露出一种期待和兴奋的喜悦。
```
If you use the Stage 2 prompt: `Please consider the speaking style, content, and directly provide the speaker's emotion in this speech.` ，the output will be:
```
这条语音的的情感为高兴
```
