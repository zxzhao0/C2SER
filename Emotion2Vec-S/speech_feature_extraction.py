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
        if sr != SAMPLING_RATE:
            wav = torchaudio.functional.resample(wav, sr, SAMPLING_RATE)
        
        wav = wav.to(device)
        out = model.extract_features(wav)
        return out
    except Exception as e:
        print(f"Error processing audio file {wav_path}: {e}")
        return None

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="./Emotion2Vec-S/ckpt/checkpoint.pt", help="Path to the model checkpoint file")
    parser.add_argument('--model_dir', type=str, default="./Emotion2Vec-S/examples/data2vec/", help="Path to the model directory")
    parser.add_argument('--dump_dir', type=str, default="./Emotion2Vec-S/features", help="Directory to save extracted features")
    parser.add_argument('--device', type=str, default='cuda', help="Device to use for computation (e.g., 'cuda' or 'cpu')")
    parser.add_argument('--data', type=str, default="./Emotion2Vec-S/wav.scp", help="Path to the wav.scp file containing audio paths")
    args = parser.parse_args()

    data = {}
    with open(args.data, 'r') as f:
        for line in f:
            seg_id, wav_path = line.strip().split(maxsplit=1)
            data[seg_id] = wav_path
    
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
        try:
            torchaudio.load(wav_path)
        except:
            print(f'ERROR!! wav file {wav_path} can not be loaded!')
            continue   

        # 提取特征
        feat = extract_fairseq_feature(wav_path, model, args.device)

        if feat is not None:
            # 处理特征输出
            feat = feat['x'].cpu().detach().numpy()[0]  # 去掉batch维度
            save_path = os.path.join(args.dump_dir, f"{seg_id}.npy")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.save(save_path, feat)
            print(f"Processed: {seg_id} | Shape: {feat.shape} | Saved to: {save_path}")
        else:
            print(f"Skipped problematic file: {seg_id}")
