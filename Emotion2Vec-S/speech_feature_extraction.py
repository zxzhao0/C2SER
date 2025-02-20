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

# We check if transformers is installed.
try:
    import transformers
    from transformers import AutoModel
    from transformers import Wav2Vec2Model, HubertModel, WavLMModel
    from transformers import Wav2Vec2Config, HubertConfig, WavLMConfig
    from transformers import AutoFeatureExtractor, AutoProcessor
    from transformers import Wav2Vec2ForPreTraining
    from transformers import Data2VecAudioModel, Data2VecAudioConfig
    from transformers import WhisperFeatureExtractor, WhisperForAudioClassification, WhisperConfig
    from transformers.models.wav2vec2.modeling_wav2vec2 import (
        _compute_mask_indices,
    )

except ImportError:
    MSG = "Please install transformers from HuggingFace to use wav2vec2 / Hubert\n"
    MSG += "E.G. run: pip install transformers"
    raise ImportError(MSG)


def extract_fairseq_feature(wav_path, channel, model, device, start_time = None, end_time = None):
    try:
        if start_time is not None and end_time is not None:
            sample_rate = torchaudio.info(wav_path).sample_rate
            frame_offset = int(start_time * sample_rate)
            num_frames = int(end_time * sample_rate) - frame_offset
            wav, sr = torchaudio.load(wav_path, frame_offset=frame_offset, num_frames=num_frames)
        else:
            wav, sr = torchaudio.load(wav_path)
        
        channel = channel - 1
        wav = wav[channel, :]
        
        if sr != SAMPLING_RATE:
            wav = torchaudio.functional.resample(wav, sr, SAMPLING_RATE)
        
        wav = wav.view(1, -1)
        wav = wav.to(device)
        out = model.extract_features(wav)
        return out
    except Exception as e:
        print(f"Error processing audio file {wav_path}: {e}")
        return None


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path', type = str, required = True)
    parser.add_argument('--model_dir', type = str, required = True)
    parser.add_argument('--dump_dir', type = str, required = True)
    parser.add_argument('--device', type = str, default = 'cuda')
    parser.add_argument('--data', type =str, required = True)
    
    args = parser.parse_args()
    print(args)

    # load metadata
    f = open(args.data)
    data = json.load(f)
    f.close()
    
    seg_ids = data.keys()
    print(f'load in {len(seg_ids)} segments')
    # load models
    my_model_path = UserDirModule(args.model_dir)
    fairseq.utils.import_user_module(my_model_path)
    model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([args.model_path])
    model = model[0]
    model.to(args.device)

    feat_func = extract_fairseq_feature
    
    # load speech ssl models
    for seg_id in tqdm(seg_ids):
        sample = data[seg_id]
        wav_path = sample['wav']
        channel = sample['channel']
        dur = float(sample['length'])
        if dur > 30. :
            print(f"SKIP {wav_path} because its duration is {dur}, which is too long!")
            continue

        if 'start_time' in sample and 'end_time' in sample:
            start_time = sample['start_time']
            end_time = sample['end_time']
        else:
            start_time = None
            end_time = None    
        assert os.path.exists(wav_path), f'{wav_path} does not exists on your disk'
        try:
            torchaudio.load(wav_path)
        except:
            print(f'ERROR!! wav file {wav_path} can not be loaded!')
            continue   
        feat = feat_func(wav_path, channel, model, args.device, start_time, end_time)
        if feat is not None:
            feat = feat['x'].cpu().detach().numpy()
            feat = feat[0]
            save_path = os.path.join(args.dump_dir, seg_id + '.npy')
            print(f'seg_id:{seg_id}\tfeat_shape:{feat.shape}\tsave_path:{save_path}')
            os.makedirs(os.path.dirname(save_path), exist_ok = True)
            np.save(save_path, feat)
        else:
            print("# 跳过有问题的音频文件")
