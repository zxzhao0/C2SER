datasets=("m3ed" "iemocap")  # Add dataset names to this array e.g., iempcap

for dataset in "${datasets[@]}"; do
    echo "Processing dataset: $dataset"
    python3 speech_feature_extraction.py \
        --model_path C2SER/Emotion2Vec-S/ckpt/checkpoint.pt \
        --model_dir C2SER/Emotion2Vec-S/examples/data2vec/ \
        --dump_dir C2SER/Emotion2Vec-S/fea_${dataset} \
        --device cuda \
        --data C2SER/Emotion2Vec-S/${dataset}.scp
done