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