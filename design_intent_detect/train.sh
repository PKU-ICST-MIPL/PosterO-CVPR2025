export CUDA_VISIBLE_DEVICES=6,7,8,9
DATASET=$1
if [ "$DATASET" = "pku" ]; then
    EPOCH=101
elif [ "$DATASET" = "cgl" ]; then
    EPOCH=36
elif [ "$DATASET" = "all" ]; then
    EPOCH=26
fi

echo "Training on $DATASET with $EPOCH epochs"

torchrun --standalone --nnodes=1 --nproc-per-node=4 main.py \
        --dataset_root $DATASET_ROOT --dataset $1 \
        --batch_size 128 --learning_rate 1e-6 --model_dm_act "none" \
        --epoch $EPOCH