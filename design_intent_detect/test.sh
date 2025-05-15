export CUDA_VISIBLE_DEVICES=0,1,2,3
SPLIT=("test" "valid" "train")
INFER_CSV=("test" "train" "train")
DATASET=$1
INFER_CKPT=$2

echo "Inferece on $DATASET with $INFER_CKPT"

# maps
for i in {0..1}; do
    torchrun --standalone --nnodes=1 --nproc-per-node=4 main.py \
        --dataset_root $DATASET_ROOT --dataset $DATASET \
        --infer --infer_ckpt $INFER_CKPT \
        --infer_csv "${INFER_CSV[i]}"
done

if [ "$DATASET" = "all" ]; then
    return 0
fi

# features
for i in {0..2}; do
torchrun --standalone --nnodes=1 --nproc-per-node=4 main.py \
        --dataset_root $DATASET_ROOT --dataset $DATASET \
        --extract --extract_split "${SPLIT[i]}" --infer_csv "${INFER_CSV[i]}" \
        ----infer --infer_ckpt $INFER_CKPT
done

# map2box
python map2box.py \
    --dataset_root $DATASET_ROOT \
    --dataset $DATASET \
    --infer_ckpt $INFER_CKPT \
    --kernel_n 37
