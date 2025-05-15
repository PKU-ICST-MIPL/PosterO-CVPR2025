GPU=$1
DATASET=$2
if [ "$DATASET" = "pku" ]; then
    EPOCH=101
elif [ "$DATASET" = "cgl" ]; then
    EPOCH=36
else
    echo "Invalid dataset name: $DATASET"
    return -1
fi
MODEL_DIR=$3
EXP_NAME=$4

EPOCH=`expr $EPOCH - 1`

echo "Inference $DATASET on CUDA: $GPU."

export CUDA_VISIBLE_DEVICES=$GPU
python main.py --dataset_name $DATASET \
            --structure "hierarchical" \
            --injection "top" \
            --design_intent_bbox_dir "design_intent_detect/${DATASET}_128_1e-06_none/result/design_intent_${DATASET}_epoch${EPOCH}" \
            --annotation_dir "$DATASET_ROOT/$DATASET/annotation/" \
            --model_dir $MODEL_DIR \
            --N 10 \
            --num_return 10 \
            --pool_strategy "metric_filter" \
            --rank_strategy "rank_by_feature" \
            --metric_path "sample_select/metric_train_${DATASET}.pt" \
            --filter_dict "sample_select/filter_metric_${DATASET}.json" \
            --feature_dir "design_intent_detect/${DATASET}_128_1e-06_none/result/design_intent_${DATASET}_epoch${EPOCH}/${DATASET}_features" \
            --label_rback \
            --sample_size 10 \
            --batch_size 1 \
            --exp_name $EXP_NAME