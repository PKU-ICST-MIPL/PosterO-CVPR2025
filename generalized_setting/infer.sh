GPU=$1
MODEL_DIR=$2
EXP_NAME=$3

export CUDA_VISIBLE_DEVICES=$GPU
for i in $(seq 0 6)
do
python main.py --dataset_name "ps" \
                  --ps_group $i \
                  --ps_dm_name "predm_zs" \
                  --canvas_size_w 513 \
                  --canvas_size_h 0 \
                  --structure "plain" \
                  --injection "top" \
                  --design_intent_bbox_dir "${DATASET_ROOT}/PStylish7" \
                  --annotation_dir "${DATASET_ROOT}/PStylish7" \
                  --model_dir $MODEL_DIR \
                  --rank_strategy "rank_by_feature" \
                  --N 1 \
                  --num_return 1 \
                  --label_rback \
                  --sample_size 10 \
                  --exp_name $EXP_NAME 
done