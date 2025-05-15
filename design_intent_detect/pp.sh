echo "Rendering binary masks for pku dataset"
python preprocess.py --dataset_root $DATASET_ROOT \
                   --dataset "pku"

echo "Rendering binary masks for cgl dataset"
python preprocess.py --dataset_root $DATASET_ROOT \
                   --dataset "cgl"