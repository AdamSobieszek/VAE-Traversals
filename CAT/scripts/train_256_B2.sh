export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
bash "$(dirname "$0")/train_cat_b.sh"
