export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
bash "$(dirname "$0")/sample_cat.sh" "$@"
