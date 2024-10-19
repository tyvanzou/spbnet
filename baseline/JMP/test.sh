# export CUDA_VISIBLE_DEVICES="4,5,6,7"
export CUDA_VISIBLE_DEVICES="0"
python ./train.py --train-type test --dataset tsd --size large
