# export CUDA_VISIBLE_DEVICES="4,5,6,7"
export CUDA_VISIBLE_DEVICES="0,1,2,3"
python ./train.py --train-type train --dataset tsd --size large
