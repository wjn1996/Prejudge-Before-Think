#!/bin/bash
MAIN_DIR="/mnt/dolphinfs/hdd_pool/docker/user/wangjianing16/project/o1_reasoning/o1-plaza"
cd $MAIN_DIR

# source activate /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/wangjianing16/conda_env/wjn1996

echo $(pwd)
echo $PYTHONPATH
echo $PATH
which python3.10
which python3

DATA_NAME=prejudge_critique
DATA_KIND=train

MODEL_NAME_OR_PATH=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/wangjianing16/pre-trained-lm/Qwen2.5-72B-Instruct
MODEL_VERSION=Qwen2.5-72B-Instruct
MODEL_NAME=qwen

# MODEL_NAME_OR_PATH=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/wangjianing16/pre-trained-lm/Meta-Llama-3-70B-Instruct
# MODEL_VERSION=Meta-Llama-3-70B-Instruct
# MODEL_NAME=llama3

# MODEL_NAME_OR_PATH=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/wangjianing16/pre-trained-lm/Meta-Llama-3-8B-Instruct
# MODEL_VERSION=Meta-Llama-3-8B-Instruct
# MODEL_NAME=llama3

DUPLICATE_N=4
SAVE_PATH=$MAIN_DIR/data/math/prejudge_examples/$MODEL_VERSION


# export CUDA_VISIBLE_DEVICES="0,1,2,3"

python3 examples/cot_distillating/run.py \
--data_name=$DATA_NAME \
--data_kind=$DATA_KIND \
--save_path=$SAVE_PATH \
--model_name_or_path=$MODEL_NAME_OR_PATH \
--model_version=$MODEL_VERSION \
--model_name=$MODEL_NAME \
--duplicate_n=$DUPLICATE_N \
--save_batch_size=100 \
--start_n=5000 \
--cut_n=5000