#!/bin/bash

SPACE_DIR=/mnt/dolphinfs/hdd_pool/docker/user/wangjianing16

echo "SPACE_DIR=$SPACE_DIR"

MAIN_DIR=$SPACE_DIR/project/o1_reasoning/o1-plaza
cd $MAIN_DIR
source activate $SPACE_DIR/conda_env/wjn1996

echo $(pwd)
echo $PYTHONPATH
echo $PATH
which python3.10
which python3



DATA_NAME=prm800k # [choose the data name here]
DATA_KIND=train # [choose the data type (train, dev, test)]


PROMPT_KIND=zeroshot_step_cot # [prompt kind, do not change it]

MODEL_NAME=qwen # [the model family]
MODEL_NAME_OR_PATH=$SPACE_DIR/pre-trained-lm/Qwen2.5-14B-Instruct # [the model path]
MODEL_VERSION=Qwen2.5-14B-Instruct # [the model name]


BOOTSTRAP_METHOD=tree_prejudge # [the bootstrapping method, do not change it]
DUPLICATE_N=1 # [sampling number per step]
WIDTH_PER_STEP=4,1,1,1,1,1 # [sampling number at each layer in tree]
DEPTH=6 # max searching step in tree
SAVE_PATH=$MAIN_DIR/data/math/sampled/$BOOTSTRAP_METHOD/$MODEL_VERSION # [saving path]


# export CUDA_VISIBLE_DEVICES="0,1"

python3.10 examples/bootstrapping/run.py \
--data_name=$DATA_NAME \
--data_kind=$DATA_KIND \
--save_path=$SAVE_PATH \
--prompt_kind=$PROMPT_KIND \
--model_name_or_path=$MODEL_NAME_OR_PATH \
--model_version=$MODEL_VERSION \
--model_name=$MODEL_NAME \
--bootstrap_method=$BOOTSTRAP_METHOD \
--duplicate_n=$DUPLICATE_N \
--width_per_step=$WIDTH_PER_STEP \
--depth=$DEPTH \
--save_batch_size=1