#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

model_name_or_path="bert-base-cased"
data_dir="data/DocRED"

num_train_ratio=1       # 0.0327(3%), 0.1(10%), 1(100%)
train_batch_size=4
gradient_accumulation_steps=1
dist_fn="cosine"        # inner, cosine
ent_pooler="logsumexp"  # max, sum, avg, logsumexp
rel_pooler="cls"        # pooler, cls
lr=3e-5
clf_lr=1e-4
temperature=0.1
warmup_ratio=0.06
seed=42
share_params=1          # 0(false) or 1(true)
log_to_file=0           # "
long_seq=0              # "

batch_size=$((train_batch_size * gradient_accumulation_steps))
IFS='/' read -ra x <<< $data_dir && dataset_name=${x[1]}            # data_dir.split("/")[1]
IFS='-' read -ra x <<< $model_name_or_path && model_type=${x[0]}    # model_name_or_path.split("-")[0]
if [ $share_params ==  1 ] ; then enc="share" ; else enc="sep" ; fi
if [ $long_seq ==  1 ] ; then long="_long" ; else long="" ; fi

exp="BS${batch_size}_LR${lr}_W${warmup_ratio}_T${temperature}_S${seed}${long}"
train_output_dir=".checkpoints/${dataset_name}/${model_name_or_path}/${enc}/${dist_fn}/${ent_pooler}/${rel_pooler}/N${num_train_ratio}/${exp}"

# HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python evaluate.py \
    --model_name_or_path ${model_name_or_path} \
    --model_type ${model_type} \
    --data_dir ${data_dir} \
    --dataset_name ${dataset_name} \
    --temperature ${temperature} \
    --num_train_ratio ${num_train_ratio} \
    --train_output_dir ${train_output_dir} \
    --dist_fn ${dist_fn} \
    --ent_pooler ${ent_pooler} \
    --rel_pooler ${rel_pooler} \
    --share_params ${share_params} \
    --log_to_file ${log_to_file} \
    --long_seq ${long_seq}
