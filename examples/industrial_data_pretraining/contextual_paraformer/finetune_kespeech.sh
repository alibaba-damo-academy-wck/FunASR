# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

# method1, finetune from model hub

# which gpu to train or finetune
export CUDA_VISIBLE_DEVICES="0"
gpu_num=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')

# model_name from model_hub, or model_dir in local path

## option 1, download model automatically
model_name_or_model_dir="iic/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404"
model_revision=v2.0.5

# option 2, download model by git
#local_path_root=${workspace}/modelscope_models
#mkdir -p ${local_path_root}/${model_name_or_model_dir}
#git clone https://www.modelscope.cn/${model_name_or_model_dir}.git ${local_path_root}/${model_name_or_model_dir}
#model_name_or_model_dir=${local_path_root}/${model_name_or_model_dir}


# data dir, which contains: train.json, val.json
data_dir="../../../data/kespeech"

train_data="${data_dir}/train.jsonl"
val_data="${data_dir}/val.jsonl"

stage=1
stop_stage=1

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  # generate train.jsonl and val.jsonl from wav.scp and text.txt
  log "stage 0: generate train.jsonl and val.jsonl from wav.scp and text.txt"
  scp2jsonl \
  ++scp_file_list='["../../../data/kespeech/train_wav.scp", "../../../data/kespeech/train_text.txt"]' \
  ++data_type_list='["source", "target"]' \
  ++jsonl_file_out="${train_data}"

  scp2jsonl \
  ++scp_file_list='["../../../data/kespeech/val_wav.scp", "../../../data/kespeech/val_text.txt"]' \
  ++data_type_list='["source", "target"]' \
  ++jsonl_file_out="${val_data}"
fi

# exp output dir
output_dir="./outputs"
log_file="${output_dir}/log.txt"


mkdir -p ${output_dir}
echo "log_file: ${log_file}"

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  log "stage 1: finetune"
  batch_size=45000   # default: 20000
  num_workers=4   # default: 4
  max_epoch=50     # default: 50
  keep_nbest_models=20   # default: 20

  torchrun \
  --nnodes 1 \
  --nproc_per_node ${gpu_num} \
  ../../../funasr/bin/train.py \
  ++model="${model_name_or_model_dir}" \
  ++model_revision="${model_revision}" \
  ++train_data_set_list="${train_data}" \
  ++valid_data_set_list="${val_data}" \
  ++dataset_conf.batch_size=${batch_size} \
  ++dataset_conf.batch_type="token" \
  ++dataset_conf.num_workers=${num_workers} \
  ++train_conf.max_epoch=${max_epoch} \
  ++train_conf.log_interval=1 \
  ++train_conf.resume=false \
  ++train_conf.validate_interval=2000 \
  ++train_conf.save_checkpoint_interval=2000 \
  ++train_conf.keep_nbest_models=${keep_nbest_models} \
  ++optim_conf.lr=0.0002 \
  ++output_dir="${output_dir}" &> ${log_file}

#  python -m ipdb ../../../funasr/bin/train.py \
#  ++model="${model_name_or_model_dir}" \
#  ++model_revision="${model_revision}" \
#  ++train_data_set_list="${train_data}" \
#  ++valid_data_set_list="${val_data}" \
#  ++dataset_conf.batch_size=${batch_size} \
#  ++dataset_conf.batch_type="token" \
#  ++dataset_conf.num_workers=4 \
#  ++train_conf.max_epoch=${max_epoch} \
#  ++train_conf.log_interval=1 \
#  ++train_conf.resume=false \
#  ++train_conf.validate_interval=2000 \
#  ++train_conf.save_checkpoint_interval=2000 \
#  ++train_conf.keep_nbest_models=${keep_nbest_models} \
#  ++optim_conf.lr=0.0002 \
#  ++output_dir="${output_dir}"
fi
