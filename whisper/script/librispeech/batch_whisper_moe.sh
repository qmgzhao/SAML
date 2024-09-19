#!/usr/bin/env bash
set -e
set -u
set -o pipefail


# device
device="cuda"
export CUDA_VISIBLE_DEVICES=0

spk_id_list="115 210 2368 3389 3615 479 492 525 6553 829"


# lora
lora_r=1
lora_scale=4
lora_alpha=$((lora_r*lora_scale))
grad_mode=3

batch_size=8
accumgrad=1

# scheduler
lr=0.001
lr2=0.0003
max_epoch=10
interleave_interval=1

# softmax-temperature
temperature=1

scheduler="steplr"  # fixlr steplr warmuplr
warmup_pct=0.0
decay_pct=0.0
if [ "$scheduler" = "warmuplr" ]; then
    warmup_pct=0.3
    decay_pct=0.7
    scheduler_conf="warmup-pct${warmup_pct}_decay-pct${decay_pct}"
elif [ "$scheduler" = "steplr" ]; then
    gamma=0.9
    scheduler_conf="step-size1_gamma${gamma}"
elif [ "$scheduler" = "fixlr" ]; then
    scheduler_conf="none-conf"
fi

# decode
beamsize=20

is_pretrain=0
loadfrom="/home/zhaoqiuming_p/project/whisper/models/nf4.linear-conv-embed.base.en.cpu.pth"

expdir="exp/test1"

mkdir -p ${expdir}

# log
echo "============================================================" >> ${expdir}/log.txt
echo $(date) >> ${expdir}/log.txt


for spk_id in $spk_id_list; do
    mkdir -p ${expdir}/spk${spk_id}

    echo "----------spk_id: ${spk_id}----------" >> "${expdir}/log.txt"

    # train
    python script/librispeech/train_whisper_moe.py \
        --train_json "/home/zhaoqiuming_p/project/whisper/data/train_clean_360_10spk/${spk_id}/${spk_id}.train.json" \
        --dev_json "/home/zhaoqiuming_p/project/whisper/data/train_clean_360_10spk/${spk_id}/${spk_id}.dev.json" \
        --lr ${lr} \
        --nepochs ${max_epoch} \
        --batch_size ${batch_size} \
        --log_interval 1 \
        --warmup_pct ${warmup_pct} \
        --decay_pct ${decay_pct} \
        --expdir "${expdir}/spk${spk_id}" \
        --logfile "${expdir}/spk${spk_id}/log_spk${spk_id}.txt" \
        --accumgrad ${accumgrad} \
        --device "${device}" \
        --loadfrom "${loadfrom}" \
        --lora_r ${lora_r} \
        --lora_alpha ${lora_alpha} \
        --scheduler ${scheduler} \
        --is_pretrain ${is_pretrain} \
        --grad_mode ${grad_mode} \
        --temperature ${temperature} \
        --gamma ${gamma} \
        --lr2 ${lr2} \
        --interleave_interval ${interleave_interval} \

    # decode
    decodedir="decode_spk${spk_id}"
    mkdir -p $expdir/$decodedir
    python script/librispeech/decode_whisper.py \
        --test_json "/home/zhaoqiuming_p/project/whisper/data/train_clean_360_10spk/${spk_id}/${spk_id}.test.json" \
        --beamsize ${beamsize} \
        --expdir "$expdir/$decodedir" \
        --logfile "${expdir}/${decodedir}/log.txt" \
        --save_nbest \
        --device "${device}" \
        --loadfrom "${expdir}/spk${spk_id}/model.acc.best" \

    # score
    sclite \
        -r $expdir/$decodedir/ref.wrd.trn trn \
        -h $expdir/$decodedir/hyp.wrd.trn trn \
        -i rm -o all stdout > $expdir/$decodedir/results.txt
done