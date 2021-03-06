#!/bin/bash

PRETRAIN=models/mbart.cc25
langs=ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN
DATA_DIR=fairseq/data-bin/ted_bel_spm8000/bel_eng/
MODEL_DIR=fairseq/checkpoints/ted_bel_spm8000_mbart/bel_eng/
mkdir -p $MODEL_DIR

# train the model
CUDA_VISIBLE_DEVICES='3' fairseq-train \
	$DATA_DIR \
	--encoder-normalize-before --decoder-normalize-before \
	--arch mbart_large --layernorm-embedding \
	--task translation_from_pretrained_bart \
	--criterion label_smoothed_cross_entropy --label-smoothing 0.2 \
	--optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
	--lr-scheduler polynomial_decay --lr 3e-05 --min-lr -1 --warmup-updates 2500 --total-num-update 40000 \
	--dropout 0.3 --attention-dropout 0.1 --weight-decay 0.0 \
	--max-tokens 1024 --update-freq 2 \
	--save-interval 1 --save-interval-updates 5000 --keep-interval-updates 10 --no-epoch-checkpoints \
	--seed 2 --log-format simple \
	--restore-file $PRETRAIN \
	--reset-optimizer --reset-meters --reset-dataloader --reset-lr-scheduler \
	--langs $langs \
	--ddp-backend no_c10d \
  	--save-dir $MODEL_DIR \
	--log-interval 20 >> $MODEL_DIR/train.log 2>&1

#--source-lang en_XX --target-lang ro_RO \
# translate the valid and test set
CUDA_VISIBLE_DEVICES='3' fairseq-generate $DATA_DIR \
          --gen-subset test \
          --path $MODEL_DIR/checkpoint_best.pt \
          --task translation_from_pretrained_bart \
          --batch-size 32 \
	  --tokenizer moses \
          --remove-bpe sentencepiece \
	  --scoring sacrebleu \
	  --langs $langs \
          --beam 5   > "$MODEL_DIR"/test_b5.log


CUDA_VISIBLE_DEVICES='3' fairseq-generate $DATA_DIR \
          --gen-subset valid \
          --path $MODEL_DIR/checkpoint_best.pt \
          --task translation_from_pretrained_bart \
          --batch-size 32 \
	  --tokenizer moses \
          --remove-bpe sentencepiece \
	  --scoring sacrebleu \
	  --langs $langs \
          --beam 5   > "$MODEL_DIR"/valid_b5.log


