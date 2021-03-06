#!/bin/bash

SUBSAMPLE_SIZE=100000
UPSAMPLE=8
DATA_DIR=fairseq/data-bin/ted_bel_spm8000/bel_eng_plus_bt_"$SUBSAMPLE_SIZE"/
MODEL_DIR=fairseq/checkpoints/ted_bel_spm8000/bel_eng_plus_bt_"$SUBSAMPLE_SIZE"_upsample_"$UPSAMPLE"/
mkdir -p $MODEL_DIR

# change the cuda_visible_device to the GPU device number you are using
# train the model
CUDA_VISIBLE_DEVICE="0" fairseq-train \
	--fp16 \
	$DATA_DIR \
	--upsample-primary "$UPSAMPLE" \
	--arch transformer_iwslt_de_en \
	--max-epoch 80 \
    --distributed-world-size 1 \
	--share-all-embeddings \
	--no-epoch-checkpoints \
	--dropout 0.3 \
	--optimizer 'adam' --adam-betas '(0.9, 0.98)' --lr-scheduler 'inverse_sqrt' \
	--warmup-init-lr 1e-7 --warmup-updates 4000 --lr 2e-4  \
	--criterion 'label_smoothed_cross_entropy' --label-smoothing 0.1 \
	--max-tokens 4500 \
	--update-freq 2 \
	--seed 2 \
  	--save-dir $MODEL_DIR \
	--log-interval 100 >> $MODEL_DIR/train.log 2>&1

# translate the valid and test set
CUDA_VISIBLE_DEVICE=xx  fairseq-generate $DATA_DIR \
          --gen-subset test \
          --path $MODEL_DIR/checkpoint_best.pt \
          --batch-size 32 \
	  --tokenizer moses \
          --remove-bpe sentencepiece \
	  --scoring sacrebleu \
          --beam 5   > "$MODEL_DIR"/test_b5.log


CUDA_VISIBLE_DEVICE=xx fairseq-generate $DATA_DIR \
          --gen-subset valid \
          --path $MODEL_DIR/checkpoint_best.pt \
          --batch-size 32 \
	  --tokenizer moses \
          --remove-bpe sentencepiece \
	  --scoring sacrebleu \
          --beam 5   > "$MODEL_DIR"/valid_b5.log


