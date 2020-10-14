#!/bin/bash

# Han: CHANGE ME!!!
DATA_DIR=fairseq/data-bin/ted_belrus_sepspm8000/M2O/
MODEL_DIR=fairseq/checkpoints/ted_belrus_sepspm8000/M2O/

rm -rf "$MODEL_DIR" # delete existing model
mkdir -p $MODEL_DIR

# Han: MAKE SURE TO CHANGE ALL THE LANG NAMES BELOW!!!

# train the model
CUDA_VISIBLE_DEVICE=0 fairseq-train \
	$DATA_DIR \
	--arch transformer_iwslt_de_en \
	--task translation_multi_simple_epoch \
	--lang-pairs bel-eng,rus-eng \
	--max-epoch 40 \
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
CUDA_VISIBLE_DEVICE=0 fairseq-generate $DATA_DIR \
          --gen-subset test \
	  --task translation_multi_simple_epoch \
	  --lang-pairs bel-eng,rus-eng \
          --source-lang bel --target-lang eng \
          --path $MODEL_DIR/checkpoint_best.pt \
          --batch-size 32 \
	  --tokenizer moses \
          --remove-bpe sentencepiece \
	  --scoring sacrebleu \
          --beam 5   > "$MODEL_DIR"/test_b5.log


CUDA_VISIBLE_DEVICE=0 fairseq-generate $DATA_DIR \
          --gen-subset valid \
	  --task translation_multi_simple_epoch \
	  --lang-pairs bel-eng,rus-eng \
          --source-lang bel --target-lang eng \
          --path $MODEL_DIR/checkpoint_best.pt \
          --batch-size 32 \
	  --tokenizer moses \
          --remove-bpe sentencepiece \
	  --scoring sacrebleu \
          --beam 5   > "$MODEL_DIR"/valid_b5.log


