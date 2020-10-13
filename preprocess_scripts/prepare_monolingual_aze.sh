#!/bin/bash

if [ ! -d mosesdecoder ]; then
  echo 'Cloning Moses github repository (for tokenization scripts)...'
  git clone https://github.com/moses-smt/mosesdecoder.git
fi

SLAN=eng
TLAN=aze
OLAN=aze
VOCAB_SIZE=8000
SUBSAMPLE_SIZE=100000
RAW_DDIR=data/mono/
PROC_DDIR=data/ted_processed/"$OLAN"_spm"$VOCAB_SIZE"/
BINARIZED_DDIR=fairseq/data-bin/ted_"$OLAN"_spm"$VOCAB_SIZE"/
FAIR_SCRIPTS=fairseq/scripts
SPM_TRAIN=$FAIR_SCRIPTS/spm_train.py
SPM_ENCODE=$FAIR_SCRIPTS/spm_encode.py
TOKENIZER=mosesdecoder/scripts/tokenizer/tokenizer.perl
CHECKPOINT_DIR=fairseq/checkpoints/ted_"$OLAN"_spm"$VOCAB_SIZE"/"$OLAN"_eng/

echo "------------------- 0 ---------------------"
echo "$TLAN"
if [ ! -a "$RAW_DDIR"/"$TLAN"_"$SUBSAMPLE_SIZE" ]; then
  mkdir -p "$RAW_DDIR"/"$TLAN"_"$SUBSAMPLE_SIZE"
  f="$RAW_DDIR"/"$TLAN"/lg."$TLAN"
  f1="$RAW_DDIR"/"$TLAN"_"$SUBSAMPLE_SIZE"/lg.mtok."$TLAN"
  echo $f
  echo $f1
  echo "tokenize $f1..."
  shuf -n $SUBSAMPLE_SIZE $f\
  | perl $TOKENIZER > $f1
fi
echo "------------------- 1 ---------------------"

if [ ! -a "$PROC_DDIR"/"$TLAN"_"$SUBSAMPLE_SIZE"/spm"$VOCAB_SIZE".lg."$TLAN" ]; then
  mkdir -p "$PROC_DDIR"/"$TLAN"_"$SUBSAMPLE_SIZE"
  python "$SPM_ENCODE" \
  --model="$PROC_DDIR"/"$OLAN"_eng/spm"$VOCAB_SIZE".model \
  --output_format=piece \
  --inputs "$RAW_DDIR"/"$TLAN"_"$SUBSAMPLE_SIZE"/lg.mtok."$TLAN"  \
  --outputs "$PROC_DDIR"/"$TLAN"_"$SUBSAMPLE_SIZE"/spm"$VOCAB_SIZE".lg."$TLAN" \
  --min-len 1 --max-len 200 
fi
echo "------------------- 2 ---------------------"

echo "Binarize the data..."
if [ ! -a $BINARIZED_DDIR/"$TLAN"_mono_"$SUBSAMPLE_SIZE"/ ]; then
  fairseq-preprocess \
  --only-source \
  --source-lang "$TLAN" \
  --target-lang "$SLAN" \
  --joined-dictionary \
  --srcdict "$BINARIZED_DDIR"/"$SLAN"_"$TLAN"/dict."$TLAN".txt \
  --testpref "$PROC_DDIR"/"$TLAN"_"$SUBSAMPLE_SIZE"/spm"$VOCAB_SIZE".lg \
  --destdir $BINARIZED_DDIR/"$TLAN"_mono_"$SUBSAMPLE_SIZE"/
  
  cp "$BINARIZED_DDIR"/"$SLAN"_"$TLAN"/dict."$TLAN".txt "$BINARIZED_DDIR"/"$TLAN"_mono_"$SUBSAMPLE_SIZE"/dict."$SLAN".txt
  echo "$BINARIZED_DDIR"/"$TLAN"_mono_"$SUBSAMPLE_SIZE"
fi


echo "------------------- 3 ---------------------"
if [ ! -a "$CHECKPOINT_DIR"/backtranslation_"$SUBSAMPLE_SIZE"/ ]; then
  mkdir -p "$CHECKPOINT_DIR"/backtranslation_"$SUBSAMPLE_SIZE"/
  CUDA_VISIBLE_DEVICES="3" fairseq-generate \
    $BINARIZED_DDIR/"$TLAN"_mono_"$SUBSAMPLE_SIZE"/ \
    --path "$CHECKPOINT_DIR"/checkpoint_best.pt \
    --skip-invalid-size-inputs-valid-test \
    --max-tokens 4096 \
    --sampling --beam 1 \
    > "$CHECKPOINT_DIR"/backtranslation_"$SUBSAMPLE_SIZE"/backtranslation.sampling.out
  echo "Done generating backtranslation data for $TLAN in:" 
  echo "$CHECKPOINT_DIR"/backtranslation_"$SUBSAMPLE_SIZE"/backtranslation.sampling.out
fi
echo "------------------- 4 ---------------------"

if [ ! -a "$CHECKPOINT_DIR"/backtranslation_"$SUBSAMPLE_SIZE"/bt_data ]; then
  python fairseq/examples/backtranslation/extract_bt_data.py \
    --minlen 1 --maxlen 250 --ratio 1.5 \
    --output "$CHECKPOINT_DIR"/backtranslation_"$SUBSAMPLE_SIZE"/bt_data --srclang "$SLAN" --tgtlang "$TLAN" \
    "$CHECKPOINT_DIR"/backtranslation_"$SUBSAMPLE_SIZE"/backtranslation.sampling.out
fi

echo "------------------- 5 ---------------------"

echo "Binarize the data..."
if [ ! -a $BINARIZED_DDIR/"$SLAN"_"$TLAN"_bt_"$SUBSAMPLE_SIZE"/ ]; then
  fairseq-preprocess \
  --source-lang "$SLAN" \
  --target-lang "$TLAN" \
  --joined-dictionary \
  --srcdict "$BINARIZED_DDIR"/"$SLAN"_"$TLAN"/dict."$SLAN".txt \
  --trainpref "$CHECKPOINT_DIR"/backtranslation_"$SUBSAMPLE_SIZE"/bt_data \
  --destdir $BINARIZED_DDIR/"$SLAN"_"$TLAN"_bt_"$SUBSAMPLE_SIZE"/
fi

echo "------------------- 6 ---------------------"

# We want to train on the combined data, so well symlink the parallel + BT data
# in the wmt18_en_de_para_plus_bt directory. We link the parallel data as "train"
# and the BT data as "train1", so that fairseq will combine them automatically
# and so that we can use the `--upsample-primary` option to upsample the
# parallel data (if desired).
PARA_DATA=$(readlink -f "$BINARIZED_DDIR"/"$SLAN"_"$TLAN")
BT_DATA=$(readlink -f "$BINARIZED_DDIR"/"$SLAN"_"$TLAN"_bt_"$SUBSAMPLE_SIZE")
COMB_DATA="$BINARIZED_DDIR"/"$SLAN"_"$TLAN"_plus_bt_"$SUBSAMPLE_SIZE"
echo "$PARA_DATA"
echo "$BT_DATA"
rm -rf $COMB_DATA
mkdir -p $COMB_DATA
for LANG in "$SLAN" "$TLAN"; do \
    ln -s ${PARA_DATA}/dict.$LANG.txt ${COMB_DATA}/dict.$LANG.txt; \
    for EXT in bin idx; do \
        ln -s ${PARA_DATA}/train."$SLAN"-"$TLAN".$LANG.$EXT ${COMB_DATA}/train."$SLAN"-"$TLAN".$LANG.$EXT; \
        ln -s ${BT_DATA}/train."$SLAN"-"$TLAN".$LANG.$EXT ${COMB_DATA}/train1."$SLAN"-"$TLAN".$LANG.$EXT; \
        ln -s ${PARA_DATA}/valid."$SLAN"-"$TLAN".$LANG.$EXT ${COMB_DATA}/valid."$SLAN"-"$TLAN".$LANG.$EXT; \
        ln -s ${PARA_DATA}/test."$SLAN"-"$TLAN".$LANG.$EXT ${COMB_DATA}/test."$SLAN"-"$TLAN".$LANG.$EXT; \
    done; \
done