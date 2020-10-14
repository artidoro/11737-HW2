#!/bin/bash

SCRIPTS=mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
NORM_PUNC=$SCRIPTS/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$SCRIPTS/tokenizer/remove-non-printing-char.perl
BPEROOT=subword-nmt/subword_nmt


LANG=en


OUTDIR=data/mono/"$LANG"
orig=orig
tmp=$OUTDIR/tmp
mkdir -p $OUTDIR
mkdir -p $OUTDIR $tmp
mkdir -p $OUTDIR/$orig
SLAN=aze
TLAN=eng
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

URLS=(
    "http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2007.en.shuffled.gz"
)
FILES=(
    "news.2007.en.shuffled.gz"
)

# "http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2008.en.shuffled.gz"
# "http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2009.en.shuffled.gz"
# "http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2010.en.shuffled.gz"
# "http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2011.en.shuffled.gz"
# "http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2012.en.shuffled.gz"
# "http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2013.en.shuffled.gz"
# "http://www.statmt.org/wmt15/training-monolingual-news-crawl-v2/news.2014.en.shuffled.v2.gz"
# "http://data.statmt.org/wmt16/translation-task/news.2015.en.shuffled.gz"
# "http://data.statmt.org/wmt17/translation-task/news.2016.en.shuffled.gz"
# "http://data.statmt.org/wmt18/translation-task/news.2017.en.shuffled.deduped.gz"


# "news.2008.en.shuffled.gz"
# "news.2009.en.shuffled.gz"
# "news.2010.en.shuffled.gz"
# "news.2011.en.shuffled.gz"
# "news.2012.en.shuffled.gz"
# "news.2013.en.shuffled.gz"
# "news.2014.en.shuffled.v2.gz"
# "news.2015.en.shuffled.gz"
# "news.2016.en.shuffled.gz"
# "news.2017.en.shuffled.deduped.gz"

cd "$OUTDIR"/"$orig"
for ((i=0;i<${#URLS[@]};++i)); do
    file=${FILES[i]}
    if [ -f $file ]; then
        echo "$file already exists, skipping download"
    else
        url=${URLS[i]}
        wget "$url"
    fi
done
cd ../../../..

echo "------------------- 0 ---------------------"
echo "$TLAN"
if [ ! -a "$RAW_DDIR"/"$TLAN"_"$SUBSAMPLE_SIZE" ]; then
  mkdir -p "$RAW_DDIR"/"$TLAN"_"$SUBSAMPLE_SIZE"
  f1="$RAW_DDIR"/"$TLAN"_"$SUBSAMPLE_SIZE"/lg.mtok."$TLAN"
  echo $f
  echo $f1
  echo "tokenize $f1..."
  gzip -c -d -k $(for FILE in "${FILES[@]}"; do echo "$OUTDIR"/"$orig"/"$FILE"; done) \
  | shuf -n $SUBSAMPLE_SIZE \
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