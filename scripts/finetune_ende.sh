# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

MODEL_NAME=finetuned_ende

BASE_DIR=..
DATA_DIR=$BASE_DIR/data/wmt20_split/data-bin
PRETRAINED_MODEL_DIR=$BASE_DIR/models/m39v1
OUT_MODEL_DIR=$BASE_DIR/models/$MODEL_NAME

fairseq-train $DATA_DIR \
    --save-dir $OUT_MODEL_DIR \
    --user-dir ../prism_finetuning  \
    --finetune-from-model $PRETRAINED_MODEL_DIR/checkpoint.pt \
    --arch transformer  \
    --encoder-embed-dim 1280  \
    --encoder-ffn-embed-dim 12288  \
    --encoder-layers 8  \
    --encoder-attention-heads 20  \
    --decoder-embed-dim 1280  \
    --decoder-ffn-embed-dim 12288  \
    --decoder-layers 8  \
    --decoder-attention-heads 20  \
    --share-all-embeddings  \
    --share-decoder-input-output-embed  \
    --task multilingual_translation_ranking  \
    --lang-pairs en-de  \
    --train-ref-hyp  \
    --train-hyp-ref  \
    --decoder-langtok  \
    --criterion max_margin  \
    --max-tokens 450  \
    --optimizer adam  \
    --lr-scheduler inverse_sqrt \
    --clip-norm 1.2 \
    --lr 1e-4 \
    --fp16 \
    --memory-efficient-fp16 \
    --num-workers 6 \
    --keep-last-epochs 100 \
    --patience 1000 \
    --warmup-updates 10 \
    --tensorboard-logdir $BASE_DIR/logs/$MODEL_NAME \
    --log-interval 20 \
    --margin 0.1 \
    --update-freq 800 \
    --length-normalize-scores \
    --train-cross-entropy \
    --cross-entropy-weight 0.1 \
    --all-gather-list-size 2000000 \
    --max-epoch 1
