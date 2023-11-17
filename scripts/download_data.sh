# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

mkdir data
mkdir data/wmt_mqm_orig

cd data/wmt_mqm_orig
wget https://github.com/google/wmt-mqm-human-evaluation/raw/main/newstest2020/ende/mqm_newstest2020_ende.no-postedits.tsv
wget https://github.com/google/wmt-mqm-human-evaluation/raw/main/newstest2020/zhen/mqm_newstest2020_zhen.no-postedits.tsv
