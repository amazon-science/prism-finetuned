# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

cd data/wmt_rr

head -n 121218 wmt20.en-de.csv > wmt20.en-de.train.csv
head -n 1 wmt20.en-de.csv > wmt20.en-de.valid.csv
tail -n 5000 wmt20.en-de.csv >> wmt20.en-de.valid.csv

head -n 159138 wmt20.zh-en.csv > wmt20.zh-en.train.csv
head -n 1 wmt20.zh-en.csv > wmt20.zh-en.valid.csv
tail -n 5000 wmt20.zh-en.csv >> wmt20.zh-en.valid.csv

cat wmt20.en-de.train.csv > wmt20.train.csv
cat wmt20.zh-en.train.csv | tail +2 >> wmt20.train.csv

cat wmt20.en-de.valid.csv > wmt20.valid.csv
cat wmt20.zh-en.valid.csv | tail +2 >> wmt20.valid.csv

for filepath in wmt20.en-de.train.csv wmt20.en-de.valid.csv wmt20.zh-en.train.csv wmt20.zh-en.valid.csv wmt20.train.csv wmt20.valid.csv; do
  head -n 1 $filepath > "${filepath}.shuf"
  tail -n +2 $filepath | shuf >> "${filepath}.shuf"
done
