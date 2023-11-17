# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import logging
import random

import numpy as np

from post_editese.post_editese_meta_evaluation import MetaEvaluation
from post_editese.post_editese_metrics import benchmark_metrics
from post_editese.post_editese_utils import WMTData

random.seed(42)

LIMIT_NUM_SYSTEMS = False
NUM_SAMPLES = 100

if LIMIT_NUM_SYSTEMS:
    logging.warning("Limiting number of systems to 3")

datasets = [
    ("wmt21.news", "en-de"),
    ("wmt21.news", "en-ru"),
    ("wmt21.news", "zh-en"),
    ("wmt21.tedtalks", "en-de"),
    ("wmt21.tedtalks", "en-ru"),
    ("wmt21.tedtalks", "zh-en"),
]

for dataset_name, language_pair in datasets:
    wmt_data = WMTData(dataset_name, language_pair, True, False)
    meta_evaluation = MetaEvaluation(
        wmt_data=wmt_data,
        exclude_human_systems=True,
        num_iterations=1,
        max_num_systems=3 if LIMIT_NUM_SYSTEMS else None,
    )
    metrics = NUM_SAMPLES * benchmark_metrics.random_baseline_metrics(wmt_data)
    sample_results = []
    print(dataset_name, language_pair)
    for metric in metrics:
        try:
            result = meta_evaluation.compute_pairwise_accuracy(metric)
            sample_results.append(result.observed)
        except NotImplementedError:
            pass
    print()
    print(np.mean(sample_results))
