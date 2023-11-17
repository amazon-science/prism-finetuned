# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import json

import dataclasses
import logging

from post_editese.post_editese_meta_evaluation import MetaEvaluation
from post_editese.post_editese_metrics import benchmark_metrics
from post_editese.post_editese_utils import WMTData

PRINT_SYSTEM_SCORES = True
LIMIT_NUM_SYSTEMS = False

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
    metrics = benchmark_metrics.chrf_metrics(wmt_data)
    results = dict()
    print(dataset_name, language_pair)
    for metric in metrics:
        try:
            result = meta_evaluation.compute_pairwise_accuracy(metric)
            print(f"{metric.name}\t{result.observed}")
            result_dict = dataclasses.asdict(result)
            result_dict["testset"] = f"{dataset_name} {language_pair}"
            print(json.dumps(result_dict))
            results[metric.name] = result
        except NotImplementedError:
            print(f"{metric.name}\t")
    print()
    for name, result in results.items():
        print(name, result.observed)
    print()
    if PRINT_SYSTEM_SCORES:
        for metric in metrics:
            if metric.name not in results:
                continue
            print(metric.name)
            for system, system_score in sorted(results[metric.name].system_scores.items(), key=lambda x: x[0]):
                print(f"{system}\t{system_score}")
            print()
        print()
