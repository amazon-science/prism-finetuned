# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import os
from pathlib import Path

from meta_evaluation.meta_evaluation import MetaEvaluation
from meta_evaluation.metrics.benchmark_metrics import all_metrics
from mt_metrics_eval_custom.data import EvalSet


cache_dir = Path(__file__).parent.parent / "predictions"
assert cache_dir.exists()
os.environ["META_EVALUATION_CACHE"] = str(cache_dir)


PRINT_SIGNIFICANCE_CLUSTER = False

metric_names = [
    "SentBLEU",
    "ChrF",
    "COMET (wmt21-comet-mqm)",
    "Prism-ref",
    "Prism-paper_ende_checkpoint1.pt-ref",
    "Prism-paper_zhen_checkpoint1.pt-ref",
    "Prism-paper_main_checkpoint1.pt-ref",
]

# testset_name = "wmt20"
testset_name = "wmt21.news"
# testset_name = "wmt21.tedtalks"

lang_pairs = [
    "en-de",
    "en-ru",
    "zh-en",
]

human_score_type = "mqm"
skip_segments_without_gold_scores = True

rows = []

for metric_name in metric_names:
    row = []
    row.append(metric_name)
    metric = all_metrics(device=None)[metric_name]
    for lang_pair in lang_pairs:
        testset = EvalSet(testset_name, lang_pair)
        meta_evaluation = MetaEvaluation(testset, [metric])
        meta_evaluation.run_metrics(use_cache=True)
        results=meta_evaluation.evaluate(
            "segment_level_kendall_correlation",
            include_human_systems=False,
            human_score_type=human_score_type,
            skip_segments_without_gold_scores=skip_segments_without_gold_scores,
        )
        result = list(results)[0]
        row.append(str(result))
    row.append("")
    for lang_pair in lang_pairs:
        testset = EvalSet(testset_name, lang_pair)
        meta_evaluation = MetaEvaluation(testset, [metric])
        meta_evaluation.run_metrics(use_cache=True)
        results = meta_evaluation.evaluate(
            "pairwise_accuracy",
            include_human_systems=False,
            human_score_type=human_score_type,
            skip_segments_without_gold_scores=skip_segments_without_gold_scores,
        )
        result = list(results)[0]
        row.append(str(result))
    row.append("")
    rows.append(row)

print()
for row in rows:
    print("\t".join(row))

if PRINT_SIGNIFICANCE_CLUSTER:
    print("Top significance clusters:")
    for meta_metric in [
        "pairwise_accuracy",
        "segment_level_kendall_correlation",
    ]:
        print(meta_metric)
        for lang_pair in lang_pairs:
            print(lang_pair)
            testset = EvalSet(testset_name, lang_pair)
            metrics = [all_metrics(device=0)[metric_name] for metric_name in metric_names]
            meta_evaluation = MetaEvaluation(testset, metrics)
            meta_evaluation.run_metrics(use_cache=True)
            results = meta_evaluation.evaluate(
                meta_metric,
                include_human_systems=False,
                human_score_type=human_score_type,
                skip_segments_without_gold_scores=skip_segments_without_gold_scores,
            )
            cluster = meta_evaluation.get_significance_cluster(
                meta_metric,
                num_samples=1000,
            )
            print(cluster)
