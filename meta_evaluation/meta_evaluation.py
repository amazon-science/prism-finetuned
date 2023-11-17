# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import os
from collections import OrderedDict, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import scipy

from meta_evaluation.metrics import MetricLoader
from mt_metrics_eval_custom import stats
from mt_metrics_eval_custom.data import EvalSet
from mt_metrics_eval_custom.stats import Correlation


class MetaEvaluation:

    def __init__(self, eval_set: EvalSet, metric_loaders: List[MetricLoader]):
        self.eval_set = eval_set
        self.metric_loaders = metric_loaders
        self.seg_scores: Dict[str, Dict[str, List[float]]] = OrderedDict()  # metric name -> system -> scores
        self.sys_scores: Dict[str, Dict[str, List[float]]] = OrderedDict()  # metric name -> system -> [score]
        self.correlations: Dict[str, Correlation] = dict()

    def run_metrics(self, use_cache: bool = True, add_average: bool = False):
        references = self.eval_set.all_refs[self.eval_set.std_ref]
        for metric_loader in self.metric_loaders:
            self.seg_scores[metric_loader.name] = defaultdict(list)
            self.sys_scores[metric_loader.name] = defaultdict(list)

            # Retrieve scores from cache, if available
            cache_path_sys = self.get_metric_cache_path(metric_loader.name, self.eval_set.std_ref, "sys")
            cache_path_seg = self.get_metric_cache_path(metric_loader.name, self.eval_set.std_ref, "seg")
            if use_cache and cache_path_sys.exists() and cache_path_seg.exists():
                with open(cache_path_sys) as f:
                    for line in f:
                        system, score_str = line.strip().split("\t")
                        score = float(score_str)
                        self.sys_scores[metric_loader.name][system].append(score)
                with open(cache_path_seg) as f:
                    for line in f:
                        system, score_str = line.strip().split("\t")
                        score = float(score_str)
                        self.seg_scores[metric_loader.name][system].append(score)
                continue

            # Otherwise, call metric
            metric = metric_loader.load_metric(self.src_lang, self.tgt_lang)
            for system, hypotheses in self.eval_set.sys_outputs.items():
                if getattr(metric, "is_source_based", False):
                    sources = self.eval_set.src
                    seg_scores = metric.score_segments_single_references(hypotheses, references, sources)
                else:
                    seg_scores = metric.score_segments_single_references(hypotheses, references)
                self.seg_scores[metric_loader.name][system] = seg_scores
                sys_score = np.mean(seg_scores)
                self.sys_scores[metric_loader.name][system] = [sys_score]

            # Write to cache
            os.makedirs(cache_path_sys.parent, exist_ok=True)
            with open(cache_path_sys, "w") as f:
                for system, scores in self.sys_scores[metric_loader.name].items():
                    for score in scores:
                        f.write(f"{system}\t{score}\n")
            with open(cache_path_seg, "w") as f:
                for system, scores in self.seg_scores[metric_loader.name].items():
                    for score in scores:
                        f.write(f"{system}\t{score}\n")

        if add_average:
            metric_mins = defaultdict(int)
            metric_maxs = defaultdict(int)
            for metric, system_scores in sorted(self.seg_scores.items()):
                for system, seg_scores in sorted(system_scores.items()):
                    metric_mins[metric] = min(metric_mins[metric], min(seg_scores))
                    metric_maxs[metric] = max(metric_maxs[metric], max(seg_scores))

            average_seg_scores = dict()
            for metric, system_scores in sorted(self.seg_scores.items()):
                for system, seg_scores in sorted(system_scores.items()):
                    if system not in average_seg_scores:
                        average_seg_scores[system] = [0] * len(seg_scores)
                    for i in range(len(seg_scores)):
                        average_seg_scores[system][i] += seg_scores[i] - metric_mins[metric] / (metric_maxs[metric] - metric_mins[metric])
            average_sys_scores = dict()
            for system, seg_scores in sorted(average_seg_scores.items()):
                average_sys_scores[system] = np.mean(seg_scores)
            self.seg_scores["Average"] = average_seg_scores
            self.sys_scores["Average"] = average_sys_scores

    def evaluate(self,
                 meta_metric: str = "pairwise_accuracy",
                 include_human_systems: bool = False,
                 skip_segments_without_gold_scores: bool = True,
                 human_score_type: str = None,
                 only_metricsystems: bool = False,
                 ):
        assert meta_metric in {"pairwise_accuracy", "segment_level_kendall_correlation"}
        assert self.sys_scores, "call run_metrics() first"
        if only_metricsystems:
            assert "wmt21" in self.eval_set.name
        human_score_type = human_score_type if human_score_type is not None else self.eval_set.StdHumanScoreName("sys")
        gold_scores_seg = self.eval_set.Scores("seg", human_score_type)
        system_names = set(gold_scores_seg)
        # print("Number of segments: ", len(gold_scores_seg[list(system_names)[0]]))
        # print("Number of annotated segments: ", len([score for score in gold_scores_seg[list(system_names)[0]] if score is not None]))
        if not include_human_systems:
            system_names -= self.eval_set.human_sys_names
        if only_metricsystems:
            system_names = {system for system in system_names if "metricsystem" in system}
            assert len(system_names) == 5
        results = []
        for metric_name, scores in self.sys_scores.items():
            predicted_scores_seg = self.seg_scores[metric_name]
            for system in system_names:
                for i, gold_score in enumerate(gold_scores_seg[system]):
                    if gold_score is None:
                        if skip_segments_without_gold_scores:
                            predicted_scores_seg[system][i] = None
            # print("Number of systems: ", len(system_names))
            # print("Systems: ", system_names)
            corr = self.eval_set.Correlation(gold_scores_seg, predicted_scores_seg, system_names)
            if meta_metric == "pairwise_accuracy":
                result = corr.PairwiseAccuracy()[0]
                # print(f'{metric_name}\tPairwiseAccuracy{"(metricsystems)" if only_metricsystems else ""}={result:f}')
            elif meta_metric == "segment_level_kendall_correlation":
                result = corr.Kendall()[0]
                # print(f'{metric_name}\tSegmentLevelKendall{"(metricsystems)" if only_metricsystems else ""}={result:f}')
            results.append(result)
            self.correlations[metric_name] = corr
        return results

    def get_significance_cluster(self,
                                 meta_metric: str = "pairwise_accuracy",
                                 alpha: float = 0.05,
                                 num_samples: int = 1000,
                                 ):
        assert meta_metric in {"pairwise_accuracy", "segment_level_kendall_correlation"}
        assert self.correlations, "call evaluate() first"
        pair_significances: Dict[Tuple[str, str], bool] = defaultdict(bool)

        def _SigTest(corr1, corr2, v1, v2):
            better = v2[0] >= v1[0]
            if not better:
                corr2, corr1 = corr1, corr2
            if meta_metric == "pairwise_accuracy":
                corr_func = stats.PairwiseAccuracy(corr1.num_sys)
            elif meta_metric == "segment_level_kendall_correlation":
                corr_func = stats.CorrFunction(scipy.stats.kendalltau, filter_nones=True)
            corr1 = deepcopy(corr1)
            corr2 = deepcopy(corr2)
            missing_segment_ids = set()
            for i in range(len(corr1.gold_scores)):
                if corr1.gold_scores[i] is None:
                    missing_segment_ids.add(i)
                if corr2.gold_scores[i] is None:
                    missing_segment_ids.add(i)
            corr1.gold_scores = [value for i, value in enumerate(corr1.gold_scores) if i not in missing_segment_ids]
            corr2.gold_scores = [value for i, value in enumerate(corr2.gold_scores) if i not in missing_segment_ids]
            corr1.metric_scores = [value for i, value in enumerate(corr1.metric_scores) if i not in missing_segment_ids]
            corr2.metric_scores = [value for i, value in enumerate(corr2.metric_scores) if i not in missing_segment_ids]
            assert corr1.gold_scores == corr2.gold_scores
            p = stats.PermutationSigDiff(corr1, corr2, corr_func, num_samples)
            return better, p

        for i in range(len(self.metric_loaders)):
            for j in range(len(self.metric_loaders)):
                if i <= j:
                    continue
                metric_loader1 = self.metric_loaders[i]
                metric_loader2 = self.metric_loaders[j]
                corr1 = self.correlations[metric_loader1.name]
                corr2 = self.correlations[metric_loader2.name]
                if meta_metric == "pairwise_accuracy":
                    v1 = corr1.PairwiseAccuracy()
                    v2 = corr2.PairwiseAccuracy()
                elif meta_metric == "segment_level_kendall_correlation":
                    v1 = corr1.Kendall()
                    v2 = corr2.Kendall()
                pa_b, pa_p = _SigTest(corr1, corr2, v1, v2)
                if pa_p >= alpha:
                    continue
                if pa_b:  # 2 is better than 1
                    pair_significances[metric_loader2.name, metric_loader1.name] = True
                else:  # 1 is better than 2
                    pair_significances[metric_loader1.name, metric_loader2.name] = True

        significance_cluster = []
        for metric_loader in self.metric_loaders:
            is_in_cluster = True
            for (metric1, metric2), is_better in pair_significances.items():
                if is_better and metric2 == metric_loader.name:
                    is_in_cluster = False
            if is_in_cluster:
                significance_cluster.append(metric_loader.name)
        return significance_cluster

    @property
    def cache_dir(self) -> Path:
        base_dir = Path(os.getenv("META_EVALUATION_CACHE", Path.home() / ".cache" / "MetricMetaEvaluation"))
        return base_dir / self.eval_set.name / "metric-scores" / self.eval_set.lp

    def get_metric_cache_path(self, metric_name: str, ref_name: str, level: str) -> Path:
        return self.cache_dir / f"{metric_name}-{ref_name}.{level}.score"

    @property
    def src_lang(self) -> str:
        return self.eval_set.lp.split("-")[0]

    @property
    def tgt_lang(self) -> str:
        return self.eval_set.lp.split("-")[1]
