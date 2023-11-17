# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from typing import Callable, List

from sacrerouge.metrics import Metric


class SegmentLevelMetric:

    def score_segments_single_references(self, hypotheses: List[str], references: List[str]) -> List[float]:
        raise NotImplementedError

    def score_segments_multiple_references(self, hypotheses: List[str], references: List[List[str]]) -> List[float]:
        raise NotImplementedError


class SacreRougeMetric(SegmentLevelMetric):

    def __init__(self, metric: Metric, sacrerouge_metric_name: str):
        self.metric = metric
        self.sacrerouge_metric_name = sacrerouge_metric_name

    def score_segments_single_references(self, hypotheses: List[str], references: List[str], sources: List[str] = None) -> List[float]:
        if sources is not None:
            assert self.is_source_based
            return self.score_segments_multiple_references(hypotheses, [[reference] for reference in references], [[source] for source in sources])
        return self.score_segments_multiple_references(hypotheses, [[reference] for reference in references])

    def score_segments_multiple_references(self, hypotheses: List[str], references: List[List[str]], sources: List[List[str]] = None) -> List[float]:
        assert len(hypotheses) == len(references)
        if sources is not None:
            assert self.is_source_based
            metric_output = self.metric.score_all(hypotheses, references, sources)
        else:
            metric_output = self.metric.score_all(hypotheses, references)
        scores = [d.flatten_keys()[self.sacrerouge_metric_name] for d in metric_output]
        return scores

    @property
    def is_source_based(self) -> bool:
        return "documents" in self.metric.required_context_fields


class MetricLoader:
    def __init__(self, name: str, load_func: Callable):
        self.name = name
        self.load_func = load_func

    def load_metric(self, src_lang: str, tgt_lang: str) -> SegmentLevelMetric:
        return self.load_func(src_lang, tgt_lang)
