# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import hashlib
from copy import deepcopy
from typing import List, Callable, Optional, Set, Tuple

import numpy as np
from sacrerouge.data import MetricsDict
from sacrerouge.metrics import SummaryBasedMetric
from sklearn.preprocessing import MinMaxScaler

import post_editese.post_editese_utils as utils


class GradedReferenceMetric:

    def __init__(self,
                 wmt_data: utils.WMTData,
                 references_loader: utils.ReferencesLoader = None,
                 normalize_reference_scores: bool = True,
                 ):
        self.wmt_data = wmt_data
        references_loader = references_loader or utils.ReferencesLoader(wmt_data)
        self.normalize_scores = normalize_reference_scores
        self.reference_lists = references_loader.get_reference_lists()
        if self.normalize_scores:
            self.normalize_reference_scores()

    def normalize_reference_scores(self):
        reference_scores = list()
        for reference_list in self.reference_lists:
            for reference in reference_list:
                if reference.score is None:
                    continue
                reference_scores.append(reference.score)
        if len(set(reference_scores)) == 1:
            # Every reference has the same score
            for reference_list in self.reference_lists:
                for reference in reference_list:
                    reference.score = 1
        else:
            # References are graded
            score_scaler = MinMaxScaler()
            score_scaler.fit(np.array(reference_scores).reshape(-1, 1))
            for reference_list in self.reference_lists:
                for reference in reference_list:
                    if reference.score is None:
                        continue
                    reference.score = score_scaler.transform(np.array(reference.score).reshape(-1, 1)).item()
                    assert 0 <= reference.score <= 1

    def print_stats(self) -> None:
        print("Number of documents with references:", sum(bool([r for r in reference_list if r.score is not None]) for reference_list in self.reference_lists))
        reference_counts = [len(reference_list) for reference_list in self.reference_lists if reference_list]
        print("Average number of references:", np.mean(reference_counts))
        print("Reference systems", list(sorted(self.get_reference_systems())))

    def get_reference_systems(self) -> Set[str]:
        reference_systems = set()
        for reference_list in self.reference_lists:
            for reference in reference_list:
                reference_systems.add(reference.system)
        return reference_systems

    def exclude_systems_from_references(self, reference_lists, exclude_systems: Set[str]):
        if exclude_systems:
            systems_related_to_excluded = deepcopy(exclude_systems)
            for system in exclude_systems:
                for family in utils.SYSTEM_FAMILIES:
                    if system in family:
                        systems_related_to_excluded |= set(family)
            filtered_reference_lists = []
            for reference_list in reference_lists:
                filtered_reference_lists.append([
                    reference for reference in reference_list
                    if reference.system not in systems_related_to_excluded
                ])
        else:
            filtered_reference_lists = reference_lists
        return filtered_reference_lists


class SegmentLevelMetric(GradedReferenceMetric):

    def __init__(self,
                 metric: SummaryBasedMetric,
                 metric_name: str,
                 wmt_data: utils.WMTData,
                 references_loader: utils.ReferencesLoader = None,
                 mask_by_references_loader: utils.ReferencesLoader = None,
                 normalize_reference_scores: bool = True,
                 ):
        super().__init__(wmt_data, references_loader, normalize_reference_scores)
        self.metric = metric
        self.metric_name = metric_name
        if metric.required_context_fields == ["references", "documents"] and "prism" not in metric_name.lower():
            self.takes_src = True
        elif metric.required_context_fields == ["references"] or "prism" in metric_name.lower():
            self.takes_src = False
        else:
            raise NotImplementedError
        self.references_loader = references_loader
        # Only use references if a reference would also be available from mask_by_references_loader
        if mask_by_references_loader is not None:
            self.mask_by_segment_level_metric = SegmentLevelMetric(metric, metric_name, wmt_data, mask_by_references_loader)
        else:
            self.mask_by_segment_level_metric = None
        # For dataset statistics
        self._num_segments_with_a_reference_per_system_pair: List[int] = []

    def score_segments(self,
                       hypotheses: Tuple[str, ...],
                       exclude_systems: Tuple[str, ...] = None,
                       ) -> List[Optional[float]]:
        """
        A score is None for a segment if there is no reference available
        """
        assert len(hypotheses) == len(self.reference_lists)
        exclude_systems = set(exclude_systems or ())
        reference_lists = self.exclude_systems_from_references(self.reference_lists, exclude_systems)
        for i in range(len(reference_lists)):
            reference_lists[i] = [
                reference for reference in reference_lists[i]
                if reference is not None and reference.score is not None
            ]
        if self.references_loader.sample_single_reference:
            for i in range(len(reference_lists)):
                # Quasi-random shuffle, but stable across experiments
                reference_lists[i] = sorted(reference_lists[i], key=lambda s: hashlib.md5(s.text.encode()).hexdigest())
                reference_lists[i] = reference_lists[i][:1]

        if self.mask_by_segment_level_metric is not None:
            mask_by_reference_lists = self.mask_by_segment_level_metric.exclude_systems_from_references(
                self.mask_by_segment_level_metric.reference_lists,
                exclude_systems
            )
            assert len(mask_by_reference_lists) == len(reference_lists)
            for i in range(len(mask_by_reference_lists)):
                mask_by_reference_list = [
                    reference for reference in mask_by_reference_lists[i]
                    if reference is not None and reference.score is not None
                ]
                if not mask_by_reference_list:
                    reference_lists[i] = []

        self._num_segments_with_a_reference_per_system_pair.append(sum([bool(reference_list) for reference_list in reference_lists]))

        # Flatten the input to call metric only once
        scores = []
        hypotheses_list = []
        references_list = []
        flat_to_original_indices = []
        if self.takes_src:
            documents_list = []
        for i in range(len(hypotheses)):
            hypothesis = hypotheses[i]
            reference_list = reference_lists[i]
            if not reference_list:
                scores.append(None)
                continue
            else:
                assert len(reference_list) == 1
                scores.append(0)  # Placeholder for eventual value
            hypotheses_list += len(reference_list) * [hypothesis]
            references_list += [[reference.text] for reference in reference_list]
            if self.takes_src:
                src_sequence = self.wmt_data.source_sentences[i]
                documents_list += len(reference_list) * [src_sequence]
            flat_to_original_indices.append(i)

        if self.takes_src:
            metric_output: List[List[MetricsDict]] = self.metric.score_all(
                hypotheses_list,
                references_list,
                documents_list,
            )
        else:
            metric_output: List[List[MetricsDict]] = self.metric.score_all(
                hypotheses_list,
                references_list,
            )
        for i in range(len(metric_output)):
            scores[flat_to_original_indices[i]] = metric_output[i][self.metric_name]
        assert len(scores) == len(hypotheses)
        return scores


class MetricLoader:
    def __init__(self, name: str, load_func: Callable):
        self.name = name
        self.load_func = load_func

    def load_metric(self, lang: str) -> SegmentLevelMetric:
        return self.load_func(lang)
