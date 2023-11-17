# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import itertools
import random
from collections import defaultdict

import scipy
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import numpy as np
from tqdm import tqdm

from meta_evaluation.metrics import MetricLoader
from post_editese.post_editese_utils import WMTData


@dataclass
class ConfidenceInterval:
    metric_name: str
    lower: float
    observed: float
    upper: float


@dataclass
class PairwiseAccuracyResult(ConfidenceInterval):
    system_scores: Dict[str, float]
    average_number_of_used_segments: float = None
    segment_level_correlation: float = None


class MetaEvaluation:

    def __init__(self,
                 wmt_data: WMTData,
                 exclude_human_systems: bool = True,
                 num_iterations: int = 1000,
                 alpha: float = 0.05,
                 max_num_systems: int = None  # To speed up testing
                 ):
        self.wmt_data = wmt_data
        self.exclude_human_systems = exclude_human_systems
        self.num_iterations = num_iterations
        self.alpha = alpha
        self.systems = self.wmt_data.systems_with_mqm_annotation
        # WMT21 EN–DE always discards "refB" system
        if self.wmt_data.name == "wmt21.news" and self.wmt_data.language_pair == "en-de":
            self.systems = [system for system in self.systems if system != "refB"]
        if exclude_human_systems:
            self.systems = [
                system for system in self.systems
                if system not in self.wmt_data.reference_names and "human" not in system.lower()
            ]
        if max_num_systems is not None:
            self.systems = self.systems[:max_num_systems]
        print("Meta-evaluating metrics using systems: ", self.systems)
        print("Number of system pairs: ", len(self.system_pairs))

    @property
    def system_pairs(self) -> List[Tuple[str, str]]:
        return [
            (system1, system2)
            for system1, system2 in itertools.product(self.systems, repeat=2)
            if system1 < system2
        ]

    @property
    def tgt_lang(self) -> str:
        return self.wmt_data.language_pair.split("-")[1]

    @staticmethod
    def _bootstrap_both_sample_for_pairwise_accuracy(*pairwise_matrices: np.ndarray, gold_labels: np.ndarray):
        """
        # Adapted from https://github.com/danieldeutsch/sacrerouge/sacrerouge/stats.py
        Resamples new matrices by sampling both systems and inputs with replacement. The sample will be the intersection of
        the sampled systems and inputs. The sample will be taken in parallel for all of the input matrices.
        """
        N, M = pairwise_matrices[0].shape
        for matrix in pairwise_matrices:
            assert matrix.shape == (N, M)
        assert gold_labels.shape == (N,)
        rows = np.random.choice(N, N, replace=True)
        cols = np.random.choice(M, M, replace=True)
        pairwise_matrices_samples = [matrix[rows][:, cols] for matrix in pairwise_matrices]
        gold_labels_samples = gold_labels[rows]
        return tuple([*pairwise_matrices_samples, gold_labels_samples])

    @staticmethod
    def _pairwise_accuracy(system1_segment_scores: np.ndarray, system2_segment_scores: np.ndarray, gold_labels: np.ndarray) -> Optional[float]:
        assert system1_segment_scores.shape == system2_segment_scores.shape
        assert system1_segment_scores.shape[0] == gold_labels.shape[0]
        if not system1_segment_scores.size:
            return None
        system1_scores = np.nanmean(system1_segment_scores, axis=1)
        system2_scores = np.nanmean(system2_segment_scores, axis=1)
        predicted_labels = system1_scores >= system2_scores
        is_correct = predicted_labels == gold_labels
        is_correct[np.isnan(system1_scores) | np.isnan(system2_scores)] = np.nan
        accuracy = np.nansum(is_correct) / np.count_nonzero(~np.isnan(is_correct))
        return accuracy if not np.isnan(accuracy) else None

    def compute_pairwise_accuracy(self, metric_loader: MetricLoader) -> PairwiseAccuracyResult:
        """
        Uses bootstrap resampling (BOOT-BOTH) for now
        """
        metric = metric_loader.load_metric(self.tgt_lang)
        num_pairs = len(self.system_pairs)
        num_documents = len(self.wmt_data)

        # Compute segment-level scores of systems w.r.t all system pairs
        system1_matrix = np.zeros((num_pairs, num_documents))  # First element of pair
        system2_matrix = np.zeros((num_pairs, num_documents))  # Second element of pair
        for i, (system1, system2) in enumerate(tqdm(self.system_pairs)):
            systems_excluded_from_references = (system1, system2)
            system1_hypotheses = self.wmt_data.get_translations(system1)
            system2_hypotheses = self.wmt_data.get_translations(system2)
            system1_segment_scores = metric.score_segments(
                hypotheses=tuple(system1_hypotheses),
                exclude_systems=systems_excluded_from_references,
            )
            system2_segment_scores = metric.score_segments(
                hypotheses=tuple(system2_hypotheses),
                exclude_systems=systems_excluded_from_references,
            )
            system1_matrix[i] = system1_segment_scores
            system2_matrix[i] = system2_segment_scores
            # Only the MQM-annotated segments are used for calculating system averages (WMT21 metrics paper p.742)
            system1_excluded_segments = self.wmt_data.get_indices_of_unannotated_segments(system1)
            system2_excluded_segments = self.wmt_data.get_indices_of_unannotated_segments(system2)
            system1_matrix[i][system1_excluded_segments] = np.nan
            system2_matrix[i][system2_excluded_segments] = np.nan

        # Compute gold labels for pairwise comparisons
        gold_labels = np.zeros(num_pairs)
        for i, (system1, system2) in enumerate(self.system_pairs):
            system1_gold_score = self.wmt_data.get_system_score(system1)
            system2_gold_score = self.wmt_data.get_system_score(system2)
            gold_labels[i] = system1_gold_score >= system2_gold_score

        # Calculate test statistic
        observed = self._pairwise_accuracy(system1_matrix, system2_matrix, gold_labels)

        # Calculate confidence interval using bootstrap resampling
        # Code is inspired by https://github.com/danieldeutsch/sacrerouge/sacrerouge/stats.py
        sample_accuracies = []
        for _ in range(self.num_iterations):
            x, y, z = self._bootstrap_both_sample_for_pairwise_accuracy(system1_matrix, system2_matrix, gold_labels=gold_labels)
            accuracy = self._pairwise_accuracy(x, y, z)
            if accuracy is not None:
                # Value is ignored if it is NaN
                sample_accuracies.append(accuracy)
        lower = np.percentile(sample_accuracies, self.alpha / 2 * 100)
        upper = np.percentile(sample_accuracies, (1.0 - self.alpha / 2) * 100)

        # Calculate system scores (average across all pairwise comparisons)
        system_scores = defaultdict(list)
        for i, (system1, system2) in enumerate(self.system_pairs):
            system_scores[system1].append(np.nanmean(system1_matrix[i]))
            system_scores[system2].append(np.nanmean(system2_matrix[i]))
        average_system_scores = {system: np.nanmean(system_scores[system]) for system in system_scores}

        # Calculate segment-level correlation
        flat_segment_level_scores_metric = []
        flat_segment_level_scores_gold = []
        for system in self.systems:
            for j in range(num_documents):
                if j in self.wmt_data.get_indices_of_unannotated_segments(system):
                    continue
                # Pick a random pairwise comparison to retrieve segment score
                try:
                    all_relevant_pair_indices = [i for i, (system1, system2) in enumerate(self.system_pairs) if system == system1]
                    sampled_pair_index = random.choice(all_relevant_pair_indices)
                    metric_score = system1_matrix[sampled_pair_index][j]
                except IndexError:  # system was on right side of comparison
                    all_relevant_pair_indices = [i for i, (system1, system2) in enumerate(self.system_pairs) if system == system2]
                    sampled_pair_index = random.choice(all_relevant_pair_indices)
                    metric_score = system2_matrix[sampled_pair_index][j]
                flat_segment_level_scores_metric.append(metric_score)
                gold_score = self.wmt_data.get_segment_score(system, j)
                flat_segment_level_scores_gold.append(gold_score)
        segment_level_correlation = scipy.stats.kendalltau(flat_segment_level_scores_metric, flat_segment_level_scores_gold, nan_policy="omit")[0]

        return PairwiseAccuracyResult(
            metric_name=metric_loader.name,
            lower=lower,
            observed=observed,
            upper=upper,
            system_scores=average_system_scores,
            average_number_of_used_segments=np.mean(metric._num_segments_with_a_reference_per_system_pair),
            segment_level_correlation=segment_level_correlation,
        )
