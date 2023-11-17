# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import random
from typing import List

from sacrerouge.common.util import flatten
from sacrerouge.data import MetricsDict
from sacrerouge.data.types import SummaryType, ReferenceType
from sacrerouge.metrics import ReferenceBasedMetric


class RandomMetric(ReferenceBasedMetric):

    def score_multi_all(self,
                        summaries_list: List[List[SummaryType]],
                        references_list: List[List[ReferenceType]] = None,
                        **kwargs
                        ) -> List[List[MetricsDict]]:
        summaries_list = [[flatten(summary) for summary in summaries] for summaries in summaries_list]

        metrics_lists = []
        for i, summaries in enumerate(summaries_list):
            metrics_lists.append([])
            for _ in summaries:
                score = random.random()
                metrics_lists[-1].append(MetricsDict({
                    f'random': score,
                }))

        return metrics_lists
