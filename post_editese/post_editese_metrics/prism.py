# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from typing import List, Union, Optional

import numpy as np
from sacrerouge.common.util import flatten
from sacrerouge.data import MetricsDict
from sacrerouge.data.types import ReferenceType, SummaryType, DocumentType
from sacrerouge.metrics import Metric, SummaryBasedMetric

from nmtscore import NMTScorer


@Metric.register("prism")
class PrismMetric(SummaryBasedMetric):
    name = "prism"

    def __init__(self,
                 scorer: Union[NMTScorer, str],
                 summaries_lang: str,
                 references_lang: Optional[str],
                 documents_lang: Optional[str],
                 ref_hyp: bool = True,
                 hyp_ref: bool = True,
                 src_hyp: bool = False,
                 hyp_src: bool = False,
                 normalize: bool = False,
                 score_kwargs: dict = None,
                 ):
        super().__init__(required_summary_fields=["summary"], required_context_fields=["references", "documents"])
        if isinstance(scorer, str):
            self.scorer = NMTScorer(scorer)
        else:
            self.scorer = scorer
        self.summaries_lang = summaries_lang
        self.references_lang = references_lang
        self.documents_lang = documents_lang
        assert ref_hyp or hyp_ref or src_hyp or hyp_src
        self.ref_hyp = ref_hyp
        self.hyp_ref = hyp_ref
        self.src_hyp = src_hyp
        self.hyp_src = hyp_src
        self.normalize = normalize
        self.score_kwargs = score_kwargs

    def _score_summaries_given_references(self, summaries: List[str], references: List[str]):
        return self.scorer.score_direct(
            summaries,
            references,
            self.summaries_lang,
            self.references_lang,
            normalize=self.normalize,
            both_directions=False,
            score_kwargs=self.score_kwargs,
        )

    def _score_references_given_summaries(self, summaries: List[str], references: List[str]):
        return self.scorer.score_direct(
            references,
            summaries,
            self.references_lang,
            self.summaries_lang,
            normalize=self.normalize,
            both_directions=False,
            score_kwargs=self.score_kwargs,
        )

    def _score_summaries_given_documents(self, summaries: List[str], documents: List[str]):
        return self.scorer.score_direct(
            summaries,
            documents,
            self.summaries_lang,
            self.documents_lang,
            normalize=self.normalize,
            both_directions=False,
            score_kwargs=self.score_kwargs,
        )

    def _score_documents_given_summaries(self, summaries: List[str], documents: List[str]):
        return self.scorer.score_direct(
            documents,
            summaries,
            self.documents_lang,
            self.summaries_lang,
            normalize=self.normalize,
            both_directions=False,
            score_kwargs=self.score_kwargs,
        )

    def score_all(self,
                  summaries: List[SummaryType],
                  references_list: List[List[ReferenceType]] = None,
                  documents_list: List[List[DocumentType]] = None,
                  **kwargs
                  ) -> List[MetricsDict]:
        references = []
        for reference_set in references_list:
            if len(reference_set) != 1:
                raise NotImplementedError
            references.append(reference_set[0])
        documents = []
        for document_set in documents_list:
            if len(document_set) != 1:
                raise NotImplementedError
            documents.append(document_set[0])

        all_dir_scores = []
        if self.ref_hyp:
            all_dir_scores.append(self._score_summaries_given_references(summaries, references))
        if self.hyp_ref:
            all_dir_scores.append(self._score_references_given_summaries(summaries, references))
        if self.src_hyp:
            all_dir_scores.append(self._score_summaries_given_documents(summaries, documents))
        if self.hyp_src:
            all_dir_scores.append(self._score_documents_given_summaries(summaries, documents))

        average_scores = np.array(all_dir_scores).mean(axis=0)
        metrics_list = [
            MetricsDict({
                self.name: average_scores[i],
            })
            for i in range(len(summaries))
        ]
        return metrics_list

    def score_multi_all(self,
                        summaries_list: List[List[SummaryType]],
                        references_list: List[List[ReferenceType]] = None,
                        documents_list: List[List[DocumentType]] = None,
                        **kwargs
                        ) -> List[List[MetricsDict]]:
        summaries_list = [[flatten(summary) for summary in summaries] for summaries in summaries_list]
        references_list = [[flatten(reference) for reference in references] for references in references_list]
        documents_list = [[flatten(document) for document in documents] for documents in documents_list]
        metrics_lists = []
        for summaries, references, documents in zip(summaries_list, references_list, documents_list):
            metric_list = []
            for summary in summaries:
                all_dir_scores = []
                if self.ref_hyp:
                    all_dir_scores.append(max(self._score_summaries_given_references(len(references) * [summary], references)))
                if self.hyp_ref:
                    all_dir_scores.append(max(self._score_references_given_summaries(len(references) * [summary], references)))
                if self.src_hyp:
                    all_dir_scores.append(max(self._score_summaries_given_documents(len(documents) * [summary], documents)))
                if self.hyp_src:
                    all_dir_scores.append(max(self._score_documents_given_summaries(len(documents) * [summary], documents)))
                average_score = np.mean(all_dir_scores)
                metric_list.append(
                    MetricsDict({
                        self.name: average_score,
                    })
                )
            metrics_lists.append(metric_list)
        return metrics_lists
