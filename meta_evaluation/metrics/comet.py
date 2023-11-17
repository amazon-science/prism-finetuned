# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from pathlib import Path
from typing import List, Union

from comet import load_from_checkpoint, download_model
from sacrerouge.common.util import flatten
from sacrerouge.data import MetricsDict
from sacrerouge.data.types import SummaryType, ReferenceType, DocumentType
from sacrerouge.metrics import SummaryBasedMetric


class CometMetric(SummaryBasedMetric):

    def __init__(self,
                 model_name_or_path: Union[Path, str] = None,
                 device=None,
                 ):
        super().__init__(required_summary_fields=["summary"], required_context_fields=["references", "documents"])
        self.model_name_or_path = model_name_or_path
        self.device = device
        if Path(model_name_or_path).exists():
            checkpoint_path = str(model_name_or_path)
        else:
            checkpoint_path = download_model(str(model_name_or_path))
        self.model = load_from_checkpoint(checkpoint_path)

    def score_multi_all(self,
                        summaries_list: List[List[SummaryType]],
                        references_list: List[List[ReferenceType]] = None,
                        documents_list: List[List[DocumentType]] = None,
                        **kwargs
                        ) -> List[List[MetricsDict]]:
        assert references_list is not None
        assert documents_list is not None
        summaries_list = [[flatten(summary) for summary in summaries] for summaries in summaries_list]
        references_list = [[flatten(reference) for reference in references] for references in references_list]
        documents_list = [[flatten(documents)] for documents in documents_list]

        # Create the candidate and reference lists for passing to the scoring function
        data = []
        for i, (summaries, references, documents) in enumerate(zip(summaries_list, references_list, documents_list)):
            for j, summary in enumerate(summaries):
                if len(references) > 1:
                    raise NotImplementedError
                if len(documents) > 1:
                    print(documents)
                    raise NotImplementedError
                data.append({
                    "src": documents[0],
                    "mt": summary,
                    "ref": references[0],
                })
        # Score the summaries
        scores, _ = self.model.predict(data, progress_bar=True, gpus=(1 if self.device is None else 1))

        # Remap the scores to the summaries
        index = 0
        metrics_lists = []
        for i, summaries in enumerate(summaries_list):
            metrics_lists.append([])
            for _ in summaries:
                score = scores[index]
                index += 1
                metrics_lists[-1].append(MetricsDict({
                    f'comet': score,
                }))

        return metrics_lists
