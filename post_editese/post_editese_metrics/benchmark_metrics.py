# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import itertools
import logging
from pathlib import Path
from typing import List, Dict

from post_editese.post_editese_metrics import MetricLoader, SegmentLevelMetric
from post_editese.post_editese_utils import ReferencesLoader

logging.basicConfig(level=logging.INFO)


reference_settings = {
    "standard": {},
    "alternative": {
        "use_standard_references": False,
        "use_alternative_references": True,
    },
    "alternative2": {
        "use_standard_references": False,
        "use_alternative_references2": True,
    },
    "alternative3": {
        "use_standard_references": False,
        "use_alternative_references3": True,
    },
    "error-free system translation": {
        "use_standard_references": False,
        "use_peer_references": True,
        "exclude_flawed_references": True,
        "exclude_segments_without_annotation": True,
        "sample_single_reference": True,
    },
}


def sentbleu_metrics(wmt_data, device=None) -> List[MetricLoader]:
    from sacrerouge.metrics import SentBleu
    metrics = []
    for setting in [
        "error-free system translation",
    ]:
        kwargs = reference_settings[setting]
        metrics.append(MetricLoader(
            name=f"SentBLEU ({setting})",
            load_func=lambda lang, kwargs=kwargs: SegmentLevelMetric(
                SentBleu(trg_lang=lang, tokenize=None),
                "sent-bleu",
                wmt_data,
                references_loader=ReferencesLoader(wmt_data, **kwargs),
                normalize_reference_scores=True,  # Use reference scores to weight segment average
            ),
        ))
    for setting in [
        "standard if error-free system translation exists",
        "alternative if error-free system translation exists",
        "alternative2 if error-free system translation exists",
        "alternative3 if error-free system translation exists",
    ]:
        kwargs = reference_settings[setting.replace(" if error-free system translation exists", "")]
        metrics.append(MetricLoader(
            name=f"SentBLEU ({setting})",
            load_func=lambda lang, kwargs=kwargs: SegmentLevelMetric(
                SentBleu(trg_lang=lang, tokenize=None),
                "sent-bleu",
                wmt_data,
                references_loader=ReferencesLoader(wmt_data, **kwargs),
                mask_by_references_loader=ReferencesLoader(wmt_data, **reference_settings["error-free system translation"]),
                normalize_reference_scores=True,  # Use reference scores to weight segment average
            ),
        ))
    return metrics


def chrf_metrics(wmt_data, device=None) -> List[MetricLoader]:
    from sacrerouge.metrics import ChrF
    metrics = []
    for setting in [
        "error-free system translation",
    ]:
        kwargs = reference_settings[setting]
        metrics.append(MetricLoader(
            name=f"ChrF ({setting})",
            load_func=lambda lang, kwargs=kwargs: SegmentLevelMetric(
                ChrF(),
                "chrf",
                wmt_data,
                references_loader=ReferencesLoader(wmt_data, **kwargs),
                normalize_reference_scores=True,  # Use reference scores to weight segment average
            ),
        ))
    for setting in [
        "standard if error-free system translation exists",
        "alternative if error-free system translation exists",
        "alternative2 if error-free system translation exists",
        "alternative3 if error-free system translation exists",
    ]:
        kwargs = reference_settings[setting.replace(" if error-free system translation exists", "")]
        metrics.append(MetricLoader(
            name=f"ChrF ({setting})",
            load_func=lambda lang, kwargs=kwargs: SegmentLevelMetric(
                ChrF(),
                "chrf",
                wmt_data,
                references_loader=ReferencesLoader(wmt_data, **kwargs),
                mask_by_references_loader=ReferencesLoader(wmt_data, **reference_settings["error-free system translation"]),
                normalize_reference_scores=True,  # Use reference scores to weight segment average
            ),
        ))
    return metrics


def comet_metrics(wmt_data, device=None) -> List[MetricLoader]:
    from meta_evaluation.metrics.comet import CometMetric
    metrics = []
    for setting in [
        "error-free system translation",
    ]:
        kwargs = reference_settings[setting]
        metrics.append(MetricLoader(
            name=f"COMET (wmt21-comet-mqm) ({setting})",
            load_func=lambda lang, kwargs=kwargs: SegmentLevelMetric(
                CometMetric("wmt21-comet-mqm"),
                "comet",
                wmt_data,
                references_loader=ReferencesLoader(wmt_data, **kwargs),
                normalize_reference_scores=True,  # Use reference scores to weight segment average
            ),
        ))
    for setting in [
        "standard if error-free system translation exists",
    ]:
        kwargs = reference_settings[setting.replace(" if error-free system translation exists", "")]
        metrics.append(MetricLoader(
            name=f"COMET (wmt21-comet-mqm) ({setting})",
            load_func=lambda lang, kwargs=kwargs: SegmentLevelMetric(
                CometMetric("wmt21-comet-mqm"),
                "comet",
                wmt_data,
                references_loader=ReferencesLoader(wmt_data, **kwargs),
                mask_by_references_loader=ReferencesLoader(wmt_data, **reference_settings["error-free system translation"]),
                normalize_reference_scores=True,  # Use reference scores to weight segment average
            ),
        ))
    return metrics


def prism_metrics(wmt_data, device=None) -> List[MetricLoader]:
    from nmtscore import NMTScorer
    from meta_evaluation.metrics.prism import PrismMetric
    from meta_evaluation.metrics.prism_model import PrismModel

    SCORE_KWARGS = {
        "use_cache": True,
        "batch_size": 8,
    }

    all_metrics = []

    def get_metrics(setting_kwargs, mask_by_references_loader=None):
        kwargs = setting_kwargs
        metrics = []
        metrics.append(MetricLoader(
            name=f"Prism-ref ({setting})",
            load_func=lambda tgt_lang, device=device, kwargs=kwargs: SegmentLevelMetric(
                PrismMetric(
                    scorer=NMTScorer(device=device, model=PrismModel(device=device)),
                    summaries_lang=tgt_lang,
                    references_lang=tgt_lang,
                    documents_lang=None,
                    score_kwargs=SCORE_KWARGS,
                ),
                "prism",
                wmt_data,
                references_loader=ReferencesLoader(wmt_data, **kwargs),
                mask_by_references_loader=mask_by_references_loader,
                normalize_reference_scores=True,  # Use reference scores to weight segment average
            ),
        ))

        checkpoint_paths = []
        model_dir = Path(__file__).parent.parent.parent / "models"
        if not model_dir.exists():
            return metrics
        for subdir in model_dir.iterdir():
            if not subdir.is_dir():
                continue
            for checkpoint_path in subdir.iterdir():
                if not checkpoint_path.is_file():
                    continue
                if not checkpoint_path.name.startswith("checkpoint") or not checkpoint_path.name.endswith(".pt"):
                    continue
                checkpoint_paths.append(checkpoint_path)

        for checkpoint_path in checkpoint_paths:
            metrics.append(MetricLoader(
                name=f"Prism-{checkpoint_path.parent.name}_{checkpoint_path.name}-ref ({setting})",
                load_func=lambda tgt_lang, checkpoint_path=checkpoint_path, device=device, kwargs=kwargs: SegmentLevelMetric(
                    PrismMetric(
                        scorer=NMTScorer(device=device, model=PrismModel(
                            model_dir=checkpoint_path,
                            device=device)
                                         ),
                        summaries_lang=tgt_lang,
                        references_lang=tgt_lang,
                        documents_lang=None,
                        score_kwargs=SCORE_KWARGS,
                    ),
                    "prism",
                    wmt_data,
                    references_loader=ReferencesLoader(wmt_data, **kwargs),
                    mask_by_references_loader=mask_by_references_loader,
                    normalize_reference_scores=True,  # Use reference scores to weight segment average
                ),
            ))
        return metrics

    for setting in [
        "error-free system translation",
    ]:
        kwargs = reference_settings[setting]
        all_metrics += get_metrics(kwargs)

    for setting in [
        "standard if error-free system translation exists",
        "alternative if error-free system translation exists",
        "alternative2 if error-free system translation exists",
        "alternative3 if error-free system translation exists",
    ]:
        kwargs = reference_settings[setting.replace(" if error-free system translation exists", "")]
        all_metrics += get_metrics(kwargs, mask_by_references_loader=ReferencesLoader(wmt_data, **reference_settings["error-free system translation"]),)

    return all_metrics


def random_baseline_metrics(wmt_data, device=None) -> List[MetricLoader]:
    from post_editese.post_editese_metrics.random_baseline import RandomMetric
    metrics = []
    metrics.append(MetricLoader(
        name=f"Random baseline",
        load_func=lambda lang: SegmentLevelMetric(
            RandomMetric(),
            "random",
            wmt_data,
            references_loader=ReferencesLoader(wmt_data),
        ),
    ))
    return metrics

def all_metrics(wmt_data, device=None) -> Dict[str, MetricLoader]:
    metrics = dict()
    for metric in itertools.chain(
            sentbleu_metrics(wmt_data, device),
            chrf_metrics(wmt_data, device),
            comet_metrics(wmt_data, device),
            prism_metrics(wmt_data, device),
    ):
        metrics[metric.name] = metric
    return metrics
