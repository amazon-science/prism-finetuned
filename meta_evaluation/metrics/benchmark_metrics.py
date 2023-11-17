# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import itertools
import logging
from pathlib import Path
from typing import List, Dict

from meta_evaluation.metrics import MetricLoader, SacreRougeMetric

logging.basicConfig(level=logging.INFO)


def sentbleu_metrics(device=None) -> List[MetricLoader]:
    from sacrerouge.metrics import SentBleu
    metrics = [MetricLoader(
        name=f"SentBLEU",
        load_func=lambda src_lang, tgt_lang: SacreRougeMetric(
            SentBleu(trg_lang=tgt_lang, tokenize=None),
            "sent-bleu",
        ),
    )]
    return metrics


def chrf_metrics(device=None) -> List[MetricLoader]:
    from sacrerouge.metrics import ChrF
    metrics = [MetricLoader(
        name=f"ChrF",
        load_func=lambda src_lang, tgt_lang: SacreRougeMetric(
            ChrF(),
            "chrf",
        ),
    )]
    return metrics


def prism_metrics(device=None) -> List[MetricLoader]:
    from nmtscore import NMTScorer
    from meta_evaluation.metrics.prism import PrismMetric
    from meta_evaluation.metrics.prism_model import PrismModel

    SCORE_KWARGS = {
        "use_cache": True,
        "batch_size": 8,
    }

    metrics = []
    for name, kwargs in (
        ("ref", {"ref_hyp": True, "hyp_ref": True, "src_hyp": False, "hyp_src": False, "normalize": False}),
    ):
        metrics.append(
            MetricLoader(
                name=f"Prism-{name}",
                load_func=lambda src_lang, tgt_lang, device=device, kwargs=kwargs: SacreRougeMetric(
                    PrismMetric(
                        scorer=NMTScorer(device=device, model=PrismModel(device=device)),
                        summaries_lang=tgt_lang,
                        references_lang=tgt_lang,
                        documents_lang=src_lang,
                        score_kwargs=SCORE_KWARGS,
                        **kwargs

                    ),
                    "prism",
                ),
            ),
        )

        checkpoint_paths = []
        model_dir = Path(__file__).parent.parent.parent / "models"
        if not model_dir.exists():
            continue
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
            metrics.append(
                MetricLoader(
                    name=f"Prism-{checkpoint_path.parent.name}_{checkpoint_path.name}-{name}",
                    load_func=lambda src_lang, tgt_lang, checkpoint_path=checkpoint_path, device=device, kwargs=kwargs: SacreRougeMetric(
                        PrismMetric(
                            scorer=NMTScorer(device=device, model=PrismModel(
                                model_dir=checkpoint_path,
                                device=device)
                            ),
                            summaries_lang=tgt_lang,
                            references_lang=tgt_lang,
                            documents_lang=src_lang,
                            score_kwargs=SCORE_KWARGS,
                            **kwargs

                        ),
                        "prism",
                    ),
                ),
            )
    return metrics


def comet_metrics(device=None) -> List[MetricLoader]:
    from meta_evaluation.metrics.comet import CometMetric
    metrics = []
    for pretrained_model in [
        "wmt21-comet-mqm",
    ]:
        metrics.append(
            MetricLoader(
                name=f"COMET ({pretrained_model})",
                load_func=lambda src_lang, tgt_lang, device=device, pretrained_model=pretrained_model: SacreRougeMetric(
                    CometMetric(pretrained_model),
                    "comet",
                ),
            ),
        )
    return metrics


def bertscore_metrics(device=None) -> List[MetricLoader]:
    from sacrerouge.metrics import BertScore
    metrics = [MetricLoader(
        name=f"BERTScore",
        load_func=lambda src_lang, tgt_lang, device=device: SacreRougeMetric(
            BertScore(
                lang=tgt_lang,
                device=device,
            ),
            "bertscore_f1",
        ),
    )]
    return metrics


def cached_metrics(device=None) -> List[MetricLoader]:
    metrics = []
    for cached_name in [
        "Prism-paper_ende_checkpoint1.pt-ref",
        "Prism-paper_zhen_checkpoint1.pt-ref",
        "Prism-paper_main_checkpoint1.pt-ref",
        "Prism-paper_ablation_with_fix_checkpoint1.pt-ref",
        "Prism-paper_ablation_no_cross_entropy_checkpoint1.pt-ref",
        "Prism-paper_ablation_no_forward_checkpoint1.pt-ref",
        "Prism-paper_ablation_no_backward_checkpoint1.pt-ref",
    ]:
        metrics.append(MetricLoader(
            name=cached_name,
            load_func=lambda src_lang, tgt_lang, device=device: None,
        ))
    return metrics


def all_metrics(device=None) -> Dict[str, MetricLoader]:
    metrics = dict()
    for metric in itertools.chain(
        cached_metrics(device),
        sentbleu_metrics(device),
        chrf_metrics(device),
        comet_metrics(device),
        prism_metrics(device),
    ):
        metrics[metric.name] = metric
    return metrics
