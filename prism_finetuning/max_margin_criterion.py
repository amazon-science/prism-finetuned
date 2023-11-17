# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import math
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from dataclasses import dataclass

from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from fairseq.logging import metrics


@dataclass
class MaxMarginCriterionConfig(FairseqDataclass):
    ...


@register_criterion("max_margin", dataclass=MaxMarginCriterionConfig)
class MaxMarginCriterion(FairseqCriterion):

    @classmethod
    def add_args(cls, parser):
        FairseqCriterion.add_args(parser)
        parser.add_argument('--margin', type=float, default=0.1)
        parser.add_argument('--length-normalize-scores', action='store_true')
        parser.add_argument('--fix-alternate', action='store_true')
        parser.add_argument('--normalize-scores', action='store_true')

    def forward(self, model, sample, update_number: int = None, reduce=True):
        with torch.no_grad() if (self.task.args.fix_alternate and update_number is not None and update_number % 2 == 0) else nullcontext():
            net_output_pos = model(**sample["net_input_pos"])
        with torch.no_grad() if (self.task.args.fix_alternate and update_number is not None and update_number % 2 == 1) else nullcontext():
            net_output_neg = model(**sample["net_input_neg"])
        (
            loss, _,
            score_sum,
            num_correct,
        ) = self.compute_loss(model, net_output_pos, net_output_neg, sample, update_number=update_number, reduce=reduce)

        if self.task.args.train_cross_entropy:
            net_output = model(**sample["net_input_ce"])
            cross_entropy_loss, _ = self.compute_cross_entropy_loss(model, net_output, sample, reduce=reduce)
            cross_entropy_loss *= self.task.args.cross_entropy_weight
            loss += cross_entropy_loss
        else:
            cross_entropy_loss = None

        sample_size = sample["target_pos"].size(0)
        logging_output = {
            "loss": loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target_pos"].size(0),
            "sample_size": sample_size,
            "score_sum": score_sum,
            "num_correct": num_correct,
        }
        if cross_entropy_loss is not None:
            logging_output["cross_entropy_loss"] = cross_entropy_loss.data
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output_pos, net_output_neg, sample, update_number: int = None, reduce=True):
        if not reduce:
            raise NotImplementedError

        lprobs_pos = model.get_normalized_probs(net_output_pos, log_probs=True)

        # Do not include target language ID in loss
        sample["target_pos"][:, 0] = self.padding_idx
        sample["target_neg"][:, 0] = self.padding_idx

        token_scores = (-1) * F.nll_loss(
            lprobs_pos.view(-1, lprobs_pos.size(-1)),
            sample["target_pos"].view(-1),
            ignore_index=self.padding_idx,
            reduction="none",
        ).view_as(sample["target_pos"])
        score_pos = token_scores.sum(dim=-1)

        lprobs_neg = model.get_normalized_probs(net_output_neg, log_probs=True)
        token_scores = (-1) * F.nll_loss(
            lprobs_neg.view(-1, lprobs_neg.size(-1)),
            sample["target_neg"].view(-1),
            ignore_index=self.padding_idx,
            reduction="none",
        ).view_as(sample["target_neg"])
        score_neg = token_scores.sum(dim=-1)

        if self.task.args.length_normalize_scores:
            num_pos_tokens = (~sample["target_pos"].eq(self.padding_idx)).count_nonzero(dim=-1)
            num_neg_tokens = (~sample["target_neg"].eq(self.padding_idx)).count_nonzero(dim=-1)
            score_pos /= num_pos_tokens
            score_neg /= num_neg_tokens

        score_sum = score_pos.detach().sum() + score_neg.detach().sum()
        score_pos = torch.exp(score_pos)
        score_neg = torch.exp(score_neg)

        if self.task.args.normalize_scores:
            score_pos /= (score_pos + score_neg).detach()
            score_neg /= (score_pos + score_neg).detach()

        if self.task.args.fix_alternate and update_number is not None:
            if update_number % 2 == 0:
                score_pos = score_pos.detach()
            if update_number % 2 == 1:
                score_neg = score_neg.detach()

        loss = torch.nn.MarginRankingLoss(
            margin=self.task.args.margin,
            reduction="none",
        )(
            score_pos,
            score_neg,
            target=torch.ones_like(score_pos),
        )

        correctly_ranked_samples = torch.greater_equal(score_pos.detach(), score_neg.detach())
        num_correct = correctly_ranked_samples.count_nonzero()
        loss = loss.sum()
        loss /= 2  # One direction of two
        return loss, loss, score_sum, num_correct

    def compute_cross_entropy_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = sample["target_ce"]

        # Ignore first token:
        lprobs = lprobs[:, 1:, :]
        target = target[:, 1:]

        lprobs = lprobs.reshape(-1, lprobs.size(-1))
        target = target.reshape(-1)
        if self.task.args.length_normalize_scores:
            reduction = "mean"
        else:
            reduction = "sum" if reduce else "none"
        loss = F.nll_loss(
            lprobs,
            target,
            ignore_index=self.padding_idx,
            reduction=reduction,
        )
        return loss, loss

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        metrics.log_scalar(
            "loss", loss_sum / sample_size, sample_size, round=10
        )

        score_sum = sum(log.get("score_sum", 0) for log in logging_outputs)
        metrics.log_scalar(
            "avg_score", 2 ** (score_sum / 2 / sample_size), sample_size, round=10
        )

        num_correct = sum(log.get("num_correct", 0) for log in logging_outputs)
        metrics.log_scalar(
            "accuracy", num_correct / sample_size, sample_size, round=10
        )

        if "cross_entropy_loss" in logging_outputs[0]:
            cross_entropy_loss = sum(log.get("cross_entropy_loss", 0) for log in logging_outputs)
            metrics.log_scalar(
                "cross_entropy_loss", cross_entropy_loss / sample_size / math.log(2), sample_size, round=10
            )
