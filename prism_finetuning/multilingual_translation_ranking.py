# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import logging
import os
from collections import OrderedDict

import torch

from fairseq import utils
from fairseq.data import RoundRobinZipDatasets, Dictionary
from fairseq.tasks import register_task
from fairseq.tasks.multilingual_translation import MultilingualTranslationTask

from prism_finetuning.contrastive_translation_dataset import load_contrastive_langpair_dataset

logger = logging.getLogger(__name__)


@register_task("multilingual_translation_ranking")
class MultilingualTranslationRankingTask(MultilingualTranslationTask):
    """
    Adapted from the parent task
    Remove MultiModel functionality of parent task - not needed for Prism

    Extensive changes to dataset loader in order to:
    - implement contrastive learning
    - replicate the import format expected by the pre-trained Prism model
    """

    @staticmethod
    def add_args(parser):
        MultilingualTranslationTask.add_args(parser)
        parser.add_argument('--train-ref-hyp', action='store_true', help='Train on samples in ref -> hyp direction')
        parser.add_argument('--train-hyp-ref', action='store_true', help='Train on samples in hyp -> ref direction')
        parser.add_argument('--train-src-hyp', action='store_true', help='Train on samples in src -> hyp direction')
        parser.add_argument('--train-hyp-src', action='store_true', help='Train on samples in hyp -> src direction')
        parser.add_argument('--train-cross-entropy', action='store_true',
                            help='Minimize cross-entropy on samples in src -> ref direction')
        parser.add_argument('--cross-entropy-weight', type=float, default=0.001,
                            help='Weight of cross-entropy loss term')

    def load_dataset(self, split, epoch=1, **kwargs):
        paths = utils.split_paths(self.args.data)
        assert len(paths) > 0
        data_path = paths[(epoch - 1) % len(paths)]

        def language_pair_dataset(lang_pair):
            src, tgt = lang_pair.split("-")
            langpair_dataset = load_contrastive_langpair_dataset(
                data_path,
                split,
                src,
                self.dicts[src],
                tgt,
                self.dicts[tgt],
                combine=True,
                dataset_impl=self.args.dataset_impl,
                upsample_primary=self.args.upsample_primary,
                left_pad_source=self.args.left_pad_source,
                left_pad_target=self.args.left_pad_target,
                max_source_positions=self.args.max_source_positions,
                max_target_positions=self.args.max_target_positions,
                prepend_tgt_lang_token=True,
                train_ref_hyp=self.args.train_ref_hyp,
                train_hyp_ref=self.args.train_hyp_ref,
                train_src_hyp=self.args.train_src_hyp,
                train_hyp_src=self.args.train_hyp_src,
                train_cross_entropy=self.args.train_cross_entropy,
            )
            return langpair_dataset

        dataset_dict = OrderedDict(
            [
                (lang_pair, language_pair_dataset(lang_pair))
                for lang_pair in self.lang_pairs
            ]
        )
        dataset_dict = {key: dataset for key, dataset in dataset_dict.items() if dataset is not None}

        self.datasets[split] = RoundRobinZipDatasets(
            dataset_dict,
            eval_key=None
            if self.training
            else "%s-%s" % (self.args.source_lang, self.args.target_lang),
        )

    def build_model(self, args, from_checkpoint=False):
        def check_args():
            messages = []
            if (
                    len(set(self.args.lang_pairs).symmetric_difference(args.lang_pairs))
                    != 0
            ):
                messages.append(
                    "--lang-pairs should include all the language pairs {}.".format(
                        args.lang_pairs
                    )
                )
            if self.args.encoder_langtok != args.encoder_langtok:
                messages.append(
                    "--encoder-langtok should be {}.".format(args.encoder_langtok)
                )
            if self.args.decoder_langtok != args.decoder_langtok:
                messages.append(
                    "--decoder-langtok should {} be set.".format(
                        "" if args.decoder_langtok else "not"
                    )
                )

            if len(messages) > 0:
                raise ValueError(" ".join(messages))

        # Update args -> the fact that the constructor here
        # changes the args object doesn't mean you get the same one here
        self.update_args(args)

        # Check if task args are consistant with model args
        check_args()

        from fairseq import models

        model = models.build_model(args, self, from_checkpoint)
        return model

    def max_positions(self):
        return (self.args.max_source_positions, self.args.max_target_positions)

    def _per_lang_pair_train_loss(
            self, lang_pair, model, update_num, criterion, sample, optimizer, ignore_grad
    ):
        # self._log_batch(lang_pair, sample)
        loss, sample_size, logging_output = criterion(model, sample[lang_pair], update_num)
        if ignore_grad:
            loss *= 0
        optimizer.backward(loss)
        return loss, sample_size, logging_output

    def _per_lang_pair_valid_loss(self, lang_pair, model, criterion, sample):
        return criterion(model, sample[lang_pair])

    def _log_batch(self, lang_pair, sample):

        def decode(token_ids):
            return self.target_dictionary.string(token_ids.tolist()).split()

        sample = sample[lang_pair]
        print(f"Language pair: {lang_pair}")
        sample_size = sample["target_pos"].size(0)
        print(f"Batch size: {sample_size}")
        print()
        for i in range(sample_size):
            print(f'src_pos: {decode(sample["net_input_pos"]["src_tokens"][i])}')
            print(f'src_neg: {decode(sample["net_input_neg"]["src_tokens"][i])}')
            print(f'tgt_pos: {decode(sample["target_pos"][i])}')
            print(f'tgt_neg: {decode(sample["target_neg"][i])}')
            if self.args.train_cross_entropy:
                print(f'ce_src: {decode(sample["net_input_ce"]["src_tokens"][i])}')
                print(f'ce_tgt: {decode(sample["target_ce"][i])}')
            print()
            # Assert that pos and neg differ on exactly one side
            assert torch.equal(sample["net_input_pos"]["src_tokens"][i], sample["net_input_neg"]["src_tokens"][i]) != \
                   torch.equal(sample["target_pos"][i], sample["target_neg"][i])
            # Assert that there is a target language ID and it is the same for pos and neg
            assert sample["target_pos"][i][0] == sample["target_neg"][i][0]
            target_lang_id = decode(sample["target_pos"][i])[0]
            src_lang = lang_pair.split('-')[0]
            tgt_lang = lang_pair.split('-')[1]
            assert target_lang_id in {f"<{src_lang}>", f"<{tgt_lang}>"}
            # Assert that in the reconstruction direction, the contrastive variant is always in the (original) target language
            if target_lang_id == f"<{src_lang}>":
                assert torch.equal(sample["target_pos"][i], sample["target_neg"][i])

    @staticmethod
    def _lang_token(lang: str) -> str:
        return f"<{lang}>"

    @staticmethod
    def _lang_token_index(dic: Dictionary, lang: str):
        idx = dic.index(MultilingualTranslationRankingTask._lang_token(lang))
        assert idx != dic.unk_index, "cannot find language token for lang {}".format(lang)
        return idx

    def get_encoder_langtok(self, src_lang, tgt_lang):
        if self.args.encoder_langtok is None:
            return self.dicts[src_lang].eos()
        if self.args.encoder_langtok == "src":
            return self._lang_token_index(self.dicts[src_lang], src_lang)
        else:
            return self._lang_token_index(self.dicts[src_lang], tgt_lang)

    def get_decoder_langtok(self, tgt_lang):
        if not self.args.decoder_langtok:
            return self.dicts[tgt_lang].eos()
        return self._lang_token_index(self.dicts[tgt_lang], tgt_lang)

    @classmethod
    def prepare(cls, args, **kargs):
        cls.update_args(args)
        sorted_langs = sorted(
            list({x for lang_pair in args.lang_pairs for x in lang_pair.split("-")})
        )
        if args.source_lang is not None or args.target_lang is not None:
            training = False
        else:
            training = True

        # load dictionaries
        dicts = OrderedDict()
        for lang in sorted_langs:
            paths = utils.split_paths(args.data)
            assert len(paths) > 0
            dicts[lang] = cls.load_dictionary(
                os.path.join(paths[0], "dict.{}.txt".format(lang))
            )
            if len(dicts) > 0:
                assert dicts[lang].pad() == dicts[sorted_langs[0]].pad()
                assert dicts[lang].eos() == dicts[sorted_langs[0]].eos()
                assert dicts[lang].unk() == dicts[sorted_langs[0]].unk()
            if args.encoder_langtok is not None or args.decoder_langtok:
                for lang_to_add in sorted_langs:
                    dicts[lang].add_symbol(cls._lang_token(lang_to_add))
            logger.info("[{}] dictionary: {} types".format(lang, len(dicts[lang])))
        return dicts, training

    def inference_step(
            self, generator, models, sample, prefix_tokens=None, constraints=None
    ):
        raise NotImplementedError
