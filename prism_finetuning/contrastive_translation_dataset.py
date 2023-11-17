# Original Copyright (c) Facebook, Inc. and its affiliates. MIT License.
# Modifications Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

from pathlib import Path

import numpy as np
import torch
from fairseq.data import data_utils, LanguagePairDataset, ConcatDataset, PrependTokenDataset


def collate(
        samples,
        pad_idx,
        eos_idx,
        left_pad_source=True,
        left_pad_target=False,
        input_feeding=True,
        pad_to_length=None,
        pad_to_multiple=1,
):
    """
    Adapted from fairseq.data.language_pair_dataset.collate
    # Copyright (c) Facebook, Inc. and its affiliates.
    """
    if len(samples) == 0:
        return {}

    # Pad pos and neg to same length
    if samples[0].get("source_ce", None) is not None:
        if pad_to_length is None:
            source_pad_to_length = max(
                max(v.size(0) for v in [s["source_pos"] for s in samples]),
                max(v.size(0) for v in [s["source_neg"] for s in samples]),
                max(v.size(0) for v in [s["source_ce"] for s in samples]),
            )
            target_pad_to_length = max(
                max(v.size(0) for v in [s["target_pos"] for s in samples]),
                max(v.size(0) for v in [s["target_neg"] for s in samples]),
                max(v.size(0) for v in [s["target_ce"] for s in samples]),
            )
        else:
            source_pad_to_length = max(pad_to_length["source_pos"], pad_to_length["source_neg"], pad_to_length["source_ce"])
            target_pad_to_length = max(pad_to_length["target_pos"], pad_to_length["target_neg"], pad_to_length["target_ce"])
    else:
        if pad_to_length is None:
            source_pad_to_length = max(
                max(v.size(0) for v in [s["source_pos"] for s in samples]),
                max(v.size(0) for v in [s["source_neg"] for s in samples]),
            )
            target_pad_to_length = max(
                max(v.size(0) for v in [s["target_pos"] for s in samples]),
                max(v.size(0) for v in [s["target_neg"] for s in samples]),
            )
        else:
            source_pad_to_length = max(pad_to_length["source_pos"], pad_to_length["source_neg"])
            target_pad_to_length = max(pad_to_length["target_pos"], pad_to_length["target_neg"])

    def merge(key, left_pad, move_eos_to_beginning=False, pad_to_length=None):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx,
            eos_idx,
            left_pad,
            move_eos_to_beginning,
            pad_to_length=pad_to_length,
            pad_to_multiple=pad_to_multiple,
        )

    id = torch.LongTensor([s["id"] for s in samples])
    src_pos_tokens = merge(
        "source_pos",
        left_pad=left_pad_source,
        pad_to_length=source_pad_to_length,
    )
    # sort by descending source length
    src_pos_lengths = torch.LongTensor(
        [s["source_pos"].ne(pad_idx).long().sum() for s in samples]
    )
    src_pos_lengths, sort_order = src_pos_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)
    src_pos_tokens = src_pos_tokens.index_select(0, sort_order)

    src_neg_tokens = merge(
        "source_neg",
        left_pad=left_pad_source,
        pad_to_length=source_pad_to_length,
    )
    src_neg_lengths = torch.LongTensor(
        [s["source_neg"].ne(pad_idx).long().sum() for s in samples]
    )
    src_neg_tokens = src_neg_tokens.index_select(0, sort_order)

    if samples[0].get("source_ce", None) is not None:
        src_ce_tokens = merge(
            "source_ce",
            left_pad=left_pad_source,
            pad_to_length=source_pad_to_length,
        )
        src_ce_lengths = torch.LongTensor(
            [s["source_ce"].ne(pad_idx).long().sum() for s in samples]
        )
        src_ce_tokens = src_ce_tokens.index_select(0, sort_order)
    else:
        src_ce_tokens = None
        src_ce_lengths = None

    prev_output_tokens_pos = None
    target_pos = None
    if samples[0].get("target_pos", None) is not None:
        target_pos = merge(
            "target_pos",
            left_pad=left_pad_target,
            pad_to_length=target_pad_to_length,
        )
        target_pos = target_pos.index_select(0, sort_order)
        tgt_lengths = torch.LongTensor(
            [s["target_pos"].ne(pad_idx).long().sum() for s in samples]
        ).index_select(0, sort_order)
        ntokens = tgt_lengths.sum().item()

        if samples[0].get("prev_output_tokens_pos", None) is not None:
            prev_output_tokens_pos = merge("prev_output_tokens_pos", left_pad=left_pad_target)
        elif input_feeding:
            # we create a shifted version of targets for feeding the
            # previous output token(s) into the next decoder step
            prev_output_tokens_pos = merge(
                "target_pos",
                left_pad=left_pad_target,
                move_eos_to_beginning=True,
                pad_to_length=target_pad_to_length,
            )
    else:
        ntokens = src_pos_lengths.sum().item()

    prev_output_tokens_neg = None
    target_neg = None
    if samples[0].get("target_neg", None) is not None:
        target_neg = merge(
            "target_neg",
            left_pad=left_pad_target,
            pad_to_length=target_pad_to_length,
        )
        target_neg = target_neg.index_select(0, sort_order)
        tgt_lengths = torch.LongTensor(
            [s["target_neg"].ne(pad_idx).long().sum() for s in samples]
        ).index_select(0, sort_order)
        ntokens = tgt_lengths.sum().item()

        if samples[0].get("prev_output_tokens_neg", None) is not None:
            prev_output_tokens_neg = merge("prev_output_tokens_neg", left_pad=left_pad_target)
        elif input_feeding:
            # we create a shifted version of targets for feeding the
            # previous output token(s) into the next decoder step
            prev_output_tokens_neg = merge(
                "target_neg",
                left_pad=left_pad_target,
                move_eos_to_beginning=True,
                pad_to_length=target_pad_to_length,
            )

    prev_output_tokens_ce = None
    target_ce = None
    if samples[0].get("target_ce", None) is not None:
        target_ce = merge(
            "target_ce",
            left_pad=left_pad_target,
            pad_to_length=target_pad_to_length,
        )
        target_ce = target_ce.index_select(0, sort_order)
        tgt_lengths = torch.LongTensor(
            [s["target_ce"].ne(pad_idx).long().sum() for s in samples]
        ).index_select(0, sort_order)
        ntokens = tgt_lengths.sum().item()

        if samples[0].get("prev_output_tokens_ce", None) is not None:
            prev_output_tokens_ce = merge("prev_output_tokens_ce", left_pad=left_pad_target)
        elif input_feeding:
            # we create a shifted version of targets for feeding the
            # previous output token(s) into the next decoder step
            prev_output_tokens_ce = merge(
                "target_ce",
                left_pad=left_pad_target,
                move_eos_to_beginning=True,
                pad_to_length=target_pad_to_length,
            )

    batch = {
        "id": id,
        "nsentences": len(samples),
        "ntokens": ntokens,
        "net_input_pos": {
            "src_tokens": src_pos_tokens,
            "src_lengths": src_pos_lengths,
        },
        "net_input_neg": {
            "src_tokens": src_neg_tokens,
            "src_lengths": src_neg_lengths,
        },
        "target_pos": target_pos,
        "target_neg": target_neg,
    }
    if src_ce_tokens is not None:
        batch["net_input_ce"] = {
            "src_tokens": src_ce_tokens,
            "src_lengths": src_ce_lengths,
        }
        batch["target_ce"] = target_ce

    if prev_output_tokens_pos is not None:
        batch["net_input_pos"]["prev_output_tokens"] = prev_output_tokens_pos.index_select(
            0, sort_order
        )
    if prev_output_tokens_neg is not None:
        batch["net_input_neg"]["prev_output_tokens"] = prev_output_tokens_neg.index_select(
            0, sort_order
        )
    if prev_output_tokens_ce is not None:
        batch["net_input_ce"]["prev_output_tokens"] = prev_output_tokens_ce.index_select(
            0, sort_order
        )

    if samples[0].get("alignment", None) is not None:
        raise NotImplementedError

    if samples[0].get("constraints", None) is not None:
        raise NotImplementedError

    return batch


class ContrastiveLanguagePairDataset(LanguagePairDataset):
    """
    Adapted from fairseq.data.language_pair_dataset.LanguagePairDataset
    # Copyright (c) Facebook, Inc. and its affiliates.
    """

    def __init__(
            self,
            src_pos,
            src_pos_sizes,
            src_neg,
            src_neg_sizes,
            src_dict,
            src_ce=None,
            src_ce_sizes=None,
            tgt_pos=None,
            tgt_pos_sizes=None,
            tgt_neg=None,
            tgt_neg_sizes=None,
            tgt_ce=None,
            tgt_ce_sizes=None,
            tgt_dict=None,
            left_pad_source=True,
            left_pad_target=False,
            shuffle=True,
            input_feeding=True,
            remove_eos_from_source=False,
            append_eos_to_target=False,
            align_dataset=None,
            constraints=None,
            append_bos=False,
            eos=None,
            num_buckets=0,
            src_lang_id=None,
            tgt_lang_id=None,
            pad_to_multiple=1,
    ):
        super().__init__(src_pos, src_pos_sizes, src_dict, tgt_pos, tgt_pos_sizes, tgt_dict, left_pad_source, left_pad_target, shuffle,
                         input_feeding, remove_eos_from_source, append_eos_to_target, align_dataset, constraints,
                         append_bos, eos, num_buckets, src_lang_id, tgt_lang_id, pad_to_multiple)
        self.src_neg = src_neg
        self.src_neg_sizes = np.array(src_neg_sizes) if src_neg_sizes is not None else None
        self.src_ce = src_ce
        self.src_ce_sizes = np.array(src_ce_sizes) if src_ce_sizes is not None else None
        self.tgt_neg = tgt_neg
        self.tgt_neg_sizes = np.array(tgt_neg_sizes) if tgt_neg_sizes is not None else None
        self.tgt_ce = tgt_ce
        self.tgt_ce_sizes = np.array(tgt_ce_sizes) if tgt_ce_sizes is not None else None
        self.sizes = np.vstack((self.src_sizes, self.src_neg_sizes, self.tgt_sizes, self.tgt_neg_sizes)).T

    def __getitem__(self, index):
        src_pos_item = self.src[index]
        src_neg_item = self.src_neg[index] if self.src_neg is not None else None
        src_ce_item = self.src_ce[index] if self.src_ce is not None else None
        tgt_pos_item = self.tgt[index] if self.tgt is not None else None
        tgt_neg_item = self.tgt_neg[index] if self.tgt_neg is not None else None
        tgt_ce_item = self.tgt_ce[index] if self.tgt_ce is not None else None
        # Append EOS to end of tgt sentence if it does not have an EOS and remove
        # EOS from end of src sentence if it exists. This is useful when we use
        # use existing datasets for opposite directions i.e., when we want to
        # use tgt_dataset as src_dataset and vice versa
        if self.append_eos_to_target:
            raise NotImplementedError
        if self.append_bos:
            raise NotImplementedError
        if self.remove_eos_from_source:
            raise NotImplementedError

        example = {
            "id": index,
            "source_pos": src_pos_item,
            "source_neg": src_neg_item,
            "target_pos": tgt_pos_item,
            "target_neg": tgt_neg_item,
        }

        if src_ce_item is not None:
            assert tgt_ce_item is not None
            example["source_ce"] = src_ce_item
            example["target_ce"] = tgt_ce_item

        if self.align_dataset is not None:
            raise NotImplementedError
        if self.constraints is not None:
            raise NotImplementedError
        return example

    def collater(self, samples, pad_to_length=None):
        res = collate(
            samples,
            pad_idx=self.src_dict.pad(),
            eos_idx=self.eos,
            left_pad_source=self.left_pad_source,
            left_pad_target=self.left_pad_target,
            input_feeding=self.input_feeding,
            pad_to_length=pad_to_length,
            pad_to_multiple=self.pad_to_multiple,
        )
        if self.src_lang_id is not None or self.tgt_lang_id is not None:
            src_tokens = res["net_input"]["src_tokens"]
            bsz = src_tokens.size(0)
            if self.src_lang_id is not None:
                res["net_input"]["src_lang_id"] = (
                    torch.LongTensor([[self.src_lang_id]]).expand(bsz, 1).to(src_tokens)
                )
            if self.tgt_lang_id is not None:
                res["tgt_lang_id"] = (
                    torch.LongTensor([[self.tgt_lang_id]]).expand(bsz, 1).to(src_tokens)
                )
        return res

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return max(
            self.src_sizes[index],
            self.src_neg_sizes[index],
            self.src_ce_sizes[index] if self.src_ce_sizes is not None else 0,
            self.tgt_sizes[index] if self.tgt_sizes is not None else 0,
            self.tgt_neg_sizes[index] if self.tgt_neg_sizes is not None else 0,
            self.tgt_ce_sizes[index] if self.tgt_ce_sizes is not None else 0,
        )

    def num_tokens_vec(self, indices):
        """Return the number of tokens for a set of positions defined by indices.
        This value is used to enforce ``--max-tokens`` during batching."""
        sizes = np.maximum(self.src_sizes[indices], self.src_neg_sizes[indices])
        if self.tgt_sizes is not None:
            sizes = np.maximum(sizes, self.tgt_sizes[indices])
        if self.tgt_neg_sizes is not None:
            sizes = np.maximum(sizes, self.tgt_neg_sizes[indices])
        if self.src_ce_sizes is not None:
            sizes = np.maximum(sizes, self.src_ce_sizes[indices])
        if self.tgt_ce_sizes is not None:
            sizes = np.maximum(sizes, self.tgt_ce_sizes[indices])
        return sizes

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return (
            self.src_sizes[index],
            self.src_neg_sizes[index],
            self.src_ce_sizes[index] if self.src_ce_sizes is not None else 0,
            self.tgt_sizes[index] if self.tgt_sizes is not None else 0,
            self.tgt_neg_sizes[index] if self.neg_sizes is not None else 0,
            self.tgt_ce_sizes[index] if self.tgt_ce_sizes is not None else 0,
        )

    def prefetch(self, indices):
        self.src.prefetch(indices)
        self.src_neg.prefetch(indices)
        if self.tgt is not None:
            self.tgt.prefetch(indices)
        if self.tgt_neg is not None:
            self.tgt_neg.prefetch(indices)
        if self.src_ce is not None:
            self.src_ce.prefetch(indices)
        if self.tgt_ce is not None:
            self.tgt_ce.prefetch(indices)
        if self.align_dataset is not None:
            self.align_dataset.prefetch(indices)


def load_contrastive_langpair_dataset(
        data_path,
        split,
        src,
        src_dict,
        tgt,
        tgt_dict,
        combine,
        dataset_impl,
        upsample_primary,
        left_pad_source,
        left_pad_target,
        max_source_positions,
        max_target_positions,
        prepend_bos=False,
        load_alignments=False,
        truncate_source=False,
        append_source_id=False,
        num_buckets=0,
        shuffle=True,
        pad_to_multiple=1,
        prepend_bos_src=None,
        prepend_tgt_lang_token=False,
        train_ref_hyp=True,
        train_hyp_ref=True,
        train_src_hyp=True,
        train_hyp_src=True,
        train_cross_entropy=False,
):
    """
    Adapted from fairseq.tasks.translation.load_langpair_dataset
    # Copyright (c) Facebook, Inc. and its affiliates.
    """
    data_path = Path(data_path)
    assert data_path.exists()

    if truncate_source:
        raise NotImplementedError
    if upsample_primary != 1:
        raise NotImplementedError
    if prepend_bos:
        raise NotImplementedError
    if prepend_bos_src:
        raise NotImplementedError
    if append_source_id:
        raise NotImplementedError
    if load_alignments:
        raise NotImplementedError

    forward_pos_src_paths = []
    forward_neg_src_paths = []
    forward_pos_tgt_paths = []
    forward_neg_tgt_paths = []
    backward_pos_src_paths = []
    backward_neg_src_paths = []
    backward_pos_tgt_paths = []
    backward_neg_tgt_paths = []

    ce_src_paths = []
    ce_tgt_paths = []

    if train_ref_hyp:
        forward_pos_src_paths.append(data_path / f"{split}.{src}-{tgt}.ref-hyp.ref.pos.{tgt}.bpe")
        forward_pos_tgt_paths.append(data_path / f"{split}.{src}-{tgt}.ref-hyp.hyp.pos.{tgt}.bpe")
        forward_neg_src_paths.append(data_path / f"{split}.{src}-{tgt}.ref-hyp.ref.neg.{tgt}.bpe")
        forward_neg_tgt_paths.append(data_path / f"{split}.{src}-{tgt}.ref-hyp.hyp.neg.{tgt}.bpe")
    if train_hyp_ref:
        forward_pos_src_paths.append(data_path / f"{split}.{src}-{tgt}.hyp-ref.hyp.pos.{tgt}.bpe")
        forward_pos_tgt_paths.append(data_path / f"{split}.{src}-{tgt}.hyp-ref.ref.pos.{tgt}.bpe")
        forward_neg_src_paths.append(data_path / f"{split}.{src}-{tgt}.hyp-ref.hyp.neg.{tgt}.bpe")
        forward_neg_tgt_paths.append(data_path / f"{split}.{src}-{tgt}.hyp-ref.ref.neg.{tgt}.bpe")
    if train_src_hyp:
        forward_pos_src_paths.append(data_path / f"{split}.{src}-{tgt}.src-hyp.src.pos.{src}.bpe")
        forward_pos_tgt_paths.append(data_path / f"{split}.{src}-{tgt}.src-hyp.hyp.pos.{tgt}.bpe")
        forward_neg_src_paths.append(data_path / f"{split}.{src}-{tgt}.src-hyp.src.neg.{src}.bpe")
        forward_neg_tgt_paths.append(data_path / f"{split}.{src}-{tgt}.src-hyp.hyp.neg.{tgt}.bpe")
    if train_hyp_src:
        backward_pos_src_paths.append(data_path / f"{split}.{src}-{tgt}.hyp-src.hyp.pos.{tgt}.bpe")
        backward_pos_tgt_paths.append(data_path / f"{split}.{src}-{tgt}.hyp-src.src.pos.{src}.bpe")
        backward_neg_src_paths.append(data_path / f"{split}.{src}-{tgt}.hyp-src.hyp.neg.{tgt}.bpe")
        backward_neg_tgt_paths.append(data_path / f"{split}.{src}-{tgt}.hyp-src.src.neg.{src}.bpe")

    if train_cross_entropy:
        for _ in range(sum((train_ref_hyp, train_hyp_ref, train_src_hyp, train_hyp_src))):
            ce_src_paths.append(data_path / f"{split}.{src}-{tgt}.src-hyp.src.pos.{src}.bpe")
            ce_tgt_paths.append(data_path / f"{split}.{src}-{tgt}.ref-hyp.ref.pos.{tgt}.bpe")

    if not (forward_pos_src_paths[0].parent / (forward_pos_src_paths[0].name + ".bin")).exists():
        assert "valid"
        return None

    forward_pos_src_datasets = []
    forward_neg_src_datasets = []
    forward_pos_tgt_datasets = []
    forward_neg_tgt_datasets = []
    backward_pos_src_datasets = []
    backward_neg_src_datasets = []
    backward_pos_tgt_datasets = []
    backward_neg_tgt_datasets = []

    ce_src_datasets = []
    ce_tgt_datasets = []

    for path in forward_pos_src_paths:
        dataset = data_utils.load_indexed_dataset(str(path), src_dict, dataset_impl)
        assert dataset is not None or split == "valid", str(path)
        forward_pos_src_datasets.append(dataset)
    for path in forward_neg_src_paths:
        dataset = data_utils.load_indexed_dataset(str(path), src_dict, dataset_impl)
        assert dataset is not None or split == "valid", str(path)
        forward_neg_src_datasets.append(dataset)
    for path in forward_pos_tgt_paths:
        dataset = data_utils.load_indexed_dataset(str(path), tgt_dict, dataset_impl)
        assert dataset is not None or split == "valid", str(path)
        forward_pos_tgt_datasets.append(dataset)
    for path in forward_neg_tgt_paths:
        dataset = data_utils.load_indexed_dataset(str(path), tgt_dict, dataset_impl)
        assert dataset is not None or split == "valid", str(path)
        forward_neg_tgt_datasets.append(dataset)
    for path in backward_pos_src_paths:
        dataset = data_utils.load_indexed_dataset(str(path), tgt_dict, dataset_impl)
        assert dataset is not None or split == "valid", str(path)
        backward_pos_src_datasets.append(dataset)
    for path in backward_neg_src_paths:
        dataset = data_utils.load_indexed_dataset(str(path), tgt_dict, dataset_impl)
        assert dataset is not None or split == "valid", str(path)
        backward_neg_src_datasets.append(dataset)
    for path in backward_pos_tgt_paths:
        dataset = data_utils.load_indexed_dataset(str(path), src_dict, dataset_impl)
        assert dataset is not None or split == "valid", str(path)
        backward_pos_tgt_datasets.append(dataset)
    for path in backward_neg_tgt_paths:
        dataset = data_utils.load_indexed_dataset(str(path), src_dict, dataset_impl)
        assert dataset is not None or split == "valid", str(path)
        backward_neg_tgt_datasets.append(dataset)

    for path in ce_src_paths:
        dataset = data_utils.load_indexed_dataset(str(path), src_dict, dataset_impl)
        assert dataset is not None or split == "valid", str(path)
        ce_src_datasets.append(dataset)
    for path in ce_tgt_paths:
        dataset = data_utils.load_indexed_dataset(str(path), tgt_dict, dataset_impl)
        assert dataset is not None or split == "valid", str(path)
        ce_tgt_datasets.append(dataset)

    assert len(forward_pos_src_datasets) == len(forward_neg_src_datasets) == len(forward_pos_tgt_datasets) == len(forward_neg_tgt_datasets)
    assert len(backward_pos_src_datasets) == len(backward_neg_src_datasets) == len(backward_pos_tgt_datasets) == len(backward_neg_tgt_datasets)
    assert len(ce_src_datasets) == len(ce_tgt_datasets)
    if not train_hyp_src:
        assert not backward_pos_src_datasets

    forward_pos_src_dataset = ConcatDataset(forward_pos_src_datasets)
    forward_neg_src_dataset = ConcatDataset(forward_neg_src_datasets)
    forward_pos_tgt_dataset = ConcatDataset(forward_pos_tgt_datasets)
    forward_neg_tgt_dataset = ConcatDataset(forward_neg_tgt_datasets)
    if train_hyp_src:
        backward_pos_src_dataset = ConcatDataset(backward_pos_src_datasets)
        backward_neg_src_dataset = ConcatDataset(backward_neg_src_datasets)
        backward_pos_tgt_dataset = ConcatDataset(backward_pos_tgt_datasets)
        backward_neg_tgt_dataset = ConcatDataset(backward_neg_tgt_datasets)
    if train_cross_entropy:
        ce_src_dataset = ConcatDataset(ce_src_datasets)
        ce_tgt_dataset = ConcatDataset(ce_tgt_datasets)
    else:
        ce_src_dataset = None
        ce_tgt_dataset = None

    if prepend_tgt_lang_token:
        forward_lang_token = tgt_dict.index(f"<{tgt}>")
        forward_pos_tgt_dataset = PrependTokenDataset(forward_pos_tgt_dataset, forward_lang_token)
        forward_neg_tgt_dataset = PrependTokenDataset(forward_neg_tgt_dataset, forward_lang_token)
        if train_hyp_src:
            backward_lang_token = src_dict.index(f"<{src}>")
            backward_pos_tgt_dataset = PrependTokenDataset(backward_pos_tgt_dataset, backward_lang_token)
            backward_neg_tgt_dataset = PrependTokenDataset(backward_neg_tgt_dataset, backward_lang_token)
        if train_cross_entropy:
            ce_tgt_dataset = PrependTokenDataset(ce_tgt_dataset, forward_lang_token)

    if train_hyp_src:
        pos_src_dataset = ConcatDataset([
            forward_pos_src_dataset,
            backward_pos_src_dataset,
        ])
        neg_src_dataset = ConcatDataset([
            forward_neg_src_dataset,
            backward_neg_src_dataset,
        ])
        pos_tgt_dataset = ConcatDataset([
            forward_pos_tgt_dataset,
            backward_pos_tgt_dataset,
        ])
        neg_tgt_dataset = ConcatDataset([
            forward_neg_tgt_dataset,
            backward_neg_tgt_dataset,
        ])
    else:
        pos_src_dataset = forward_pos_src_dataset
        neg_src_dataset = forward_neg_src_dataset
        pos_tgt_dataset = forward_pos_tgt_dataset
        neg_tgt_dataset = forward_neg_tgt_dataset

    return ContrastiveLanguagePairDataset(
        pos_src_dataset,
        pos_src_dataset.sizes,
        neg_src_dataset,
        neg_src_dataset.sizes,
        src_dict,
        ce_src_dataset,
        ce_src_dataset.sizes if ce_src_dataset else None,
        pos_tgt_dataset,
        pos_tgt_dataset.sizes,
        neg_tgt_dataset,
        neg_tgt_dataset.sizes,
        ce_tgt_dataset,
        ce_tgt_dataset.sizes if ce_tgt_dataset else None,
        tgt_dict,
        left_pad_source=left_pad_source,
        left_pad_target=left_pad_target,
        eos=None,
        num_buckets=num_buckets,
        shuffle=shuffle,
        pad_to_multiple=pad_to_multiple,
    )
