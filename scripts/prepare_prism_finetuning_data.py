# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import contextlib
import csv
import logging
import shutil
from collections import defaultdict
from pathlib import Path
from unittest.mock import patch

from fairseq.binarizer import VocabularyDatasetBinarizer, FileBinarizer
from fairseq.tasks import FairseqTask


logging.basicConfig(level=logging.INFO)


class PrismFinetuningData:

    PRISM_LANGUAGES = {'ar', 'bg', 'bn', 'ca', 'cs', 'da', 'de', 'el', 'en', 'es', 'et', 'eo', 'fi', 'fr', 'he', 'hr', 'hu', 'id', 'it', 'ja', 'kk', 'lt', 'lv', 'mk', 'nl', 'no', 'pl', 'pt', 'ro', 'ru', 'sk', 'sl', 'sq', 'sr', 'sv', 'tr', 'uk', 'vi', 'zh'}

    def __init__(self, comet_train_path: Path, comet_valid_path: Path, output_dir: Path, model_dir: Path = None):
        output_dir.mkdir()
        self.comet_train_path = comet_train_path
        self.comet_valid_path = comet_valid_path
        self.output_dir = output_dir
        self.model_dir = model_dir or Path(__file__).parent.parent / "models" / "m39v1"
        assert self.model_dir.exists()
        with open(self.comet_train_path) as f:
            self.comet_train_lines = list(csv.DictReader(f))
        with open(self.comet_valid_path) as f:
            self.comet_valid_lines = list(csv.DictReader(f))
        self.language_pairs = list(sorted({line["lp"] for line in self.comet_train_lines} |
                                          {line["lp"] for line in self.comet_valid_lines}))
        self.language_pairs = [
            language_pair for language_pair in self.language_pairs
            if language_pair.split("-")[0] in self.PRISM_LANGUAGES and language_pair.split("-")[1] in self.PRISM_LANGUAGES
        ]
        self.raw_paths_per_language_pair = defaultdict(list)
        self.bpe_paths_per_language_pair = defaultdict(list)

    def extract_data(self):
        for split, input_lines in [
            ("train", self.comet_train_lines),
            ("valid", self.comet_valid_lines),
        ]:
            for language_pair in self.language_pairs:
                language_pair_lines = [line for line in input_lines if line["lp"] == language_pair]
                if not language_pair_lines:
                    continue
                src_lang, tgt_lang = language_pair.split("-")
                assert src_lang in self.PRISM_LANGUAGES
                assert tgt_lang in self.PRISM_LANGUAGES

                # Format the paths of the raw data (16 per split)
                for tag in ("pos", "neg"):
                    for score_from, score_to in (
                            ("ref", "hyp"),
                            ("hyp", "ref"),
                            ("src", "hyp"),
                            ("hyp", "src"),
                    ):
                        for side in (score_from, score_to):
                            if side == "hyp" or side == "ref":
                                side_lang = tgt_lang
                            else:
                                side_lang = src_lang
                            path = Path(self.output_dir / f"{split}.{language_pair}.{score_from}-{score_to}.{side}.{tag}.{side_lang}")
                            self.raw_paths_per_language_pair[language_pair].append(path)

                # Open all files in the list above
                with contextlib.ExitStack() as stack:
                    files = {
                        path.name: stack.enter_context(open(path, "w"))
                        for path in self.raw_paths_per_language_pair[language_pair]
                        if path.name.startswith(f"{split}.")
                    }
                    for line in language_pair_lines:
                        # Write fields to all the relevant paths
                        for path_name, f in files.items():
                            if ".src." in path_name:
                                f.write(line["src"] + "\n")
                            if ".ref." in path_name:
                                f.write(line["ref"] + "\n")
                            if ".hyp.pos." in path_name:
                                f.write(line["pos"] + "\n")
                            if ".hyp.neg." in path_name:
                                f.write(line["neg"] + "\n")

    def tokenize_data(self):
        from spm_encode import main as spm_encode_main

        for language_pair, raw_paths in self.raw_paths_per_language_pair.items():
            for raw_path in raw_paths:
                bpe_path = raw_path.parent / (raw_path.name + ".bpe")
                self.bpe_paths_per_language_pair[language_pair].append(bpe_path)

        for language_pair in self.raw_paths_per_language_pair:
            for split in ("train", "valid"):
                input_paths = [path for path in self.raw_paths_per_language_pair[language_pair] if path.name.startswith(f"{split}.")]
                output_paths = [path for path in self.bpe_paths_per_language_pair[language_pair] if path.name.startswith(f"{split}.")]
                assert len(input_paths) == len(output_paths)
                if not input_paths:
                    continue
                with patch('argparse._sys.argv', [
                                                     "...",
                                                     "--model", str(self.model_dir / "spm.model"),
                                                     "--max-len", str(250),
                                                     "--inputs"] + list(map(str, input_paths)) + [
                                                     "--outputs"] + list(map(str, output_paths)) + [
                                                 ]):
                    spm_encode_main()

    def preprocess_data(self):
        """
        We cannot call fairseq preprocess because it is too opinionated on how the paths look like
        """
        bin_dir = self.output_dir / "data-bin"
        bin_dir.mkdir()
        src_dict_path = self.model_dir / "dict.src.txt"
        assert src_dict_path.exists()

        # Assuming that src_dict and tgt_dict are identical
        src_dict = FairseqTask.load_dictionary(str(src_dict_path))

        for language_pair in self.language_pairs:
            src_lang, tgt_lang = language_pair.split("-")
            for lang in (src_lang, tgt_lang):
                from fairseq_cli.preprocess import _dict_path
                src_dict.save(_dict_path(lang, bin_dir))

        binarizer = VocabularyDatasetBinarizer(
            src_dict,
            append_eos=True,
        )

        for language_pair, bpe_paths in self.bpe_paths_per_language_pair.items():
            for path in bpe_paths:
                final_summary = FileBinarizer.multiprocess_dataset(
                    input_file=str(path),
                    dataset_impl="mmap",
                    binarizer=binarizer,
                    output_prefix=str(bin_dir / path.name),
                    vocab_size=len(src_dict),
                )
                logging.info(final_summary)


if __name__ == '__main__':
    raw_data_dir = Path(__file__).parent.parent / "data" / "wmt_rr"
    assert raw_data_dir.exists()
    prism_data_dir = Path(__file__).parent.parent / "data" / "prism_finetuning_data"
    assert prism_data_dir.exists()
    comet_train_path = raw_data_dir / "wmt20.train.csv.shuf"
    assert comet_train_path.exists()
    comet_test_path = raw_data_dir / "wmt20.valid.csv.shuf"
    assert comet_test_path.exists()
    out_dir = prism_data_dir
    shutil.rmtree(out_dir, ignore_errors=True)
    data = PrismFinetuningData(comet_train_path, comet_test_path, out_dir)
    print(data.language_pairs)
    data.extract_data()
    data.tokenize_data()
    data.preprocess_data()
