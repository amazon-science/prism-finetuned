# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import csv
from collections import defaultdict, Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Tuple

from mt_metrics_eval_custom.data import EvalSet


ErrorSpan = Tuple[int, int]


@dataclass
class Reference:
    text: str
    system: str
    seg_id: int
    score: Optional[float]
    error_spans: Optional[List[ErrorSpan]] = None


DEFAULT_REFERENCES = {
    "wmt20": {
        "en-de": ("ref", "ref"),  # Reference name, system name
        "zh-en": ("ref", "ref"),
    },
    # https://aclanthology.org/2021.wmt-1.73/ - Table 7
    "wmt21.news": {
        "en-de": ("refC", "refC"),
        "en-ru": ("refA", "refA"),
        "zh-en": ("refB", "refB"),
    },
    "wmt21.tedtalks": {
        "en-de": ("refA", "refA"),
        "en-ru": ("refA", "refA"),
        "zh-en": ("refB", "refB"),
    },
}
ALTERNATIVE_REFERENCES = {
    "wmt20": {
        "en-de": ("refb", "refb"),  # Reference name, system name
        "zh-en": ("refb", "refb"),
    },
    # https://aclanthology.org/2021.wmt-1.73/ - Table 7
    "wmt21.news": {
        "en-de": ("refA", "refA"),
        "en-ru": ("refB", "refB"),
        "zh-en": ("refA", "refA"),
    },
    "wmt21.tedtalks": {
        "zh-en": ("refA", "refA"),
    },
}
ALTERNATIVE_REFERENCES2 = {
    "wmt21.news": {
        "en-de": ("refB", "refB"),
    },
}
ALTERNATIVE_REFERENCES3 = {
    "wmt21.news": {
        "en-de": ("refD", "refD"),
    },
}
PARAPHRASED_REFERENCES = {
    "wmt20": {
        "en-de": ("refp", "refp"),
    }
}

MQM_FILENAMES = {
    ("wmt20", "en-de"): "mqm_newstest2020_ende.no-postedits.tsv",
    ("wmt20", "zh-en"): "mqm_newstest2020_zhen.no-postedits.tsv",
    ("wmt21.news", "en-de"): "mqm-newstest2021_ende.tsv",
    ("wmt21.news", "zh-en"): "mqm-newstest2021_zhen.tsv",
    ("wmt21.tedtalks", "en-de"): "mqm-ted_ende.tsv",
    ("wmt21.tedtalks", "zh-en"): "mqm-ted_zhen.tsv",
}

SYSTEM_FAMILIES = [
    ('metricsystem1', 'metricsystem2', 'metricsystem3', 'metricsystem4', 'metricsystem5'),
]


@dataclass
class MQMRating:
    rater: str
    source: str
    target: str
    category: str
    severity: str


@dataclass
class MQMSample:
    system: str
    doc: str
    doc_id: int
    seg_id: int
    target: str
    ratings: List[MQMRating]

    def get_error_spans(self, flat=True) -> List[ErrorSpan]:
        error_spans = []
        for rating in self.ratings:
            if rating.severity == "No-error":
                continue
            if "<v>" not in rating.target:
                continue
            if rating.target.count("<v>") != rating.target.count("</v>"):
                continue
            current_text = rating.target
            while "<v>" in current_text:
                start_index = current_text.index("<v>")
                current_text = current_text.replace("<v>", "", 1)
                end_index = current_text.index("</v>")
                if "<v>" in current_text:
                    assert end_index < current_text.index("<v>")
                current_text = current_text.replace("</v>", "", 1)
                error_spans.append((start_index, end_index))
        if flat:
            return self._flatten_error_spans(error_spans)
        return error_spans

    def _flatten_error_spans(self, error_spans: List[ErrorSpan]) -> List[ErrorSpan]:
        char_flags = [False for _ in self.target]
        for start_index, end_index in error_spans:
            for i in range(start_index, end_index):
                char_flags[i] = True
        flat_indices = []
        next_span = []
        for i, char_flag in enumerate(char_flags + [False]):
            if char_flag == True and len(next_span) == 0:
                next_span.append(i)
            elif char_flag == False and len(next_span) == 1:
                next_span.append(i)
            if len(next_span) == 2:
                flat_indices.append((next_span[0], next_span[1]))
                next_span = []
        return flat_indices


class MQMData:

    def __init__(self, name: str, language_pair: str):
        self.name = name
        self.language_pair = language_pair
        mqm_dir = Path(__file__).parent / "data" / "mqm"
        mqm_filename = MQM_FILENAMES[(name, language_pair)]
        self.data_path = mqm_dir / mqm_filename
        assert self.data_path.exists()
        self.mqm_samples = self.load_samples()

    def load_samples(self) -> Dict[Tuple[str, int], MQMSample]:
        mqm_samples = dict()
        with open(self.data_path, newline='') as f:
            reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
            for line in reader:
                sample = MQMSample(
                    system=self.clean_system_name(line["system"].strip()),
                    doc=line["doc"].strip(),
                    doc_id=int(line["doc_id"]),
                    seg_id=int(line["seg_id"]),
                    target=line["target"].replace("<v>", "").replace("</v>", "").strip(),
                    ratings=[MQMRating(
                        rater=line["rater"].strip(),
                        source=line["source"].strip(),
                        target=line["target"].strip(),
                        category=line["category"].strip(),
                        severity=line["severity"].strip() if line["severity"] else None,
                    )],
                )
                if (sample.system, sample.seg_id) in mqm_samples:
                    old_sample = mqm_samples[(sample.system, sample.seg_id)]
                    assert old_sample.target == sample.target
                    old_sample.ratings += sample.ratings
                else:
                    mqm_samples[(sample.system, sample.seg_id)] = sample
        return mqm_samples

    def clean_system_name(self, system: str) -> str:
        if self.name.startswith("wmt21"):
            if system.startswith("hyp."):
                system = system[4:]
            elif system.startswith("ref."):
                system = "ref" + system[4:]
        if self.name == "wmt21.tedtalks":
            if system == "ref":
                system = "refA"
            if system == "refB":
                system = "refB"
        return system

    @property
    def num_annotated_segments(self) -> int:
        seg_ids = set()
        for system, seg_id in self.mqm_samples:
            if system.startswith("ref.") or system.startswith("ref") or "human" in system.lower():
                continue
            seg_ids.add(seg_id)
        return len(seg_ids)

    @property
    def raters(self) -> List[str]:
        raters = set()
        for sample in self.mqm_samples.values():
            for rating in sample.ratings:
                raters.add(rating.rater)
        return list(sorted(raters, key=lambda rater: int(rater.replace("rater", ""))))

    @property
    def num_raters_per_segment(self) -> float:
        rater_sets = defaultdict(set)
        for system, seg_id in self.mqm_samples:
            if system.startswith("ref.") or system.startswith("ref") or "human" in system.lower():
                continue
            sample = self.mqm_samples[(system, seg_id)]
            if not sample.ratings:
                continue
            rater_sets[(system, seg_id)].update({rating.rater for rating in sample.ratings})
        counts = [len(rater_set) for rater_set in rater_sets]
        return sum(counts) / len(counts)


class WMTData:

    def __init__(self,
                 name: str,
                 language_pair: str,
                 load_segment_scores: bool = False,
                 load_mqm_spans: bool = False,
                 ):
        self.name = name
        self.language_pair = language_pair
        self.wmt_data = EvalSet(name, language_pair, read_stored_metric_scores=load_segment_scores)
        self.mqm_data = None
        if load_mqm_spans:
            self.mqm_data = MQMData(name, language_pair)
            for (system, seg_id) in self.mqm_data.mqm_samples:
                assert system in self.systems_with_mqm_annotation

    def __len__(self):
        return len(self.source_sentences)

    def __str__(self):
        return f"{self.name}.{self.language_pair}"

    @property
    def source_sentences(self) -> List[str]:
        return self.wmt_data.src

    def get_translations(self, system: str) -> List[str]:
        return self.wmt_data.sys_outputs[system]

    @property
    def systems(self) -> List[str]:
        return list(sorted(self.wmt_data.sys_names))

    @property
    def systems_with_mqm_annotation(self) -> List[str]:
        return list(sorted(self.wmt_data._scores["sys"]["mqm"]))

    @property
    def reference_names(self) -> List[str]:
        return list(sorted(self.wmt_data.all_refs))

    @property
    def default_reference_name(self) -> str:
        return DEFAULT_REFERENCES[self.name][self.language_pair][0]

    @property
    def default_reference_system(self) -> str:
        return DEFAULT_REFERENCES[self.name][self.language_pair][1]

    @property
    def paraphrased_reference_name(self) -> str:
        try:
            return PARAPHRASED_REFERENCES[self.name][self.language_pair][0]
        except KeyError:
            return None

    @property
    def paraphrased_reference_system(self) -> str:
        try:
            return PARAPHRASED_REFERENCES[self.name][self.language_pair][1]
        except KeyError:
            return None

    @property
    def alternative_reference_name(self) -> str:
        try:
            return ALTERNATIVE_REFERENCES[self.name][self.language_pair][0]
        except KeyError:
            return None

    @property
    def alternative_reference_system(self) -> str:
        try:
            return ALTERNATIVE_REFERENCES[self.name][self.language_pair][1]
        except KeyError:
            return None

    @property
    def alternative_reference2_name(self) -> str:
        try:
            return ALTERNATIVE_REFERENCES2[self.name][self.language_pair][0]
        except KeyError:
            return None

    @property
    def alternative_reference2_system(self) -> str:
        try:
            return ALTERNATIVE_REFERENCES2[self.name][self.language_pair][1]
        except KeyError:
            return None

    @property
    def alternative_reference3_name(self) -> str:
        try:
            return ALTERNATIVE_REFERENCES3[self.name][self.language_pair][0]
        except KeyError:
            return None

    @property
    def alternative_reference3_system(self) -> str:
        try:
            return ALTERNATIVE_REFERENCES3[self.name][self.language_pair][1]
        except KeyError:
            return None

    def print_rater_coverage(self) -> None:
        print("\t" + "\t".join(self.mqm_data.raters))
        for system in self.systems_with_mqm_annotation:
            print(system, end="\t")
            rater_counts = Counter()
            for (system_, seg_id), sample in self.mqm_data.mqm_samples.items():
                if system_ != system:
                    continue
                for rater in {rating.rater for rating in sample.ratings}:
                    rater_counts[rater] += 1
            for rater in self.mqm_data.raters:
                print(rater_counts[rater], end="\t")
            print()

    def get_system_score(self, system: str, type: str = "mqm") -> Optional[float]:
        if system not in self.systems_with_mqm_annotation:
            return None
        if type == "mqm":
            return self.wmt_data._scores["sys"]["mqm"][system][0]
        else:
            raise ValueError

    def get_segment_score(self, system: str, seg_index: int, type: str = "mqm") -> Optional[float]:
        if system not in self.systems_with_mqm_annotation:
            return None
        return self.wmt_data._scores["seg"][type][system][seg_index]

    def get_indices_of_unannotated_segments(self, system: str) -> List[int]:
        return [i for i, score in enumerate(self.wmt_data._scores["seg"]["mqm"][system]) if score is None]


class ReferencesLoader:

    def __init__(self,
                 wmt_data: WMTData,
                 use_standard_references: bool = True,
                 use_uniform_scores: bool = True,
                 use_all_human_references: bool = False,  # Except paraphrased
                 use_paraphrased_references: bool = False,
                 use_alternative_references: bool = False,
                 use_alternative_references2: bool = False,
                 use_alternative_references3: bool = False,
                 use_peer_references: bool = False,
                 use_mqm_scores: bool = False,
                 exclude_segments_without_annotation: bool = False,
                 exclude_flawed_references: bool = False,
                 use_references_from_dir: Path = None,
                 sample_single_reference: bool = False,
                 ):
        self.use_standard_references = use_standard_references
        self.use_uniform_scores = use_uniform_scores
        self.use_all_human_references = use_all_human_references
        if use_paraphrased_references and wmt_data.paraphrased_reference_name is None:
            raise NotImplementedError
        self.use_paraphrased_references = use_paraphrased_references
        if use_alternative_references and wmt_data.alternative_reference_name is None:
            raise NotImplementedError
        self.use_alternative_references = use_alternative_references
        if use_alternative_references2 and wmt_data.alternative_reference2_name is None:
            raise NotImplementedError
        self.use_alternative_references2 = use_alternative_references2
        if use_alternative_references3 and wmt_data.alternative_reference3_name is None:
            raise NotImplementedError
        self.use_alternative_references3 = use_alternative_references3
        self.use_peer_references = use_peer_references
        self.use_mqm_scores = use_mqm_scores
        self.exclude_segments_without_annotation = exclude_segments_without_annotation
        self.exclude_flawed_references = exclude_flawed_references
        if use_references_from_dir is not None:
            assert use_references_from_dir.exists()
            self.use_references_from_path = use_references_from_dir / f"{wmt_data}.ref"
            assert self.use_references_from_path.exists()
        else:
            self.use_references_from_path = None
        self.sample_single_reference = sample_single_reference
        self.wmt_data = wmt_data

    def get_reference_lists(self) -> List[List[Reference]]:  # segments -> references
        reference_lists: List[List[Reference]] = [list() for _ in range(len(self.wmt_data))]
        for system in self.wmt_data.systems:
            if not any([
                system == self.wmt_data.default_reference_system and self.use_standard_references,
                system == self.wmt_data.paraphrased_reference_system and self.use_paraphrased_references,
                system == self.wmt_data.alternative_reference_system and self.use_alternative_references,
                system == self.wmt_data.alternative_reference2_system and self.use_alternative_references2,
                system == self.wmt_data.alternative_reference3_system and self.use_alternative_references3,
                self._system_is_human(system) and system != self.wmt_data.paraphrased_reference_system and self.use_all_human_references,
                not self._system_is_human(system) and self.use_peer_references,
            ]):
                continue

            for seg_index in range(len(self.wmt_data)):
                reference = Reference(
                    text=self.wmt_data.get_translations(system)[seg_index],
                    system=system,
                    seg_id=(seg_index + 1),
                    score=None,
                )
                if not reference.text.strip():
                    continue
                mqm_score = self.wmt_data.get_segment_score(system, seg_index)
                if (mqm_score is None or mqm_score < 0) and self.exclude_flawed_references:
                    continue
                if self.use_uniform_scores:
                    reference.score = 1
                elif self.use_mqm_scores:
                    reference.score = self.wmt_data.get_segment_score(system, seg_index)
                reference_lists[seg_index].append(reference)

        if self.exclude_segments_without_annotation:
            for seg_index in range(len(self.wmt_data)):
                for system in self.wmt_data.systems_with_mqm_annotation:
                    if self.wmt_data.get_segment_score(system, seg_index) is None:
                        reference_lists[seg_index] = []

        assert len(reference_lists) == len(self.wmt_data)

        if self.use_references_from_path is not None:
            with open(self.use_references_from_path) as f:
                lines = list(f)
            assert len(lines) == len(reference_lists)
            for seg_index, line in enumerate(lines):
                reference = Reference(
                    text=line.strip(),
                    system=self.use_references_from_path.parent.name,
                    seg_id=(seg_index + 1),
                    score=1,
                )
                reference_lists[seg_index].append(reference)

        return reference_lists

    def _system_is_human(self, system: str) -> bool:
        return system in self.wmt_data.reference_names or "human" in system.lower()
