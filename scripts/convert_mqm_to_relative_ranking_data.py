# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import csv
import itertools
from collections import defaultdict, Counter
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Iterable, Optional

from mt_metrics_eval_custom.data import EvalSet

ErrorSpan = Tuple[int, int]


@dataclass
class MQMRating:
    rater: str
    source: str
    target: str
    category: str
    severity: str

    @property
    def penalty(self) -> float:
        if self.severity.lower() == "major":
            if self.category.lower() == "non-translation":
                return -25
            else:
                return -5
        elif self.severity.lower() == "minor":
            if self.category.lower() == "fluency/punctuation":
                return -0.1
            else:
                return -1
        elif self.severity.lower() in {"no-error", "neutral"}:
            return -0
        else:
            raise ValueError


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

    def get_score_for_rater(self, rater: str) -> Optional[float]:
        rater_ratings = [rating for rating in self.ratings if rating.rater == rater]
        if not rater_ratings:
            return None
        return sum([rating.penalty for rating in rater_ratings])


@dataclass
class RelativeRanking:
    rater: str
    pos_sample: MQMSample
    neg_sample: MQMSample
    pos_score: float
    neg_score: float


class MQMData:

    def __init__(self, csv_path: Path, name: str):
        self.data_path = csv_path
        assert self.data_path.exists(), self.data_path
        assert name in {"wmt20", "wmt21.news", "wmt21.tedtalks"}
        self.name = name
        self.mqm_samples: Dict[Tuple[str, int], MQMSample] = self.load_samples()

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
                    assert old_sample.target.strip() == sample.target.strip(), sample.target + " ... " + old_sample.target
                    old_sample.ratings += sample.ratings
                else:
                    mqm_samples[(sample.system, sample.seg_id)] = sample
        return mqm_samples

    def clean_system_name(self, system: str) -> str:
        if self.name.startswith("wmt21"):
            if system.startswith("hyp."):
                system = system[4:]
            elif system.startswith("ref."):
                system = "ref-" + system[4:]
        if self.name == "wmt21.tedtalks":
            if system == "ref":
                system = "ref-A"
            if system == "refB":
                system = "ref-B"
        return system

    @property
    def num_annotated_segments(self) -> int:
        seg_ids = set()
        for system, seg_id in self.mqm_samples:
            if system.startswith("ref.") or system.startswith("ref-") or "human" in system.lower():
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
            if system.startswith("ref.") or system.startswith("ref-") or "human" in system.lower():
                continue
            sample = self.mqm_samples[(system, seg_id)]
            if not sample.ratings:
                continue
            rater_sets[(system, seg_id)].update({rating.rater for rating in sample.ratings})
        counts = [len(rater_set) for rater_set in rater_sets]
        return sum(counts) / len(counts)

    @property
    def num_translations_per_rater(self) -> float:
        rater_seg_id_counter = Counter()
        for system, seg_id in self.mqm_samples:
            if system.startswith("ref.") or system.startswith("ref-") or "human" in system.lower():
                continue
            sample = self.mqm_samples[(system, seg_id)]
            for rating in sample.ratings:
                rater_seg_id_counter[(rating.rater, seg_id)] += 1
        return sum(rater_seg_id_counter.values()) / len(rater_seg_id_counter)

    def get_relative_rankings_by_same_rater(self) -> Iterable[RelativeRanking]:
        raters = self.raters
        seg_id_samples: Dict[int, List[MQMSample]] = defaultdict(list)
        for (system, seg_id), sample in self.mqm_samples.items():
            seg_id_samples[seg_id].append(sample)
        for seg_id, samples in seg_id_samples.items():
            for sample1, sample2 in itertools.combinations(samples, 2):
                for rater in raters:
                    score1 = sample1.get_score_for_rater(rater)
                    score2 = sample2.get_score_for_rater(rater)
                    if score1 is None or score2 is None:
                        continue
                    if score1 > score2:
                        pos_sample = sample1
                        neg_sample = sample2
                    elif score1 < score2:
                        pos_sample = sample2
                        neg_sample = sample1
                    else:
                        continue
                    yield RelativeRanking(
                        rater=rater,
                        pos_sample=pos_sample,
                        neg_sample=neg_sample,
                        pos_score=pos_sample.get_score_for_rater(rater),
                        neg_score=neg_sample.get_score_for_rater(rater),
                    )


if __name__ == '__main__':
    in_dir = Path(__file__).parent.parent / "data" / "wmt_mqm_orig"
    out_dir = Path(__file__).parent.parent / "data" / "wmt_rr"
    for testset, language_pair, csv_name in [
        ("wmt20", "en-de", "mqm_newstest2020_ende.no-postedits.tsv"),
        ("wmt20", "zh-en", "mqm_newstest2020_zhen.no-postedits.tsv"),
    ]:
        csv_path = in_dir / csv_name
        data = MQMData(csv_path, testset)
        wmt_data = EvalSet(testset, language_pair)
        # print(data.num_annotated_segments)
        # print(data.raters)
        # print(data.num_raters_per_segment)
        # print("Avg. translation per rater:", data.num_translations_per_rater)
        out_path = out_dir / f"{testset}.{language_pair}.csv"
        out_fieldnames = [
            "domain",
            "src",
            "pos",
            "neg",
            "ref",
            "better_score",
            "worse_score",
            "lp",
            "rater",
            "better_system",
            "worse_system",
        ]
        with open(out_path, "w") as f:
            writer = csv.DictWriter(f, fieldnames=out_fieldnames)
            writer.writeheader()
            minor_diff_count = 0
            for relative_ranking in data.get_relative_rankings_by_same_rater():
                assert relative_ranking.pos_sample.seg_id == relative_ranking.neg_sample.seg_id
                seg_id = relative_ranking.pos_sample.seg_id
                if relative_ranking.pos_score - relative_ranking.neg_score <= 0.1:
                    minor_diff_count += 1
                    continue
                writer.writerow({
                    "domain": testset,
                    "src": wmt_data.src[seg_id-1],
                    "pos": relative_ranking.pos_sample.target,
                    "neg": relative_ranking.neg_sample.target,
                    "ref": wmt_data.all_refs[wmt_data.std_ref][seg_id-1],
                    "better_score": relative_ranking.pos_score,
                    "worse_score": relative_ranking.neg_score,
                    "lp": language_pair,
                    "rater": relative_ranking.rater,
                    "better_system": relative_ranking.pos_sample.system,
                    "worse_system": relative_ranking.neg_sample.system,
                })
            print("Skipped minor diffs", minor_diff_count)
