Code for the research paper "Trained MT Metrics Learn to Cope with Machine-translated References"

## Installation
- Install PyTorch
- cd .. && pip install -r requirements.txt
- pip install git+https://github.com/google-research/mt-metrics-eval

### mt-metrics-eval
- python -m mt_metrics_eval.mtme --download (Puts ~1G of data into $HOME/.mt-metrics-eval)

## Downloading the pre-trained model
- wget http://data.statmt.org/prism/m39v1.tar
- tar xf m39v1.tar
- mkdir models
- mv m39v1 models

## Preparing the data

### Downloading MQM data
See scripts/download_data.sh

### Extracting relative rankings
- mkdir data/wmt_rr
- python scripts/convert_mqm_to_relative_ranking_data.py

### Concatenating the language pairs and creating a trainâ€“valid split
See scripts/create_data_split.sh

### Preprocessing data for Prism fine-tuning with fairseq
- mkdir data/prism_finetuning_data
- python scripts/prepare_prism_finetuning_data.py (might take a while)

## Fine-tuning
- bash scripts/finetune_main.sh

## Metric usage
Please refer to the reference implementation of Prism (https://github.com/thompsonb/prism) for instructions on using the metric

## Meta-evaluation
- pip install -r requirements-eval.txt
- python scripts/run_meta_evaluation.py

## Post-editese experiments
- python post_editese/scripts/run_<metric>.py

## Citation

Please cite this work as:

```bibtex
@misc{vamvasetal2023trainedmetrics,
      title={Trained MT Metrics Learn to Cope with Machine-translated References},
      author={Vamvas, Jannis and Domhan, Tobias and Trenous, Sony and Sennrich, Rico and Hasler, Eva},
      booktitle={Proceedings of the Eighth Conference on Machine Translation (WMT)},
      year={2023}
}
```

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the CC-BY-NC-4.0 License.
