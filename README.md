# PRiSM

Source code for our *Findings of IJCNLP-AACL 2023* paper <a href="https://github.com/brightjade/PRiSM" target="_blank">PRiSM: Enhancing Low-Resource Document-Level Relation Extraction with Relation-Aware Score Calibration</a>.

## Requirements

- Python (tested on 3.8.16)
- CUDA (tested on 11.7)
- PyTorch (tested on 1.13.1)
- Transformers (tested on 4.30.0)
- numpy (tested on 1.22.4)
- wandb
- tqdm

## Datasets

Datasets can be downloaded here: <a href="https://drive.google.com/drive/folders/1c5-0YwnoJx8NS6CV2f-NoTHR__BdkNqw" target="_blank">DocRED</a>, <a href="https://github.com/tonytan48/Re-DocRED" target="_blank">Re-DocRED</a>, <a href="https://github.com/klimzaporojets/DWIE/" target="_blank">DWIE</a>. The expected structure of files is:

```
[working directory]
 |-- data
 |    |-- DocRED
 |    |    |-- train_distant.json        
 |    |    |-- train.json
 |    |    |-- dev.json
 |    |    |-- test.json
 |    |    |-- label_map.json
 |    |    |-- rel_info.json
 |    |    |-- rel_desc.json
 |    |-- Re-DocRED
 |    |    |-- train_distant.json        
 |    |    |-- train.json
 |    |    |-- dev.json
 |    |    |-- test.json
 |    |    |-- label_map.json
 |    |    |-- rel_info.json
 |    |    |-- rel_desc.json
 |    |-- DWIE
 |    |    |-- train/
 |    |    |-- dev/
 |    |    |-- test/
 |    |    |-- label_map.json
 |    |    |-- rel_desc.json
```

## Training and Evaluation

Train the model with the following command:

```bash
>> bash scripts/train.sh
```

Evaluate the model with the following command:

```bash
>> bash scripts/evaluate.sh
```
<!-- 
## Citation

If you make use of this code in your work, please kindly cite our paper:

```bibtex
@inproceedings{choi2023prism,
               title={PRiSM: Enhancing Low-Resource Document-Level Relation Extraction with Relation-Aware Score Calibration},
               author={Choi, Minseok and Lim, Hyesu and Choo, Jaegul},
               booktitle={Findings of the Association for Computational Linguistics: AACL-IJCNLP 2023},
               month=nov,
               year={2023},
               address="Bali, Indonesia",
               publisher="Association of Computational Linguistics",
               url="",
               pages="",
               abstract=""
}
```

## Acknowledgements

This work was supported by ... -->