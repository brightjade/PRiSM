# PRiSM

Source code for our *Findings of IJCNLP-AACL 2023* paper [PRiSM: Enhancing Low-Resource Document-Level Relation Extraction with Relation-Aware Score Calibration](https://arxiv.org/abs/2309.13869).

## Requirements

- Python (tested on 3.8.16)
- CUDA (tested on 11.7)
- PyTorch (tested on 1.13.1)
- Transformers (tested on 4.30.0)
- numpy (tested on 1.22.4)
- wandb
- tqdm

## Datasets

Datasets can be downloaded here: [DocRED](https://drive.google.com/drive/folders/1c5-0YwnoJx8NS6CV2f-NoTHR__BdkNqw), [Re-DocRED](https://github.com/tonytan48/Re-DocRED), [DWIE](https://github.com/klimzaporojets/DWIE). The expected structure of files is:

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

## Citation

If you make use of this code in your work, please kindly cite our paper:

```bibtex
@inproceedings{choi2023prism,
               author={Choi, Minseok and Lim, Hyesu and Choo, Jaegul},
               title={P{R}i{S}{M}: Enhancing Low-Resource Document-Level Relation Extraction with Relation-Aware Score Calibration},
               booktitle={Proceedings of the 13th International Joint Conference on Natural Language Processing and the 3rd Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics},
               month={November},
               year={2023},
               address={Nusa Dua, Bali},
               publisher={Association for Computational Linguistics},
               pages={39--47},
               url={https://aclanthology.org/2023.findings-ijcnlp.4}
}
```

## Acknowledgements

This work was supported by Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government (MSIT) (No.2019-0-00075, Artificial Intelligence Graduate School Program (KAIST)), the National Supercomputing Center with supercomputing resources including technical support (KSC-2022-CRE-0312), and Samsung Electronics Co., Ltd.
