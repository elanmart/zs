# AWA dataset setup

Note: Currently only AWA dataset uses makefile. For other datasets please use dataset specific instructions below.

### Step 1:
Set `ROOT` to place where you want to store your data (for example `export ROOT=/datasets/AWA`).
### Step 2:
Simply run
```make```

This should:
* Download all necessary zips
* Extract and preprocess data
* Validate correctness


### Cleanup
Use `make AWA_clean_all` to remove all the results.
Consutl `Makefile` for other, partial cleanup possibilities.


# CUB dataset setup

Estimated running time: `TODO`

### Step 1:
* Download birds data directly from
[this link](https://drive.google.com/file/d/0B0ywwgffWnLLZW9uVHNjb2JmNlE/view)
(or see `How to train a char-CNN-RNN model`, step `1` at
https://github.com/reedscot/cvpr2016).

* Move it to a desired directory `${DATA}`

### Step 2:

To download the data only, replace `RunAllTask` in the command below
with `UnpackDataTask`

Run the pipeline with:

```bash
ROOT="/where/data/should/be/stored"
DATA="/path/to/google/drive/download"

export PYTHONPATH='.'

luigi --module cub.tasks RunAllTask \
      --path ${ROOT} \
      --reed-data ${DATA} \
      --local-scheduler
```

### Cleaning generated data

If you want to clean the `CUB` directory, but retain the downloaded,
compressed files, run from the root directory where you built the dataset:

```bash
rm -rf config.py
rm -rf descriptions
rm -rf features
rm -rf labels
rm -rf indices
rm -rf metadata
rm -rf tmp
rm -rf vocabularies

rm -rf raw/CUB_200_2011
rm -rf raw/attributes.txt
rm -rf raw/descriptions
rm -rf raw/att_splits.mat
rm -rf raw/images
rm -rf raw/testclasses.txt
rm -rf raw/trainclasses1.txt
rm -rf raw/valclasses1.txt

rm -rf meta.pkl
rm -rf X.npy
rm -rf Y.npy
```


### Citations:

Please remember to cite `[1]`, `[2]`, and `[3]`


# Wikipedia dataset setup:

**THIS IS WORK IN PROGRESS. DO NOT USE**

Estimated running time: `TODO`

### About

This is a dataset based on a `Wikipedia` data dump.

See `config.py` for available settings.

Basically we can use `simplewiki` (tests) or `enwiki` (full data)

TODO: describe formats and conventions.

Basiaclly we:
* transform raw dump into file, where each row is `json`. Each `json`
contains document `title`, `summary`, `fulltext` and `categories`
* definitions are also `json`s. Each contains `name` and `definitions`
* From this we generate a bunch of representations and vocabularies
    * joint or seperate vocab for definitions and documents
    * datasets as `CSR` matrices or dense matrices of indices
    * etc... (TODO)
    * You can take a look into `./tasks/final.py` for some more formats
    we generate

### Running

Test extraction with
```bash
luigi --module tasks ExtarctTask \
      --path /tmp/wikipedia-dump-simple \
      --wiki simplewiki \
      --local-scheduler
```

Test definition generation with
```bash
luigi --module tasks TmpDefinitionsTask \
      --path /tmp/wikipedia-dump-simple \
      --wiki simplewiki \
      --local-scheduler
```


# Citations



```
[1]
@inproceedings{reed2016learning,
 title = {Learning Deep Representations of Fine-Grained Visual Descriptions,
 booktitle = {IEEE Computer Vision and Pattern Recognition},
 year = {2016},
 author = {Scott Reed and Zeynep Akata and Bernt Schiele and Honglak Lee},
}

[2]
@techreport{WelinderEtal2010,
	Author = {P. Welinder and S. Branson and T. Mita and C. Wah and F. Schroff and S. Belongie and P. Perona},
	Institution = {California Institute of Technology},
	Number = {CNS-TR-2010-001},
	Title = {{Caltech-UCSD Birds 200}},
	Year = {2010}
}

[3]
@inproceedings {xianCVPR17,
 title = {Zero-Shot Learning - The Good, the Bad and the Ugly},
 booktitle = {IEEE Computer Vision and Pattern Recognition (CVPR)},
 year = {2017},
 author = {Yongqin Xian and Bernt Schiele and Zeynep Akata}
}
```
