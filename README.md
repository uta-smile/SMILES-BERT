# SMILES-BERT

Code for paper 
> Wang, Sheng, Yuzhi Guo, Yuhong Wang, Hongmao Sun, and Junzhou Huang. "SMILES-BERT: large scale unsupervised pre-training for molecular property prediction." In Proceedings of the 10th ACM International Conference on Bioinformatics, Computational Biology and Health Informatics, pp. 429-436. 2019.

**Note** This code was developed with [fairseq](https://github.com/facebookresearch/fairseq), a sequence-to-sequence
learning toolkit from Facebook AI Research. The fairseq version that we used in our code was around early 2019. Let us know if there is any license concern.

**Note** There are many unrated files/code to this paper and it could be hard to read and use the code. Following will be some commands we used for training, do not hesitate to reach out if you have any question.

### Binarize the pre-training dataset
A binarized dataset could speed up the dataset loading process. Here is the command:
```
python binarize_smiles.py --data /path/to/zinc --destdir /path/to/bin/zinc --workers 16
```

The dataset used for pretraining should contain three files `train`, `valid`, `test` and each of the file should be one SMILEs in one line, without header.

### Pre-training 
```
python train.py /path/to/pretrain/data-bin --data-bin --arch bertsmall --save-dir=/path/to/save/ckpts --task bert --max-sentences=256 --bert-pretrain True --optimizer adam --lr 0.0001 --adam-betas '(0.9, 0.999)' --weight-decay 0.01 --lr-scheduler inverse_sqrt --warmup-init-lr 1e-09 --warmup-updates 10000
```

### Fine-tuning on labeled dataset

A sample fine-tuning command could be

```
python train.py /path/to/labeled/data --arch bertsmall --task smile_property_prediction --save-dir /path/to/save/ckpts --max-sentences 16 --optimizer adam --lr 0.000001 --min-lr 1e-10 --adam-betas '(0.9, 0.999)' --weight-decay 0.01 --dropout 0.5 --lr-scheduler fixed --reverse-input False --pad-go True --left-pad-source=False --input-feed False --criterion seq3seq --prop-pred --num-props 1 --cls-index=[0] --pred-hidden-dim 0 --reset-optimizer --max-epoch=100
```
The dataset used for fine-tuning should contain three files `train`, `valid`, `test` and each of the file should be one SMILEs and properties separated by comma.

Our pre-trained model will be uploaded soon and the link will be updated here.

#### Join the fairseq community

* Facebook page: https://www.facebook.com/groups/fairseq.users
* Google group: https://groups.google.com/forum/#!forum/fairseq-users

