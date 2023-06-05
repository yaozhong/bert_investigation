# Investigation of the BERT model on nucleotide sequences with non-standard pre-training and the evaluation of different k-mer embeddings

In this study, we focused on nucleotide sequences and illustrated the learning outcomes of a typical BERT model learned through pre-training.
We used a non-standard pre-training approach to scrutinize different modules by incorporating randomness at both the data and model levels.

![](figures/nonstandard_pretrain.eps)



* **data/ptData**: source code of generate random sequences.

* **ft_tasks**: source code of using different k-mer embeddings in the downstream tasks of TATA prediciton and TBFS prediction.
* **pt_models**:  download link of the pre-trained model using random data.
* **results**: detailed results of each dataset of TBFS tasks.
