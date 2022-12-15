# Digerati: a deep-learning approach for identifying PE and PPE family proteins rapidly and accurately.
## introduction

The genome of Mycobacterium tuberculosis contains a relatively high percentage (10%) of genes that are poorly characterized because of their poorly characterised due to their highly repetitive nature and high GC content. Some of these genes encode proteins of the PE/PPE family, which are though to be involved in host-pathogen interactions, virulence and the pathogenicity of disease. Members of this family are genet-ically divergent and challenging to both identify and classify using conventional computational tools. Thus, advanced in silico methods are needed to rapidly and accurately identify proteins of this family for subsequent functional annotation. Here, we developed the first deep learning-based approach, termed Di-gerati, for the rapidly and accurate identification of PE and PPE family proteins. Digerati was built upon a multipath parallel hybrid deep learning framework, which equips multi-layer convolutional neural net-works with bidirectional, long short-term memory, equipped with a self-attention module to effectively learn the higher-order feature representations of PE/PPE proteins. Empirical studies demonstrated that Di-gerati achieved a significantly better performance (~18-20%) than alignment-based approaches, including BLASTP, PHMMER and HHsuite, in both prediction accuracy and speed. Digerati is anticipated to facilitate community-wide efforts to conduct high-throughput identification and analysis of members of the PE/PPE family.
In addition, we constructed an user-friendly web server based on this framework for the public to use. We sincerely hope Digerati serves as a prominent tool for identiying PE and PPE family proteins rapidly and accurately.
## Environment
* Ubuntu
* Anaconda
* python 3.8
## Dependency
* biopython                     1.79
* pandas                        1.4.2
* scikit-learn                  1.1.1
* scipy                         1.8.1
* torch                         1.12.1
* wheel                         0.37.1
* numpy                         1.23.1
* tqdm                          4.64.0

## Usage
```
python predict.py -i {fasta file for predicting} -o {file name of prediction results}
```
For example:
* using the example test fasta file (examples/samples.fasta)
```
python predict.py -i examples/samples.fasta -o public/20221121102821_o2yC5H6N/result.txt 
```
output:
The output file includes categories: 
The second column is the Header information.
The third column is the label, 0 indicates that it is predicted Neither ppe nor pgrs, followed by its predicted probability. 
1 indicates it belongs to ppe family, followed by its predicted probability.
2 indicates it belongs to pgrs family, followed by its predicted probability.
```
0,Example1,0,0.51822644
1,Example2,0,0.6173083
```
* using other files, just change the file name, you can download the final prediction results in '.csv' or '.excel' format .
## Reference