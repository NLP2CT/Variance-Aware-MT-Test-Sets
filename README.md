# Variance-Aware-MT-Test-Sets
Variance-Aware Machine Translation Test Sets

**Update**: If you are looking for the huggingface ``Datasets`` version of these test sets, [Gabriele Sarti](https://huggingface.co/gsarti) has created a repository [here](https://huggingface.co/datasets/gsarti/wmt_vat). 

### License
See `LICENSE`. We follow the data licensing plan as the same as the WMT benchmark. 

### VAT Data
We release 70 lightweight and discriminative test sets for machine translation evaluation, covering 35 translation directions from WMT16 to WMT20 competitions.  See `VAT_data` folder for detailed information.

For each translation direction of a specific year, both source and reference are provided for different types of evaluation metrics.  For example,

```
VAT_data/
├── wmt20
    ├── ...
    ├── vat_newstest2020-zhen-ref.en.txt
    └── vat_newstest2020-zhen-src.zh.txt
```

### Meta-Information of VAT
We also provide the meta-inforamtion of reserved data. Each json file contains the IDs of retained data in the original test set. For instance,  file  `wmt20/bert-r_filter-std60.json` describes:

```
{
	...
	"en-de": [4, 6, 10, 13, 14, 15, ...],
	"de-en": [0, 3, 4, 5, 7, 9, ...],
	...
}
```

### Reproduce & Create VAT
The reported results in the paper were produced by single NVIDIA GeForce 1080Ti card.

We will keep updating the code and related documentation after the response.

#### Requirements
* [sacreBLEU](https://github.com/mjpost/sacrebleu) version >= 1.4.14
* [BLEURT](https://github.com/google-research/bleurt) version >= 0.0.2
* [COMET](https://github.com/Unbabel/COMET) version >= 0.1.0
* [BERTScore](https://github.com/Tiiiger/bert_score) version >= 0.3.7 (hug_trans==4.2.1)
* [PyTorch](http://pytorch.org/) version >= 1.5.1
* Python version >= 3.8
* CUDA & cudatoolkit >= 10.1

Note: the minimal version is for reproducing the results

#### Pipeline
1. Use `score_xxx.py` to generate the CSV files that stores the sentence-level scores evaluated by the corresponding metrics. For example, evaluating all the WMT20 submissions of all the language pairs using BERTScore:
	```shell
	CUDA_VISIBLE_DEVICES=0 python score_bert.py -b 128 -s -r dummy -c dummy --rescale_with_baseline \
		--hypos-dir ${WMT_DATA_PATH}/system-outputs \
		--refs-dir ${WMT_DATA_PATH}/references \
		--scores-dir ${WMT_DATA_PATH}/results/system-level/scores_ALL \
		--testset-name newstest2020 --score-dump wmt20-bertscore.csv
	```
	(Alternative Option) You can use your implementation for dumping the scores given by the metrics. But the CSV header should contain:
	```csv
	,TESTSET,LP,ID,METRIC,SYS,SCORE
	```
2. Use `cal_filtering.py` to filter the test set based on the score warehouse calculated in the last step. For example, 
	```shell
	python cal_filtering.py --score-dump wmt20-bertscore.csv --output VAT_meta/wmt20-test/ --filter-per 60
	```
	It will produce the json files which contain the IDs of reserved sentences.
 
### Statistics of VAT (References)

| Benchmark | Translation Direction | # Sentences | # Words | # Vocabulary |
| :-------: | :-------------------: | :--------: | :-----: | :--------------: |
|   wmt20   |         km-en         |    928     |  17170  |       3645       |
|   wmt20   |         cs-en         |    266     |  12568  |       3502       |
|   wmt20   |         en-de         |    567     |  21336  |       5945       |
|   wmt20   |         ja-en         |    397     |  10526  |       3063       |
|   wmt20   |         ps-en         |    1088    |  20296  |       4303       |
|   wmt20   |         en-zh         |    567     |  18224  |       5019       |
|   wmt20   |         en-ta         |    400     |  7809   |       4028       |
|   wmt20   |         de-en         |    314     |  16083  |       4046       |
|   wmt20   |         zh-en         |    800     |  35132  |       6457       |
|   wmt20   |         en-ja         |    400     |  12718  |       2969       |
|   wmt20   |         en-cs         |    567     |  16579  |       6391       |
|   wmt20   |         en-pl         |    400     |  8423   |       3834       |
|   wmt20   |         en-ru         |    801     |  17446  |       6877       |
|   wmt20   |         pl-en         |    400     |  7394   |       2399       |
|   wmt20   |         iu-en         |    1188    |  23494  |       3876       |
|   wmt20   |         ru-en         |    396     |  6966   |       2330       |
|   wmt20   |         ta-en         |    399     |  7427   |       2148       |
|   wmt19   |         zh-en         |    800     |  36739  |       6168       |
|   wmt19   |         en-cs         |    799     |  15433  |       6111       |
|   wmt19   |         de-en         |    800     |  15219  |       4222       |
|   wmt19   |         en-gu         |    399     |  8494   |       3548       |
|   wmt19   |         fr-de         |    680     |  12616  |       3698       |
|   wmt19   |         en-zh         |    799     |  20230  |       5547       |
|   wmt19   |         fi-en         |    798     |  13759  |       3555       |
|   wmt19   |         en-fi         |    799     |  13303  |       6149       |
|   wmt19   |         kk-en         |    400     |  9283   |       2584       |
|   wmt19   |         de-cs         |    799     |  15080  |       6166       |
|   wmt19   |         lt-en         |    400     |  10474  |       2874       |
|   wmt19   |         en-lt         |    399     |  7251   |       3364       |
|   wmt19   |         ru-en         |    800     |  14693  |       3817       |
|   wmt19   |         en-kk         |    399     |  6411   |       3252       |
|   wmt19   |         en-ru         |    799     |  16393  |       6125       |
|   wmt19   |         gu-en         |    406     |  8061   |       2434       |
|   wmt19   |         de-fr         |    680     |  16181  |       3517       |
|   wmt19   |         en-de         |    799     |  18946  |       5340       |
|   wmt18   |         en-cs         |    1193    |  19552  |       7926       |
|   wmt18   |         cs-en         |    1193    |  23439  |       5453       |
|   wmt18   |         en-fi         |    1200    |  16239  |       7696       |
|   wmt18   |         en-tr         |    1200    |  19621  |       8613       |
|   wmt18   |         en-et         |    800     |  13034  |       6001       |
|   wmt18   |         ru-en         |    1200    |  26747  |       6045       |
|   wmt18   |         et-en         |    800     |  20045  |       5045       |
|   wmt18   |         tr-en         |    1200    |  25689  |       5955       |
|   wmt18   |         fi-en         |    1200    |  24912  |       5834       |
|   wmt18   |         zh-en         |    1592    |  42983  |       7985       |
|   wmt18   |         en-zh         |    1592    |  34796  |       8579       |
|   wmt18   |         en-ru         |    1200    |  22830  |       8679       |
|   wmt18   |         de-en         |    1199    |  28275  |       6487       |
|   wmt18   |         en-de         |    1199    |  25473  |       7130       |
|   wmt17   |         en-lv         |    800     |  14453  |       6161       |
|   wmt17   |         zh-en         |    800     |  20590  |       5149       |
|   wmt17   |         en-tr         |    1203    |  17612  |       7714       |
|   wmt17   |         lv-en         |    800     |  18653  |       4747       |
|   wmt17   |         en-de         |    1202    |  22055  |       6463       |
|   wmt17   |         ru-en         |    1200    |  24807  |       5790       |
|   wmt17   |         en-fi         |    1201    |  17284  |       7763       |
|   wmt17   |         tr-en         |    1203    |  23037  |       5387       |
|   wmt17   |         en-zh         |    800     |  18001  |       5629       |
|   wmt17   |         en-ru         |    1200    |  22251  |       8761       |
|   wmt17   |         fi-en         |    1201    |  23791  |       5300       |
|   wmt17   |         en-cs         |    1202    |  21278  |       8256       |
|   wmt17   |         de-en         |    1202    |  23838  |       5487       |
|   wmt17   |         cs-en         |    1202    |  22707  |       5310       |
|   wmt16   |         tr-en         |    1200    |  19225  |       4823       |
|   wmt16   |         ru-en         |    1199    |  23010  |       5442       |
|   wmt16   |         ro-en         |    800     |  16200  |       3968       |
|   wmt16   |         de-en         |    1200    |  22612  |       5511       |
|   wmt16   |         en-ru         |    1199    |  20233  |       7872       |
|   wmt16   |         fi-en         |    1200    |  20744  |       5176       |
|   wmt16   |         cs-en         |    1200    |  23235  |       5324       |
