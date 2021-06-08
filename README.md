# Variance-Aware-MT-Test-Sets
Variance-Aware Machine Translation Test Sets
(Under review)

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
We will update the code of this part after the final decision.

### License
See `LICENSE`. We follow the data licensing plan as the same as the WMT benchmark. 


