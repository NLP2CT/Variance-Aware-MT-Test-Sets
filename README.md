# Variance-Aware-MT-Test-Sets
Variance-Aware Machine Translation Test Sets

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
We will update the code of this part after the final decision.

### Statistics of VAT
#### Grouped by Released Year

|           | **# Instances** | **# Tokens** | **# Words** |
| --------: | :-------------: | :----------: | :---------: |
| **WMT16** |      15996      |    277071    |    53406    |
| **WMT17** |      30432      |    583164    |    74035    |
| **WMT18** |      33536      |    691481    |    77989    |
| **WMT19** |      23510      |    503969    |    76279    |
| **WMT20** |      19756      |    484575    |    78226    |



#### Grouped by Translation Direction

|           | **#  Instances** | **# Tokens** | **# Words** |
| --------: | :--------------: | :----------: | :---------: |
| **en-cs** |       7522       |    154955    |    31314    |
| **en-de** |       7534       |    175866    |    27839    |
| **en-et** |       1600       |    30031     |    9877     |
| **en-fi** |       6400       |    111187    |    26233    |
| **en-gu** |       798        |    17814     |    6475     |
| **en-ja** |       800        |    22360     |    6005     |
| **en-kk** |       798        |    14436     |    5744     |
| **en-lt** |       798        |    15741     |    5738     |
| **en-lv** |       1600       |    31916     |    10441    |
| **en-pl** |       800        |    17115     |    6239     |
| **en-ru** |      10398       |    210241    |    40888    |
| **en-ta** |       800        |    17778     |    7022     |
| **en-tr** |       4806       |    86227     |    21519    |
| **en-zh** |       7516       |    197199    |    14504    |
| **cs-en** |       7722       |    151172    |    30916    |
| **de-cs** |       1598       |    32192     |    10585    |
| **de-en** |       9430       |    206105    |    32027    |
| **de-fr** |       1360       |    28984     |    6818     |
| **et-en** |       1600       |    35986     |    11199    |
| **fi-en** |       8798       |    143184    |    33107    |
| **fr-de** |       1360       |    28507     |    6768     |
| **gu-en** |       812        |    14957     |    5385     |
| **iu-en** |       2376       |    36985     |    9552     |
| **ja-en** |       794        |    21484     |    5842     |
| **kk-en** |       800        |    16461     |    5956     |
| **km-en** |       1856       |    23222     |    7042     |
| **lt-en** |       800        |    18500     |    6356     |
| **lv-en** |       1600       |    33648     |    10751    |
| **pl-en** |       800        |    13632     |    5086     |
| **ps-en** |       2176       |    42374     |    10087    |
| **ro-en** |       1600       |    32484     |    8352     |
| **ru-en** |       9590       |    180130    |    37108    |
| **ta-en** |       798        |    13340     |    5041     |
| **tr-en** |       7206       |    120336    |    28226    |
| **zh-en** |       7984       |    243711    |    15566    |
