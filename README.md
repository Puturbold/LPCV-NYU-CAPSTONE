<img src="docs/NYC_MOCT.png" alt="NYC MOCT Logos" width="250">
<img src="docs/nyc-dot-logo.png" alt="NYC DOT Logo" width="250">
<img src="docs/NYU_CUSP_logo.png" alt="NYU CUSP logo" width="250">

## Features

Counts people on the street

## Motivation


## Examples


## Comparison to other trackers

## Benchmarks

[MOT17](https://motchallenge.net/data/MOT17/) results obtained using [motmetrics4norfair](https://github.com/tryolabs/norfair/tree/master/demos/motmetrics4norfair) demo script. Hyperparameters were tuned for reaching a high `MOTA` on this dataset. A more balanced set of hyperparameters, like the default ones used in the other demos, is recommended for production.

|                | Rcll  | Prcn  | GT    MT   PT   ML  | FP     FN      IDs   FM   | MOTA   MOTP  |
|:--------------:|:-----:|:-----:|:-------------------:|:-------------------------:|:------------:|
| MOT17-13-DPM   | 18.0% | 83.5% | 110   5    28   77  | 416    9543    120   125  | 13.4%  26.8% |
| MOT17-04-FRCNN | 56.3% | 93.2% | 83    18   43   22  | 1962   20778   90    104  | 52.0%  10.7% |
| MOT17-11-FRCNN | 61.5% | 93.1% | 75    19   33   23  | 431    3631    64    61   | 56.3%  10.1% |
| MOT17-04-SDP   | 77.6% | 97.4% | 83    48   26   9   | 1001   10672   225   254  | 75.0%  13.2% |
| MOT17-13-SDP   | 57.0% | 82.1% | 110   45   28   37  | 1444   5008    229   209  | 42.6%  20.1% |
| MOT17-05-DPM   | 38.0% | 82.2% | 133   10   58   65  | 570    4291    96    96   | 28.3%  24.2% |
| MOT17-09-DPM   | 59.9% | 75.4% | 26    4    18   4   | 1042   2137    119   113  | 38.1%  26.2% |
| MOT17-10-DPM   | 37.3% | 84.6% | 57    6    20   31  | 871    8051    127   154  | 29.5%  24.8% |
| MOT17-02-SDP   | 51.0% | 76.1% | 62    11   39   12  | 2979   9103    268   284  | 33.5%  18.2% |
| MOT17-11-DPM   | 54.2% | 84.5% | 75    12   24   39  | 935    4321    88    64   | 43.4%  21.7% |
| MOT17-09-FRCNN | 58.6% | 98.5% | 26    7    17   2   | 49     2207    40    39   | 56.9%   9.5% |
| MOT17-11-SDP   | 75.8% | 91.1% | 75    34   30   11  | 697    2285    103   101  | 67.3%  14.0% |
| MOT17-02-FRCNN | 36.6% | 79.7% | 62    7    26   29  | 1736   11783   119   131  | 26.6%  13.4% |
| MOT17-05-FRCNN | 54.7% | 89.2% | 133   24   68   41  | 457    3136    95    96   | 46.7%  18.1% |
| MOT17-04-DPM   | 42.5% | 83.6% | 83    7    44   32  | 3965   27336   401   420  | 33.3%  21.0% |
| MOT17-10-SDP   | 74.2% | 88.1% | 57    30   24   3   | 1292   3316    308   289  | 61.7%  19.8% |
| MOT17-10-FRCNN | 61.0% | 75.8% | 57    16   35   6   | 2504   5013    319   313  | 39.0%  17.3% |
| MOT17-09-SDP   | 67.6% | 94.6% | 26    12   14   0   | 204    1726    52    55   | 62.8%  13.0% |
| MOT17-02-DPM   | 20.2% | 81.6% | 62    5    14   43  | 843    14834   111   112  | 15.0%  24.6% |
| MOT17-13-FRCNN | 58.8% | 73.6% | 110   29   57   24  | 2455   4802    371   366  | 34.5%  18.5% |
| MOT17-05-SDP   | 66.7% | 87.6% | 133   32   81   20  | 653    2301    134   140  | 55.4%  16.5% |
| OVERALL        | 53.6% | 87.2% | 1638  381  727  530 | 26506  156274  3479  3526 | 44.7%  16.3% |


## Citing Norfair

This pestrian counter is based on Norfair. A multi-object tracker built and maintained by Tryolabs.

https://github.com/tryolabs/norfair

Joaquín Alori, Alan Descoins, KotaYuhara, David, facundo-lezama, Braulio Ríos, fatih, shafu.eth, Agustín Castro, & David Huh. (2022). tryolabs/norfair: v1.0.0 (v1.0.0). Zenodo. https://doi.org/10.5281/zenodo.6596178