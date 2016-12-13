Transfer learning
====
Using source domain knowledge to improve target domain recommandation results
. This project is implemented by tensroflow and libMF

##Environment
Mac os x, tensorflow 0.11.0

##Usage

compile the libmf 
```
$ cd libmf-2.01
$ make
$ cd ..
```

split the validation set from training data
this python code will create a directory with validation data (./valid)
```
$ python split_valid.py [data_directory] [cross_validation_size]
```

if the test.txt in the directory contain answer, then valid = 1 (default is 0)
```
$ python mapping.py --data_dir [data_directory] --valid [1/0]
```

use the mapping and the source data to learn a matrix factorization
```
$ python mf.py --data_dir [data_directory] --valid [1/0] --iter [iterations]
```

##example:

```
$ python split_valid.py ./test1 5 
$ python mapping.py --data_dir ./valid --valid 1
$ python mf.py --data_dir ./valid --valid 1 --iter 30
```

##Performance

| 5-cross_validation | MF | Transferred MF |
| :---: |:---:| :---:|
| test1 | RMSE = 0.18502 | RMSE = 0.18008 |
| test2 | RMSE = 0.18548 | RMSE = 0.17934 |
| test3 | RMSE = 1.35124 | RMSE = 1.25096 |

#Data

1. test1

   target and source data are exactly same item and user but with different id mapping

2. test2 

   disjoin user and item (items in the same domain)

3. test3 

   disjoin user and item (totaly different domain)



