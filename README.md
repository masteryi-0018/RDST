# RDST

This repository contains the official implementation for RDST



### Citation

If you find our work useful in your research, please cite



### Enviroment

- Python 3.8.11
- Pytorch 1.8.0
- TensorboardX
- yaml, numpy, tqdm, imageio



### Train

the dataset for train is DIV2K and Flickr2k. Download and put them into the directory you like, remember to change the path in `/configs`.

```python
python train_liif.py
```



### Test

the benchmark for test is Set5, Set14, B100, Urban100 and Manga109. Download and put them into the directory you like, remember to change the path in `/configs`.

- for CNN based models:

```python
sh test-benchmark.sh
```

- for Transformer based models:

```pyth
sh test-benchmark_swin.sh
```



### Acknowledgements

This code is built on [LIIF](https://github.com/yinboc/liif) and [SwinIR](https://github.com/JingyunLiang/SwinIR). We thank the authors for sharing their codes of LIIF and SwinIR.

