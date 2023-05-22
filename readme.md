
# BinCola

Official code of **BinCola: Diversity-sensitive Contrastive Learning for Binary Code Similarity Detection**

![Illustrating the performance of the proposed jTrans](/figures/TOP1-poolsize.png)

## Get Started

### Environmental preparation

- Python 3.8+
- PyTorch 1.10+
- CUDA 10.2+
- IDA pro 7.5+

### Create virtual environment

```python
conda create -n env_name python=3.8 # create
Linux: source activate env_nam # activate
Linux: source deactivate env_name
conda info -e # check env
conda remove -n env_name --all # delete
# dependent libraries
pip install torch torchvision torchaudio tensorboard numpy pandas coloredlogs matplotlib PyYAML seaborn sklearn tqdm info-nce-pytorch
```

### Datasets

- [Mix_options_pickle.zip](https://drive.google.com/file/d/17Svo5WHn0Y-jgqYlm4JmNG1b5GarkTXV/view?usp=sharing)
- [Obfus_pickle.zip](https://drive.google.com/file/d/1H5hzcYxya1-3fTl0WoHIlPMzJ7T0jYjO/view?usp=sharing)
- [Vulner_Search_Bin.zip](https://drive.google.com/file/d/1LfbstrsZjfhDZjNHSPRveDu1k9p9848e/view?usp=sharing)
- [CVE_Search_Bin.zip](https://drive.google.com/file/d/1WQXL809uiwWcLevQFVrbXthaqlbjI05I/view?usp=sharing)

### Models

- [model_save_test](https://drive.google.com/file/d/1M3rPmlIawCPPrCbQ9wAfGKA7uKwEck6p/view?usp=sharing)

### Quick Start

**a. Get code of BinCola.**

```python
git clone https://github.com/BinCola-Future/BinCola.git && cd BinCola
```

**b. Preprocess binary files**

```python
python IDA_Process/run.py \
    --src_folder "binary file directory" 
    --out_folder "The parsed pickle file save directory"
    --ida_path "ida tool path"
    --script_path "ida script path"
    --log_folder "log result save folder"
```

**c. Train and evaluate the model**

```python
cd core/
# Generate a list of train files
python preprocess_bcsa.py \
    --src_folder "The parsed pickle file save directory"
    --out_folder "training file list save folder"
# Train the model
# train.py - one positive sample
# train_select.py - set positive samples num
python train.py or train_select.py \
    --debug # fix random seed
    --train # Set to train mode, otherwise evaluation mode
    --input_list "training file list save path"
    --config_folder "config folder"
    --log_out "result save folder"
```

**d. Evaluate the model**

```python
cd core/
# Generate a list of eval files
python preprocess_bcsa.py \
    --src_folder "The parsed pickle file save directory"
    --out_folder "eval file list save folder"
# Evaluate the model
python eval.py \
    --debug # fix random seed
    --input_list "training file list save path"
    --config_folder "config folder"
    --log_out "result save folder"
    --model_path "eval model path"
```
