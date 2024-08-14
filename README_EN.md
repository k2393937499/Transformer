# Transformer
Transformer: An Easy Implementation of Transformer Using Pytorch 2.3.0 and Torchtext 0.18.0

## Project Features
- Optimize structure of [Annotated Transformer](https://github.com/harvardnlp/annotated-transformer/), easy to learn the implementation of each module of Transformer
- Add new Dataloader module in `dataset.py`, which use intuitively implementation similar to general computer vision projects
- Using new versions of torch and torchtext, avoid using some function that are not in use anymore

## Environment Setup
- `git clone https://github.com/k2393937499/Transformer.git`
- `cd Transformer`
- `conda create -n Transformer python=3.10`
- `conda activate Transformer`
- `pip install -r requirements.txt`

## File Modification
It is necessary to make some modifications to the contents of torch-related libraries, it can be finished using VS Code or vim.
- `vim /home/*user*/miniconda3/envs/Transformer/lib/python3.10/site-packages/torch/utils/data/datapipes/utils/common.py`

Add `DILL_AVAILABLE = dill_available()` after `from torch.utils._import_utils import dill_available`（Detials: [https://stackoverflow.com/questions/78537817/importerror-cannot-import-name-dill-available](https://stackoverflow.com/questions/78537817/importerror-cannot-import-name-dill-available)）
- `vim /home/*user*/miniconda3/envs/Transformer/lib/python3.10/site-packages/torchtext/datasets/multi30k.py`
Modify `URL MD5 _PREFIX` as follows（Detials: [https://github.com/pytorch/text/issues/1756](https://github.com/pytorch/text/issues/1756)）：

```
URL = {
    "train": r"https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/training.tar.gz",
    "valid": r"https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/validation.tar.gz",
    "test": r"https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/mmt_task1_test2016.tar.gz",
}

MD5 = {
    "train": "20140d013d05dd9a72dfde46478663ba05737ce983f478f960c1123c6671be5e",
    "valid": "a7aa20e9ebd5ba5adce7909498b94410996040857154dab029851af3a866da8c",
    "test": "876a95a689a2a20b243666951149fd42d9bfd57cbbf8cd2c79d3465451564dd2",
}

_PREFIX = {
    "train": "train",
    "valid": "val",
    "test": "test2016",
}
```

## File Structure
```
  ├── saves: path to save checkpoints
  ├── dataset.py: dataset to load Multi30k
  ├── draw.py: read logs and draw loss and bleu curves
  ├── loss.py: loss computation module
  ├── model.py: Transformer model code
  ├── test.py: test the performences of trained model, including loss, bleu score and examples of translation.
  ├── train_eval_utils.py: Dependencies for training and validation
  └── train.py: training script
```

## Quick Start
- Use `python train.py` to train the model, checkpoints will be saved in `saves/`，log will be saved to root path，log file includes loss and bleu score of each epoch
- Use `python test.py` test model's performence in testset and check the translation examples.
- Use `python draw.py` draw loss and bleu curves during traing

## Note
- batchsize can be modified in dataloader(such as`train_dataloader = DataLoader(train_data, batch_size=16)`).
- Computation of bleu score is implementated using lib evaluate from Hugging Face, the module is loaded online, make sure your device is online（Detials: [https://huggingface.co/spaces/evaluate-metric/bleu](https://huggingface.co/spaces/evaluate-metric/bleu)）
- This project share the same weights trained by Annotated Transformer using `AnnotatedTransformer.ipynb`, you can combine them to study

## Acknowledgements
This Project Mainly Refers to the Following Repository
[https://github.com/harvardnlp/annotated-transformer/](https://github.com/harvardnlp/annotated-transformer/)