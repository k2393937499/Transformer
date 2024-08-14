# Transformer
Transformer: An Easy Implementation of Transformer Using Pytorch 2.3.0 and Torchtext 0.18.0

[EN_ver](README_EN.md)

[视频链接](https://www.bilibili.com/video/BV1ZPepeMER9)

## 项目特点
- 优化Annotated Transformer项目的结构，便于学习Transformer各模块的实现
- 加入新的数据集加载代码`dataset.py`，使用了与一般计算机视觉项目类似的实现，更加直观
- 使用新版本torch与torchtext，避免使用一些已经移除的函数

## 环境配置
- `git clone https://github.com/k2393937499/Transformer.git`
- `cd Transformer`
- `conda create -n Transformer python=3.10`
- `conda activate Transformer`
- `pip install -r requirements.txt`

## 文件修改
需要对torch相关库的内容进行一定修改，可通过VS Code跳转修改，Linux用户可通过vim修改。
- `vim /home/*user*/miniconda3/envs/Transformer/lib/python3.10/site-packages/torch/utils/data/datapipes/utils/common.py`

在`from torch.utils._import_utils import dill_available`后换行添加`DILL_AVAILABLE = dill_available()`（详见[https://stackoverflow.com/questions/78537817/importerror-cannot-import-name-dill-available](https://stackoverflow.com/questions/78537817/importerror-cannot-import-name-dill-available)）
- `vim /home/*user*/miniconda3/envs/Transformer/lib/python3.10/site-packages/torchtext/datasets/multi30k.py`
修改`URL MD5 _PREFIX`为如下内容（详见[https://github.com/pytorch/text/issues/1756](https://github.com/pytorch/text/issues/1756)）：

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

## 文件结构
```
  ├── saves: 权重保存路径
  ├── dataset.py: 自定义dataset读取Multi30k数据集
  ├── draw.py: 读取日志绘制loss与bleu曲线
  ├── loss.py: Loss计算模块
  ├── model.py: Transformer模型代码
  ├── test.py: 测试权重的表现，包括loss、bleu以及翻译示例
  ├── train_eval_utils.py: 训练、验证的依赖
  └── train.py: 训练权重需要的脚本
```

## 快速开始
- 使用`python train.py`进行训练，权重会被保存到`saves/`路径，日志将保存到根目录，日志内容包括每一代的loss以及bleu分数
- 使用`python test.py`查看模型在测试集的表现以及翻译示例
- 使用`python draw.py`绘制模型的loss、bleu变化

## 注意事项
- 训练的batchsize可自行在加载数据集代码（例如`train_dataloader = DataLoader(train_data, batch_size=16)`）中进行修改
- 计算bleu分数使用Hugging Face的evaluate库实现，其中计算bleu的模块需要联网加载，打开加速器以保证加载正常进行（详见[https://huggingface.co/spaces/evaluate-metric/bleu](https://huggingface.co/spaces/evaluate-metric/bleu)）
- 该项目训练得到的权重与Annotated Transformer项目`AnnotatedTransformer.ipynb`文件训练得到的权重是可以共享的，可以结合两个项目进行学习

## 该项目主要参考了以下仓库
- [https://github.com/harvardnlp/annotated-transformer/](https://github.com/harvardnlp/annotated-transformer/)
