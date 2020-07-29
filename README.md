# Structured Sparsity Learning
Attempt to implement Structured Sparsity Learning from [Wen et al., 2016, "Learning Structured Sparsity in Deep Neural Networks"](https://arxiv.org/abs/1608.03665).

The source code files are organized as below:
- `dataloader.py`: Implements train and test dataset loader for CIFAR10
- `models.py`: Defines model to be used for training and inference
- `params.py`: Defines arguments and hyperparameters
- `pretrain.py`: Generates a model pretrained on CIFAR10
- `ssl.py`: Implements loss functions for Structured Sparsity Learning (SSL)
- `train_sparse.py`: Fine-tunes the pretrained model based on SSL loss function
- `filter_channel_result.txt`: The result of running filter-wise, channel-wise structured sparsity
- `shape_result.txt`: The result of running shape-wise structured sparsity

Before running the repository code, first check the `params.py` to check whether GPU is used or not. By default, the code is run with GPU.

To run this, enter `python3 pretrain.py` to download the CIFAR10 dataset, train and evaluate the network (defined in `models.py`), and save the model to `saved_model.pth` file.

SSL can be applied on the pretrained model by running `python3 train_sparse.py`. Please make sure to select the desired type of SSL in the `params.py` beforehand. By running `train_sparse.py`, the initial and final percentage of sparse weights will be displayed, as well as the performance (in accuracy) of the model.