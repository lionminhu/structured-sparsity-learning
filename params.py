class Params():
    def __init__(self):
        # Number of training epochs for pretraining
        self.num_pretrain_epochs = 100

        # Number of training epochs for sparse learning
        self.num_sparse_train_epochs = 10

        # Learning rate for optimizer
        self.learning_rate = 0.001

        # Whether to use GPU or not
        self.use_gpu = True

        # Hyperparameters for structured sparsity learning
        self.ssl_hyperparams = {
            "wgt_decay": 5e-4,
            "lambda_n": 5e-2,
            "lambda_c": 5e-2,
            "lambda_s": 5e-2,
        }

        # Threshold below which a weight value should be counted as too low
        self.threshold = 1e-5

        # Structured sparse learning method. Either "filter_channel" or "shape"
        self.ssl_type = "filter_channel"
