class NetworkData:

    def __init__(self, config):

        self.lr = config["lr"]
        self.kernel_size = config["kernel_size"]
        self.stride = config["stride"]
        self.padding = config["padding"]
        self.dropout_rate = config["dropout_rate"]
        self.maxp_ks = config["maxp_ks"]
        self.maxp_stride = config["maxp_stride"]
        self.nclass = config["nclass"]
        self.in_channel = config["in_channel"]
        self.out_channel = config["out_channel"]
        self.dense_in = config["dense_in"]
        self.dense_out = config["dense_out"]
