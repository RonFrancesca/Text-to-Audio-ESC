class TrainClass:

    def __init__(self, config):

        self.runs = config["runs"]
        self.model = config["model"]
        self.data_type = config["data_type"]
        self.data_aug = config["data_aug"]
        self.n_rep = config["n_rep"]
        self.normalization = config["normalization"]
        self.batch_size = config["batch_size"]
        self.batch_size_val = config["batch_size_val"]
        self.batch_size_test = config["batch_size_test"]
        self.n_epochs = config["n_epochs"]
        self.val_thresholds = config["val_thresholds"]
        self.testing_mode = config["testing_mode"]
    
        if 1 <= config["replace_n_folder"] <= 8:
            self.replace_n_folder = config["replace_n_folder"]
        elif config["replace_n_folder"] == 0 and config["data_type"]:
            print("Using only original data")
        else:
            print("The number of folders to replace is not valid")
            exit()
