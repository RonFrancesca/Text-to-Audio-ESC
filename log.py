from torch.utils.tensorboard import SummaryWriter


def get_writer(log_dir):
    writer = SummaryWriter(log_dir)
    return writer
