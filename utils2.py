import os
import argparse
import torch

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def set_device(gpu=None):
    if gpu is not None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    try:
        print(f'Available GPUs Index : {os.environ["CUDA_VISIBLE_DEVICES"]}')
    except KeyError:
        print('No GPU available, using CPU ... ')
    return torch.device('cuda') if torch.cuda.device_count() >= 1 else torch.device('cpu')