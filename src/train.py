from torch.utils.data import DataLoader
from optimization import LHNet_optimization
from argparse import ArgumentParser

from Preprocess import TrainData_train, ValData_train

def parse_args():
    """Implementation of LHNet from Shenghai Yuan et al. (ACM MM 2023)."""
    """Command-line argument parser for training."""

    # New parser
    parser = ArgumentParser(description='PyTorch implementation of LHNet from Shenghai Yuan et al. (ACM MM 2023)')

    # Training hyperparameters
    parser.add_argument('-lr', '--learning-rate', help='learning rate', default=0.0001, type=float)
    parser.add_argument('-a', '--adam', help='adam parameters', nargs='+', default=[0.9, 0.99, 1e-8], type=list)
    parser.add_argument('-b', '--batch-size', help='minibatch size', default=2, type=int)
    parser.add_argument('-e', '--nb-epochs', help='number of epochs', default=100, type=int)
    parser.add_argument('-l', '--loss', help='loss function', choices=['l1', 'l2'], default='l1', type=str)
    parser.add_argument('--cuda', help='use cuda', default=True, action='store_true')
    parser.add_argument('--plot-stats', help='plot stats after every epoch', action='store_true')
    parser.add_argument('-s', '--seed', help='fix random seed', type=int)
    parser.add_argument('--stage-of-dis', help='Interval epochs for training the discriminator', default=2, type=int)
    parser.add_argument('--threshold-of-dis', help='Freeze the discriminator for the required number of epochs', default=20, type=int)
    parser.add_argument('-c', '--crop-size', help='random crop size', default=224, type=int)

    # Data parameters
    parser.add_argument('-d', '--dataset-name', help='name of dataset',choices=['NH', 'Dense', 'OTS', 'ITS'], default='Dense')
    parser.add_argument('-t', '--data-dir', help='datasets path', default='../datasets')
    parser.add_argument('--ckpt-save-path', help='checkpoint save path', default='./../weights')
    parser.add_argument('--ckpt-overwrite', help='overwrite model checkpoint on save', action='store_true')
    parser.add_argument('--ckpt-load-path', help='start training with a pretrained model',default=None)
    parser.add_argument('--ckpt-dis-load-path', help='start training with a pretrained Discriminator model',default=None)
    parser.add_argument('--report-interval', help='batch report interval', default=1, type=int)
    parser.add_argument('-ts', '--train-size',nargs='+', help='size of train dataset',default=[512,512], type=int) 
    parser.add_argument('-vs', '--valid-size',nargs='+', help='size of valid dataset',default=[512,512], type=int)  


 
    return parser.parse_args()

 
if __name__ == '__main__':
    params = parse_args()

    train_loader = DataLoader(TrainData_train(params.dataset_name,params.train_size, params.data_dir), batch_size=params.batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(ValData_train(params.dataset_name,params.valid_size,params.data_dir), batch_size=params.batch_size, shuffle=False, num_workers=0)

    # Initialize model and train 
    LHNet = LHNet_optimization(params, trainable=True)
    LHNet.train(train_loader, valid_loader)
