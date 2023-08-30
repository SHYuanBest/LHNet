# --- Imports --- #
import time
import torch
import argparse
import torch.nn as nn
from torch.utils.data import DataLoader
from Preprocess import ValData_test
from LHNet import LHNet
from utils import validation_PSNR, generate_filelist
import os

# --- Parse hyper-parameters  --- #
"""Implementation of LHNet from Shenghai Yuan et al. (ACM MM 2023)."""
parser = argparse.ArgumentParser(description='PyTorch implementation of LHNet from Shenghai Yuan et al. (ACM MM 2023)')
parser.add_argument('-d', '--dataset-name', help='name of dataset',choices=['NH', 'Dense', 'ITS','OTS'], default='Dense')
parser.add_argument('-t', '--data-dir', help='datasets path', default='../datasets')
parser.add_argument('-c', '--ckpts-dir', help='ckpts path', default=None)
parser.add_argument('-val_batch_size', help='Set the validation/test batch size', default=1, type=int)
args = parser.parse_args()

val_batch_size = args.val_batch_size
dataset_name = args.dataset_name
val_data_dir = args.data_dir
ckpts_dir = args.ckpts_dir

# --- Gpu device --- #
device_ids =  [Id for Id in range(torch.cuda.device_count())]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- Validation data loader --- #
val_data_loader = DataLoader(ValData_test(dataset_name,val_data_dir), batch_size=val_batch_size, shuffle=False, num_workers=0)

# --- Define the network --- #
net = LHNet()

# --- Multi-GPU --- # 
net = net.to(device)
net = nn.DataParallel(net, device_ids=device_ids)

if ckpts_dir:
    net.load_state_dict(torch.load(ckpts_dir), strict=False)

# --- Use the evaluation model in testing --- #
net.eval() 
print('--- Testing starts! ---') 
start_time = time.time()
val_psnr, val_ssim = validation_PSNR(net, val_data_loader, device, dataset_name, save_tag=True)
end_time = time.time() - start_time
print('val_psnr: {0:.2f}, val_ssim: {1:.4f}'.format(val_psnr, val_ssim))
print('validation time is {0:.4f}'.format(end_time))