import torch
import torch.nn as nn
from torch.optim import Adam, lr_scheduler
from torch.autograd import Variable
import torch.autograd as autograd
from LHNet import LHNet
from utils_modules import Discriminator
from utils import *
import torchvision
import os
import json

class LHNet_optimization(object):
    """Implementation of LHNet from Shenghai Yuan et al. (ACM MM 2023)."""

    def __init__(self, params, trainable):
        """Initializes model."""
        self.p = params
        self.trainable = trainable      # True or False
        self._compile()
        self.max_psnr = 0

    def _compile(self):
        
        self.model = LHNet()
        self.discriminator = Discriminator(img_size=self.p.train_size[0])
        self.ema = EMA(self.model, 0.999, True)
        self.ema1 = EMA(self.discriminator, 0.999, True)
        self.ema.register()
        self.ema1.register()

        # Set optimizer and loss, if in training mode
        if self.trainable:
            self.optim = Adam(self.model.parameters(),
                              lr=self.p.learning_rate,
                              betas=self.p.adam[:2],
                              eps=self.p.adam[2])

            self.optim_dis = Adam(self.discriminator.parameters(),
                              lr=self.p.learning_rate*5,
                              betas=self.p.adam[:2],
                              eps=self.p.adam[2])

            # Learning rate adjustment
            self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optim,
                patience=self.p.nb_epochs/4, factor=0.5, verbose=True)
            self.scheduler_dis = lr_scheduler.ReduceLROnPlateau(self.optim_dis,
                patience=self.p.nb_epochs/4, factor=0.5, verbose=True)

            # Loss function
            if self.p.loss == 'l2':
                self.loss = nn.MSELoss()
            else:
                self.loss = nn.L1Loss()

            self.loss_dis = nn.MSELoss()

        # CUDA support
        self.use_cuda = torch.cuda.is_available() and self.p.cuda
        if self.use_cuda:
            self.model = self.model.cuda()
            self.discriminator = self.discriminator.cuda()
            if self.trainable:
                self.loss = self.loss.cuda()
                self.loss_dis = self.loss_dis.cuda()
        self.model = torch.nn.DataParallel(self.model)
        self.discriminator = torch.nn.DataParallel(self.discriminator)


    def _print_params(self):
        """Formats parameters to print when training."""

        print('Training parameters: ')
        self.p.cuda = self.use_cuda
        param_dict = vars(self.p)
        pretty = lambda x: x.replace('_', ' ').capitalize()
        print('\n'.join('  {} = {}'.format(pretty(k), str(v)) for k, v in param_dict.items()))
        print()


    def save_model(self, epoch, stats, first=False):
        """Saves model to files; can be overwritten at every epoch to save disk space."""

        # Create directory for model checkpoints, if nonexistent
        if first:
            ckpt_dir_name = f'{datetime.now():{self.p.dataset_name}-%m%d-%H%M}'
            if self.p.ckpt_overwrite:
                ckpt_dir_name = self.p.dataset_name

            self.ckpt_dir = os.path.join(self.p.ckpt_save_path, ckpt_dir_name)
            if not os.path.isdir(self.p.ckpt_save_path):
                os.mkdir(self.p.ckpt_save_path)
            if not os.path.isdir(self.ckpt_dir):
                os.mkdir(self.ckpt_dir)

        # Save checkpoint dictionary
        if self.p.ckpt_overwrite:
            fname_unet = '{}/LHNet-{}-gen'.format(self.ckpt_dir, self.p.dataset_name)
            fname_disc = '{}/LHNet-{}-dis.pt'.format(self.ckpt_dir, self.p.dataset_name)
        else:
            valid_loss = stats['valid_loss'][epoch]
            valid_loss_dis = stats['valid_dis_loss'][epoch]
            fname_unet = '{}/LHNet-gen-epoch{}-{:>1.5f}'.format(self.ckpt_dir, epoch + 1, valid_loss)
            fname_disc = '{}/LHNet-dis-epoch{}-{:>1.5f}.pt'.format(self.ckpt_dir, epoch + 1, valid_loss_dis)

        if epoch == 0:
            self.max_psnr = stats['valid_psnr'][epoch]

        if epoch == 0 or (epoch != 0 and stats['valid_psnr'][epoch] > self.max_psnr):
            self.max_psnr =  stats['valid_psnr'][epoch] 
            print('Saving checkpoint to: {} and {}, valid_psnr: {}\n'.format(fname_unet, fname_disc, self.max_psnr))
            torch.save(self.model.state_dict(), fname_unet+'-best.pt')
            torch.save(self.discriminator.state_dict(), fname_disc)

        # Save stats to JSON
        fname_dict = '{}/LHNet-stats.json'.format(self.ckpt_dir)
        with open(fname_dict, 'w') as fp:
            json.dump(stats, fp, indent=2)

    def _on_epoch_end(self, stats, train_loss, train_dis_loss, epoch, epoch_start, valid_loader, iter_total):
        """Tracks and saves starts after each epoch."""
        # import pdb;pdb.set_trace()
        # Evaluate model on validation set
        print('\rTesting model on validation set... ', end='')
        epoch_time = time_elapsed_since(epoch_start)[0]
        valid_loss, valid_time, valid_psnr, valid_loss_dis, valid_time_dis = self.eval(valid_loader, iter_total)
        show_on_epoch_end(epoch_time, valid_time, valid_loss, valid_psnr, valid_loss_dis, valid_time_dis)

        # Decrease learning rate if plateau
        self.scheduler.step(valid_loss)
        self.scheduler_dis.step(valid_loss_dis)

        # Save checkpoint
        stats['train_loss'].append(train_loss)
        stats['valid_loss'].append(valid_loss)
        stats['valid_psnr'].append(valid_psnr)
        stats['train_dis_loss'].append(train_dis_loss)
        stats['valid_dis_loss'].append(valid_loss_dis)
        self.save_model(epoch, stats, epoch == 0)

        # Plot stats
        if self.p.plot_stats:
            loss_str = f'{self.p.loss.upper()} loss'
            plot_per_epoch(self.ckpt_dir, 'Valid loss', stats['valid_loss'], loss_str)
            plot_per_epoch(self.ckpt_dir, 'Valid PSNR', stats['valid_psnr'], 'PSNR (dB)')

    def eval(self, valid_loader,epoch):
        with torch.no_grad():
            self.model.train(False)
            self.discriminator.train(False)

            self.ema.apply_shadow()
            self.ema.restore()
            self.ema1.apply_shadow()
            self.ema1.restore()

            # Generator
            valid_start = datetime.now()
            loss_meter = AvgMeter()
            psnr_meter = AvgMeter()

            for batch_idx, (source, target,haze_name) in enumerate(valid_loader):
                if self.use_cuda:
                    source = source.cuda()
                    target = target.cuda()
                labels = 1
                # dehaze
                source_dehazed = self.model(source)

                # Update loss
                # loss = self.loss(source_dehazed, target)

                # recons_loss = torch.abs(source_dehazed-target).mean()

                fake_validity = self.discriminator(source_dehazed, labels)
                if epoch <= self.p.threshold_of_dis:
                    loss = self.loss(source_dehazed, target)
                else:
                    # MSA
                    loss = 0.6*self.loss(source_dehazed, target) + 0.4*self.loss_dis(fake_validity,torch.ones_like(fake_validity) - 0.2)
                    
                    # WGAN_GP
                    # loss = -torch.mean(fake_validity)

                loss_meter.update(loss.item())

                # Compute PSRN
                for i in range(source_dehazed.shape[0]):
                    # import pdb;pdb.set_trace()
                    source_dehazed = source_dehazed.cpu()
                    target = target.cpu()
                    psnr_meter.update(psnr(source_dehazed[i], target[i]).item())

            valid_loss = loss_meter.avg
            valid_time = time_elapsed_since(valid_start)[0] 
            psnr_avg = psnr_meter.avg


            # Discriminator
            valid_start_dis = datetime.now()
            loss_meter_dis = AvgMeter()

            for batch_idx, (source, target, haze_name) in enumerate(valid_loader):
                if self.use_cuda:
                    source = source.cuda()
                    target = target.cuda()
                labels = 1
                # dehaze
                source_dehazed = self.model(source)

                # Update loss

                real_validity = self.discriminator(target, labels)
                fake_validity = self.discriminator(source_dehazed.detach(), labels)

                # MSA
                haze_loss = self.loss_dis(fake_validity, torch.zeros_like(fake_validity) + 0.2)
                gt_loss = self.loss_dis(real_validity, torch.ones_like(real_validity) - 0.2)
                loss_dis = (haze_loss + gt_loss)
                    
                # # WGAN-GP
                # gradient_penalty = 0.1
                # loss_dis = -torch.mean(real_validity) + torch.mean(fake_validity) + 10 * gradient_penalty

                loss_meter_dis.update(loss_dis.item())

            valid_loss_dis = loss_meter_dis.avg
            valid_time_dis = time_elapsed_since(valid_start_dis)[0]

            return valid_loss, valid_time, psnr_avg, valid_loss_dis, valid_time_dis
 
    def compute_gradient_penalty(self,cuda=True, D=None, real_samples=None, fake_samples=None, center=0, LAMBDA=1):
        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake samples
        Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

        alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates = D(interpolates)

        # Get gradient w.r.t. interpolates
        gradients = autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones(d_interpolates.size()).to('cuda'),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - center) ** 2).mean() * LAMBDA
        return gradient_penalty


    def train(self, train_loader, valid_loader): 
        """Trains denoiser on training set."""  
 
        self.model.train(True)
        self.discriminator.train(True)

        if self.p.ckpt_load_path is not None:
            self.model.load_state_dict(torch.load(self.p.ckpt_load_path), strict=False)
            print('The pretrain generator is loaded.')
        if self.p.ckpt_dis_load_path is not None:
            self.discriminator.load_state_dict(torch.load(self.p.ckpt_dis_load_path))
            print('The pretrain discriminator is loaded.')

        self._print_params()
        num_batches = len(train_loader)
        assert num_batches % self.p.report_interval == 0, 'Report interval must divide total number of batches'

        # Dictionaries of tracked stats
        stats = {'dataset_name': self.p.dataset_name, 
                 'train_loss': [],
                 'valid_loss': [], 
                 'valid_psnr': [],
                 'train_dis_loss': [],
                 'valid_dis_loss': []}

        # Main training loop 
        train_start = datetime.now()

        iter_total = 0
        for epoch in range(self.p.nb_epochs): 
            print('EPOCH {:d} / {:d}'.format(epoch + 1, self.p.nb_epochs))

            # Some stats trackers
            epoch_start = datetime.now()
            train_loss_meter = AvgMeter()
            loss_meter = AvgMeter()
            time_meter = AvgMeter()

            loss_meter_dis = AvgMeter()
            train_loss_meter_dis = AvgMeter()


            # Minibatch SGD
            for batch_idx, (source, target) in enumerate(train_loader):
                labels = 1
                batch_start = datetime.now()
                progress_bar(batch_idx, num_batches, self.p.report_interval, loss_meter.val)
    
                # Generator
                source_dehazed = self.model(source)
                # recons_loss = torch.abs(source_dehazed-target).mean()
                fake_validity = self.discriminator(source_dehazed, labels)
                
                if self.use_cuda:
                    source = source.cuda()
                    target = target.cuda()
                    source_dehazed = source_dehazed.cuda()
                    fake_validity = fake_validity.cuda()
                                
                if iter_total <= self.p.threshold_of_dis:
                    g_loss = self.loss(source_dehazed, target)
                else:
                    # MSA
                    g_loss = 0.6*self.loss(source_dehazed, target) + 0.4*self.loss_dis(fake_validity,torch.ones_like(fake_validity) - 0.2)
                loss_meter.update(g_loss.item())
                self.optim.zero_grad()
                g_loss.backward()
                self.optim.step()
                self.ema.update()
                
                # Discriminator                
                if iter_total > self.p.threshold_of_dis and iter_total % self.p.stage_of_dis == 0:
                    gradient_penalty = self.compute_gradient_penalty(True, self.discriminator, target, source_dehazed.detach())

                    target = target + torch.randn_like(target)
                    source_dehazed = source_dehazed + torch.randn_like(source_dehazed)

                    real_validity = self.discriminator(target, labels)
                    fake_validity = self.discriminator(source_dehazed.detach(), labels)

                    # MSA
                    haze_loss = self.loss_dis(fake_validity, torch.zeros_like(fake_validity) + 0.2)
                    gt_loss = self.loss_dis(real_validity, torch.ones_like(real_validity) - 0.2)
                    d_loss = haze_loss + gt_loss + gradient_penalty
                    
                    # WGAN-GP
                    # d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + gradient_penalty
                    
                    loss_meter_dis.update(d_loss.item())
                    self.optim_dis.zero_grad()
                    d_loss.backward()
                    self.optim_dis.step()
                    self.ema1.update()

                # Report/update statistics
                time_meter.update(time_elapsed_since(batch_start)[1])
                if (batch_idx + 1) % self.p.report_interval == 0 and batch_idx:
                    show_on_report(batch_idx, num_batches, loss_meter.avg, time_meter.avg, loss_meter_dis.avg)
                    train_loss_meter.update(loss_meter.avg)
                    train_loss_meter_dis.update(loss_meter_dis.avg)
                    loss_meter_dis.reset()
                    loss_meter.reset()
                    time_meter.reset()
                iter_total = iter_total + 1

            # Epoch end, save and reset tracker
            self._on_epoch_end(stats, train_loss_meter.avg, train_loss_meter_dis.avg, epoch, epoch_start, valid_loader, iter_total)
            train_loss_meter.reset()
            train_loss_meter_dis.reset()

        train_elapsed = time_elapsed_since(train_start)[0]
        print('Training done! Total elapsed time: {}\n'.format(train_elapsed))
        print('The best psnr: {}\n'.format(self.max_psnr))


class EMA():
    def __init__(self, model, decay, cuda):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.use_cuda = cuda

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
 
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                if self.use_cuda:
                    param.data = param.data.cuda()
                    self.shadow[name] = self.shadow[name].cuda()
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
 
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]
 
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}