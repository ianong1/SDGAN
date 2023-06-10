# -*- coding:utf-8 -*-
import argparse
from base64 import encode
from email.policy import default
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import optim
import os
import time
import random
import torch.nn as nn
from model.encoder import Encoder
from model.decoder import Decoder
from model.discriminator_col import XuNet
import utils_col as utils
import pytorch_ssim


def main():
    parent_parser = argparse.ArgumentParser(description='Training of our nets')
    subparsers = parent_parser.add_subparsers(dest='command', help='Sub-parser for commands')

    new_run_parser = subparsers.add_parser('new', help='starts a new run')
    new_run_parser.add_argument('--train-data-dir', default='/public/wangxiangkun/chat-gan-grey/gen1000_7.4980/train', type=str,
                                help='The directory where the data for training is stored.')
    new_run_parser.add_argument('--valid-data-dir', default='/public/wangxiangkun/chat-gan-grey/gen1000_7.4980/valid', type=str,
                                help='The directory where the data for validation is stored.')
    new_run_parser.add_argument('--run-folder', type=str, default='checkpoint1',
                                help='The experiment folder where results are loggedpoint1.')
    new_run_parser.add_argument('--title', type=str, default='1',
                                help='The experiment name.')

    new_run_parser.add_argument('--size', default=512, type=int,
                                help='The size of the images (images are square so this is height and width).')
    new_run_parser.add_argument('--data-depth', default=1, type=int, help='The depth of the message.')

    new_run_parser.add_argument('--batch-size', type=int, help='The batch size.', default=8)
    new_run_parser.add_argument('--epochs', default=40, type=int, help='Number of epochs.')

    new_run_parser.add_argument('--gray', action='store_true', default=False,
                                help='Use gray-scale images.')
    new_run_parser.add_argument('--hidden-size', type=int, default=32,
                                help='Hidden channels in networks.')
    new_run_parser.add_argument('--tensorboard', action='store_true',
                                help='Use to switch on Tensorboard logging.')
    new_run_parser.add_argument('--seed', type=int, default=20,
                                help='Random seed.')
    new_run_parser.add_argument('--no-cuda', action='store_true', default=False,
                                help='Disables CUDA training.')
    new_run_parser.add_argument('--gpu', type=int, default=0,
                                help='Index of gpu used (default: 0).')
    new_run_parser.add_argument('--use-vgg', action='store_true', default=False,
                                help='Use VGG loss.')

    continue_parser = subparsers.add_parser('continue', help='Continue a previous run')
    continue_parser.add_argument('--continue-folder', type=str, required=True,
                                 help='The experiment folder where results are logged.')
    continue_parser.add_argument('--continue-checkpoint', type=str, required=True,
                                 help='The experiment folder where results are logged.')
    continue_parser.add_argument('--size', default=256, type=int,
                                 help='The size of the images (images are square so this is height and width).')
    continue_parser.add_argument('--data-depth', default=2, type=int, help='The depth of the message.')

    continue_parser.add_argument('--batch-size', type=int, help='The batch size.', default=16)
    continue_parser.add_argument('--epochs', default=40, type=int, help='Number of epochs.')

    continue_parser.add_argument('--title', type=str, required=True,
                                 help='The experiment name.')

    lambda_adv, lambda_g, lambda_p, lambda_m = 1, 1, 1, 100
    args = parent_parser.parse_args()
    use_discriminator = True
    continue_from_checkpoint = None
    if args.command == 'continue':
        log_dir = args.continue_folder
        checkpoints_dir = os.path.join(log_dir, 'checkpoints')
        continue_from_checkpoint = torch.load(args.continue_checkpoint)
    else:
        assert args.command == 'new'
        log_dir = os.path.join(args.run_folder, time.strftime("%Y-%m-%d--%H-%M-%S-") + args.title)
        checkpoints_dir = os.path.join(log_dir, 'checkpoints')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            os.makedirs(checkpoints_dir)
    train_csv_file = os.path.join(log_dir, args.title + '_train.csv')
    valid_csv_file = os.path.join(log_dir, args.title + '_valid.csv')

    # set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda') if args.cuda else torch.device('cpu')
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        torch.cuda.set_device(args.gpu)
        kwargs = {'num_workers': 0, 'pin_memory': False}
    else:
        kwargs = {}

    # Load Datasets
    print('---> Loading Datasets...')
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop((args.size, args.size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip()
    ])
    train_dataset = utils.Mydataset(args.train_data_dir)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True, shuffle=True, **kwargs)

    valid_transform = transforms.Compose([
        transforms.CenterCrop((args.size, args.size))
    ])
    valid_dataset = utils.Mydataset(args.valid_data_dir)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, drop_last=True, shuffle=False, **kwargs)

    # Load Models
    print('---> Constructing Network Architectures...')
    color_band = 1 if args.gray else 3
    encoder = Encoder(args.data_depth, args.hidden_size, color_band)
    decoder = Decoder(args.data_depth, args.hidden_size, color_band)
    discriminator = XuNet()


    # VGG for perceptual loss
    print('---> Constructing VGG-16 for Perceptual Loss...')
    vgg = utils.VGGLoss(3, 1, False)

    # Define Loss
    print('---> Defining Loss...')
    optimizer_coders = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()),
                                  lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    scheduler = optim.lr_scheduler.StepLR(optimizer_coders, step_size=10, gamma=0.1)
    optimizer_discriminator = optim.Adam(discriminator.parameters(), lr=1e-4, weight_decay=0)
    mse_loss = torch.nn.MSELoss()
    bce_loss = torch.nn.BCELoss()
    if args.command == 'continue':
        encoder.load_state_dict(continue_from_checkpoint['encoder_state_dict'])
        decoder.load_state_dict(continue_from_checkpoint['decoder_state_dict'])
        discriminator.load_state_dict(continue_from_checkpoint['discriminator_state_dict'])
        scheduler.load_state_dict(continue_from_checkpoint['scheduler_state_dict'])

    # Use GPU
    if args.cuda:
        print('---> Loading into GPU memory...')
        encoder.cuda()
        decoder.cuda()
        discriminator.cuda()
        mse_loss.cuda()
        bce_loss.cuda()
        vgg.cuda()

    

    start_epoch = 0
    iteration = 0
    if args.command == 'continue':
        start_epoch = continue_from_checkpoint['epoch'] + 1
        iteration = continue_from_checkpoint['iteration']
    metric_names = ['adv_loss', 'mse_loss', 'vgg_loss', 'decoder_loss', 'loss',
                    'bit_err', 'decode_accuracy','decode_accuracy2', 'psnr', 'ssim']
    metrics = {m: 0 for m in metric_names}

    tic = time.time()
    for e in range(start_epoch, args.epochs):
        print('---> Epoch %d starts training...' % e)
        epoch_start_time = time.time()
        # ------ train ------
        encoder.train()
        decoder.train()
        discriminator.train()
        i = 0  # batch idx
        train_iter = iter(train_loader)
        while i < len(train_loader):
            # ---------------- Train the discriminator -----------------------------
            if use_discriminator:
                Diters = 5
                j = 0
                while j < Diters and i < len(train_loader):
                    for p in discriminator.parameters():  # reset requires_grad
                        p.requires_grad = True
                    image_in = next(train_iter)
                    batch_size, _, h, w = image_in.size()
                    message_in = torch.zeros((batch_size, args.data_depth, h, w)).random_(0, 2)
                    if args.cuda:
                        message_in = message_in.cuda()
                        image_in = image_in.cuda()

                    optimizer_discriminator.zero_grad()
                    stego = encoder(image_in, message_in)
                    d_on_cover = discriminator(image_in)
                    d_on_stego = discriminator(stego.detach())
                    d_loss = d_on_cover.mean() - d_on_stego.mean()
                    d_loss.backward()
                    optimizer_discriminator.step()

                    for p in discriminator.parameters():
                        p.data.clamp_(-0.01, 0.01)

                    j += 1
                    i += 1

            if i == len(train_loader):
                break
            # -------------- Train the (encoder-decoder) ---------------------
            for p in discriminator.parameters():
                p.requires_grad = False
            image_in = next(train_iter)
            batch_size, _, h, w = image_in.size()
            message_in = torch.zeros((batch_size, args.data_depth, h, w)).random_(0, 2)
            if args.cuda:
                message_in = message_in.cuda()
                image_in = image_in.cuda()

            optimizer_coders.zero_grad()
            stego = encoder(image_in, message_in)
            extract_message = decoder(stego)
            extract_message2 = decoder(torch.round(stego))
            g_on_stego = discriminator(stego)

            g_adv_loss = g_on_stego.mean()
            g_mse_loss = mse_loss(stego, image_in)
            g_vgg_loss = torch.tensor(0.)
            if args.use_vgg:
                vgg_on_cov = vgg(image_in)
                vgg_on_enc = vgg(stego)
                g_vgg_loss = mse_loss(vgg_on_enc, vgg_on_cov)
            g_decoder_loss = bce_loss(extract_message, message_in)

            g_loss = (lambda_adv * g_adv_loss + lambda_g * g_mse_loss + lambda_m * g_decoder_loss + lambda_p * g_vgg_loss) \
                if use_discriminator else (lambda_g * g_mse_loss + lambda_m * g_decoder_loss + lambda_p * g_vgg_loss)
            g_loss.backward()
            optimizer_coders.step()
            with torch.no_grad():
                decoded_rounded = extract_message.detach().cpu().numpy().round().clip(0, 1)
                decoded_rounded2 = extract_message2.detach().cpu().numpy().round().clip(0, 1)
                decode_accuracy = (decoded_rounded == message_in.detach().cpu().numpy()).sum() / decoded_rounded.size
                decode_accuracy2 = (decoded_rounded2 == message_in.detach().cpu().numpy()).sum() / decoded_rounded.size
                bit_err = 1 - decode_accuracy
                stego_round = torch.round(stego)

                metrics['adv_loss'] += g_adv_loss.item()
                metrics['mse_loss'] += g_mse_loss.item()
                metrics['vgg_loss'] += g_vgg_loss.item()
                metrics['decoder_loss'] += g_decoder_loss.item()
                metrics['loss'] += g_loss.item()
                metrics['decode_accuracy'] += decode_accuracy.item()
                metrics['decode_accuracy2'] += decode_accuracy2.item()
                metrics['bit_err'] += bit_err.item()
                metrics['psnr'] += utils.psnr_between_batches(image_in, stego_round)
                metrics['ssim'] += pytorch_ssim.ssim(image_in / 255.0, stego_round / 255.0).item()

            i += 1
            iteration += 1

            if iteration % 50 == 0:
                for k in metrics.keys():
                    metrics[k] /= 50
                print('\nEpoch: %d, iteration: %d' % (e, iteration))
                for k in metrics.keys():
                    if 'loss' in k:
                        print(k + ': %.6f' % metrics[k], end='\t')
                print('')
                for k in metrics.keys():
                    if 'loss' not in k:
                        print(k + ': %.6f' % metrics[k], end='\t')
                utils.write_losses(train_csv_file, iteration, e, metrics, time.time() - tic)
                for k in metrics.keys():
                    metrics[k] = 0

        # ------ validate ------
        val_metrics = {m: 0 for m in metric_names}
        print('\n---> Epoch %d starts validating...' % e)
        encoder.eval()
        decoder.eval()
        discriminator.eval()
        for batch_id, image_in in enumerate(valid_loader):
            batch_size, _, h, w = image_in.size()
            message_in = torch.zeros((batch_size, args.data_depth, h, w)).random_(0, 2)
            if args.cuda:
                message_in = message_in.cuda()
                image_in = image_in.cuda()

            with torch.no_grad():
                stego = encoder(image_in, message_in)
                extract_message = decoder(stego)
                stego_round = torch.round(stego)
                extract_message2 = decoder(stego_round)

                d_on_cover = discriminator(image_in)
                d_on_stego = discriminator(stego_round)
                d_loss = d_on_cover.mean() - d_on_stego.mean()

                g_adv_loss = d_on_stego.mean()
                g_mse_loss = mse_loss(stego_round, image_in)
                g_vgg_loss = torch.tensor(0.)
                if args.use_vgg:
                    vgg_on_cov = vgg(image_in)
                    vgg_on_enc = vgg(stego_round)
                    g_vgg_loss = mse_loss(vgg_on_enc, vgg_on_cov)

                g_decoder_loss = bce_loss(extract_message, message_in)
                g_loss = (lambda_adv * g_adv_loss + lambda_g * g_mse_loss + lambda_m * g_decoder_loss + lambda_p * g_vgg_loss) \
                    if use_discriminator else (lambda_g * g_mse_loss + lambda_m * g_decoder_loss + lambda_p * g_vgg_loss)

                decoded_rounded = extract_message.detach().cpu().numpy().round().clip(0, 1)
                decoded_rounded2 = extract_message2.detach().cpu().numpy().round().clip(0, 1)
                decode_accuracy = (decoded_rounded == message_in.detach().cpu().numpy()).sum() / decoded_rounded.size
                decode_accuracy2 = (decoded_rounded2 == message_in.detach().cpu().numpy()).sum() / decoded_rounded.size
                bit_err = 1 - decode_accuracy

                val_metrics['adv_loss'] += g_adv_loss.item()
                val_metrics['mse_loss'] += g_mse_loss.item()
                val_metrics['vgg_loss'] += g_vgg_loss.item()
                val_metrics['decoder_loss'] += g_decoder_loss.item()
                val_metrics['loss'] += g_loss.item()
                val_metrics['decode_accuracy'] += decode_accuracy.item()
                val_metrics['decode_accuracy2'] += decode_accuracy2.item()
                val_metrics['bit_err'] += bit_err.item()
                val_metrics['psnr'] += utils.psnr_between_batches(image_in, stego_round)
                val_metrics['ssim'] += pytorch_ssim.ssim(image_in / 255.0, stego_round / 255.0).item()

        for k in val_metrics.keys():
            val_metrics[k] /= len(valid_loader)
        print('Valid epoch: {}'.format(e))
        for k in val_metrics.keys():
            if 'loss' in k:
                print(k + ': %.6f' % val_metrics[k], end='\t')
        print('')
        for k in val_metrics.keys():
            if 'loss' not in k:
                print(k + ': %.6f' % val_metrics[k], end='\t')
        print('time:%.0f' % (time.time() - tic))
        print('Epoch %d finished, taking %0.f seconds\n' % (e, time.time() - epoch_start_time))
        utils.write_losses(valid_csv_file, iteration, e, val_metrics, time.time() - tic)

        scheduler.step()

        # save model
        if (e + 1) % 10 == 0 or e == args.epochs - 1:
            checkpoint = {
                'epoch': e,
                'iteration': iteration,
                'encoder_state_dict': encoder.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_coders_state_dict': optimizer_coders.state_dict(),
                'optimizer_discriminator_state_dict': optimizer_discriminator.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'metrics': val_metrics
            }
            filename = os.path.join(checkpoints_dir, "epoch_" + str(e) + ".pt")
            torch.save(checkpoint, filename)


if __name__ == '__main__':
    main()
