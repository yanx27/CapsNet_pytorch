import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import torchvision
from torch.autograd import Variable
from tqdm import tqdm
from collections import defaultdict
from CapsNet import CapsNet, CapsNetLoss,exponential_decay
from utils import save_checkpoint,show_example,test,plot_loss_curve
import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser('CapsNet')
    parser.add_argument('--input_size', default=(1, 28, 28),
                        help='the input size of image')
    parser.add_argument('--batchsize', default=128,
                        help='batch size in training')
    parser.add_argument('--epoch',
                        help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.001,
                        help='learning rate in training')
    parser.add_argument('--gpu', type=str, default='0',
                        help='specify gpu device')
    parser.add_argument('--train_metric', type=str, default=False,
                        help='whether evaluate on training dataset')
    parser.add_argument('--pretrain', type=str, default=None,
                        help='whether use pretrain model')
    parser.add_argument('--result_dir', type=str, default='./results/',
                        help='dir to save pictures')
    return parser.parse_args()

def main(args):
    '''Load MNIST'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    save_dir = Path(args.result_dir)
    save_dir.mkdir(exist_ok=True)
    INPUT_SIZE = args.input_size
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(INPUT_SIZE[1:], padding=2),
        torchvision.transforms.ToTensor(),
    ])
    trn_dataset = torchvision.datasets.MNIST('.', train=True, download=True, transform=transforms)
    tst_dataset = torchvision.datasets.MNIST('.', train=False, download=True, transform=transforms)
    print('Images for training: %d' % len(trn_dataset))
    print('Images for testing: %d' % len(tst_dataset))

    BATCH_SIZE = args.batchsize  # Batch size not specified in the paper
    trn_loader = torch.utils.data.DataLoader(trn_dataset, BATCH_SIZE, shuffle=True)
    tst_loader = torch.utils.data.DataLoader(tst_dataset, BATCH_SIZE, shuffle=False)

    model = CapsNet(INPUT_SIZE).cuda()
    if args.pretrain is not None:
        print('Use pretrain model')
        checkpoint = torch.load(args.pretrain)
        model.load_state_dict(checkpoint['model_state_dict'])

    print(model)
    print('Number of Parameters: %d' % model.n_parameters())

    criterion = CapsNetLoss()

    LEARNING_RATE = args.learning_rate
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=(0.9, 0.999),
        eps=1e-08
    )

    global_epoch = 0
    global_step = 0
    best_tst_accuracy = 0.0
    history = defaultdict(lambda: list())
    COMPUTE_TRN_METRICS = args.train_metric

    '''Training'''
    for epoch in range(int(args.epoch)):
        print('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))

        for batch_id, (x, y) in tqdm(enumerate(trn_loader), total=len(trn_loader), smoothing=0.9):
            optimizer = exponential_decay(optimizer, LEARNING_RATE, global_epoch, 1,
                                          0.90)  # Configurations not specified in the paper

            x = Variable(x).float().cuda()
            y = Variable(y).cuda()

            y_pred, x_reconstruction = model(x, y)
            loss, margin_loss, reconstruction_loss = criterion(x, y, x_reconstruction, y_pred.cuda())

            history['margin_loss'].append(margin_loss.cpu().data.numpy())
            history['reconstruction_loss'].append(reconstruction_loss.cpu().data.numpy())
            history['loss'].append(loss.cpu().data.numpy())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1

        trn_metrics = test(model, trn_loader) if COMPUTE_TRN_METRICS else None
        tst_metrics = test(model, tst_loader)

        print('Margin Loss: %f' % history['margin_loss'][-1])
        print('Reconstruction Loss: %f' % history['reconstruction_loss'][-1])
        print('Loss: %f' % history['loss'][-1])
        if COMPUTE_TRN_METRICS:
            print('Train Accuracy: %f' % (trn_metrics['accuracy']))
        print('Test Accuracy: %f' % tst_metrics['accuracy'])

        idx = np.random.randint(0, len(x))
        show_example(x[idx], y[idx], x_reconstruction[idx], y_pred[idx], args.result_dir, 'Epoch_{}'.format(epoch))

        if (tst_metrics['accuracy'] >= best_tst_accuracy):
            best_tst_accuracy = tst_metrics['accuracy']
            save_checkpoint(
                global_epoch + 1,
                trn_metrics['accuracy'] if COMPUTE_TRN_METRICS else 0.0,
                tst_metrics['accuracy'],
                model,
                optimizer
            )
        global_epoch += 1

    n_points_avg = 10
    n_points_plot = 1000
    plt.figure(figsize=(20, 10))

    '''Loss Curve'''
    plot_loss_curve(history, n_points_avg, n_points_plot, args.result_dir)


if __name__ == '__main__':
    args = parse_args()
    main(args)
