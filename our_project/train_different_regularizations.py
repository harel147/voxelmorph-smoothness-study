#!/usr/bin/env python
import os
import argparse
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd


# import voxelmorph with pytorch backend
os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph as vxm  # nopep8

# parse the commandline
parser = argparse.ArgumentParser()

# data organization parameters
parser.add_argument('--model-dir', default='models',
                    help='model output directory (default: models)')

# training parameters
parser.add_argument('--gpu', default='0', help='GPU ID number(s), comma-separated (default: 0)')
parser.add_argument('--batch-size', type=int, default=1, help='batch size (default: 1)')
parser.add_argument('--epochs', type=int, default=100,
                    help='number of training epochs (default: 1500)')
parser.add_argument('--steps-per-epoch', type=int, default=100,
                    help='frequency of model saves (default: 100)')
parser.add_argument('--load-model', help='optional model file to initialize with')
parser.add_argument('--initial-epoch', type=int, default=0,
                    help='initial epoch number (default: 0)')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')
parser.add_argument('--cudnn-nondet', action='store_true',
                    help='disable cudnn determinism - might slow down training')

# network architecture parameters
parser.add_argument('--enc', type=int, nargs='+',
                    help='list of unet encoder filters (default: 16 32 32 32)')
parser.add_argument('--dec', type=int, nargs='+',
                    help='list of unet decorder filters (default: 32 32 32 32 32 16 16)')
parser.add_argument('--int-steps', type=int, default=7,
                    help='number of integration steps (default: 7)')
parser.add_argument('--int-downsize', type=int, default=2,
                    help='flow downsample factor for integration (default: 2)')

# loss hyperparameters
parser.add_argument('--image-loss', default='mse',
                    help='image reconstruction loss - can be mse or ncc (default: mse)')

# our project hyperparameters
parser.add_argument('--reg-loss', default='l2', help='regularization loss type')
args = parser.parse_args()

lamb = 0.01
regularizations = ['l2', 'l1', 'bend']

results = []
for regu in regularizations:
    print(f"start regularization {regu}")
    model_dir_regu = os.path.join(args.model_dir, "regularizations", f"regu_{regu}")
    os.makedirs(model_dir_regu, exist_ok=True)

    # load and prepare data
    train_img_list = os.path.join(os.path.dirname(os.path.abspath(__file__)).replace("our_project", ""), 'OASIS_DATASET', 'train_list.txt')
    val_img_list = os.path.join(os.path.dirname(os.path.abspath(__file__)).replace("our_project", ""), 'OASIS_DATASET', 'val_list.txt')

    train_files = vxm.py.utils.read_file_list(train_img_list)
    assert len(train_files) > 0, 'Could not find any training data.'
    val_files = vxm.py.utils.read_file_list(val_img_list)
    assert len(val_files) > 0, 'Could not find any val data.'

    # no need to append an extra feature axis if data is multichannel
    add_feat_axis = True

    # scan-to-scan generators
    generator = vxm.generators.scan_to_scan(
        train_files, batch_size=args.batch_size, add_feat_axis=add_feat_axis)
    val_generator = vxm.generators.scan_to_scan(
            val_files, batch_size=args.batch_size, add_feat_axis=add_feat_axis)

    # extract shape from sampled input
    inshape = next(generator)[0][0].shape[1:-1]

    # device handling
    gpus = args.gpu.split(',')
    nb_gpus = len(gpus)
    device = 'cuda'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # enabling cudnn determinism appears to speed up training by a lot
    torch.backends.cudnn.deterministic = not args.cudnn_nondet

    # unet architecture
    enc_nf = args.enc if args.enc else [16, 32, 32, 32]
    dec_nf = args.dec if args.dec else [32, 32, 32, 32, 32, 16, 16]

    if args.load_model:
        # load initial model (if specified)
        model = vxm.networks.VxmDense.load(args.load_model, device)
    else:
        # otherwise configure new model
        model = vxm.networks.VxmDense(
            inshape=inshape,
            nb_unet_features=[enc_nf, dec_nf],
            int_steps=args.int_steps,
            int_downsize=args.int_downsize
        )

    if nb_gpus > 1:
        # use multiple GPUs via DataParallel
        model = torch.nn.DataParallel(model)
        model.save = model.module.save

    # prepare the model for training and send to device
    model.to(device)
    model.train()

    # set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # prepare image loss
    if args.image_loss == 'ncc':
        image_loss_func = vxm.losses.NCC().loss
    elif args.image_loss == 'mse':
        image_loss_func = vxm.losses.MSE().loss
    else:
        raise ValueError('Image loss should be "mse" or "ncc", but found "%s"' % args.image_loss)

    losses = [image_loss_func]
    weights = [1]

    # prepare deformation loss
    losses += [vxm.losses.Grad(regu, loss_mult=args.int_downsize).loss]
    weights += [lamb]

    best_val_loss = float('inf')
    train_loss_of_best_val_loss = float('inf')
    best_model_path = os.path.join(model_dir_regu, 'best_model.pt')

    # training loops
    epoch_total_loss_per_epoch = []
    val_loss_per_epoch = []
    for epoch in range(args.initial_epoch, args.epochs):

        epoch_loss = []
        epoch_total_loss = []
        epoch_step_time = []

        for step in range(args.steps_per_epoch):

            step_start_time = time.time()

            # generate inputs (and true outputs) and convert them to tensors
            inputs, y_true = next(generator)
            # inputs = [torch.from_numpy(d).to(device).float().permute(0, 4, 1, 2, 3) for d in inputs]
            # y_true = [torch.from_numpy(d).to(device).float().permute(0, 4, 1, 2, 3) for d in y_true]
            # for 2D images
            inputs = [torch.from_numpy(d).to(device).float().permute(0, 3, 1, 2) for d in inputs]
            y_true = [torch.from_numpy(d).to(device).float().permute(0, 3, 1, 2) for d in y_true]

            # run inputs through the model to produce a warped image and flow field
            y_pred = model(*inputs)

            # calculate total loss
            loss = 0
            loss_list = []
            for n, loss_function in enumerate(losses):
                curr_loss = loss_function(y_true[n], y_pred[n]) * weights[n]
                loss_list.append(curr_loss.item())
                loss += curr_loss

            epoch_loss.append(loss_list)
            epoch_total_loss.append(loss.item())

            # backpropagate and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # get compute time
            epoch_step_time.append(time.time() - step_start_time)

        epoch_total_loss_per_epoch.append(epoch_total_loss)

        # Validation loss calculation
        model.eval()
        with torch.no_grad():
            val_losses = []
            for _ in range(len(val_files)):
                val_inputs, val_y_true = next(val_generator)
                val_inputs = [torch.from_numpy(d).to(device).float().permute(0, 3, 1, 2) for d in val_inputs]
                val_y_true = [torch.from_numpy(d).to(device).float().permute(0, 3, 1, 2) for d in val_y_true]
                val_y_pred = model(*val_inputs)

                val_loss = sum(
                    loss_func(val_y_true[n], val_y_pred[n]) * weights[n]
                    for n, loss_func in enumerate(losses)
                )
                val_losses.append(val_loss.item())

            mean_val_loss = np.mean(val_losses)
        model.train()

        val_loss_per_epoch.append(mean_val_loss)

        # print epoch info
        epoch_info = 'Epoch %d/%d' % (epoch + 1, args.epochs)
        time_info = '%.4f sec/step' % np.mean(epoch_step_time)
        losses_info = ', '.join(['%.4e' % f for f in np.mean(epoch_loss, axis=0)])
        loss_info = 'loss: %.4e  (%s)' % (np.mean(epoch_total_loss), losses_info)
        val_info = 'val_loss: %.4e' % mean_val_loss
        print(' - '.join((epoch_info, time_info, loss_info, val_info)), flush=True)

        # save model if val loss is lower
        if mean_val_loss < best_val_loss and epoch % 20 == 0:
            best_val_loss = mean_val_loss
            train_loss_of_best_val_loss = np.mean(epoch_total_loss)
            model.save(best_model_path)
            print(f'Saving best model so far - epoch {epoch + 1}/{args.epochs}, val_loss: {mean_val_loss}')

    if mean_val_loss < best_val_loss:
        best_val_loss = mean_val_loss
        train_loss_of_best_val_loss = np.mean(epoch_total_loss)
        model.save(best_model_path)
        print(f'Saving best model so far - epoch {epoch + 1}/{args.epochs}, val_loss: {mean_val_loss}')

    # compute percentage of non-positive Jacobian determinants
    model.eval()
    jacobians = []
    for i in range(len(val_files)):
        val_inputs, val_y_true = next(val_generator)
        val_inputs = [torch.from_numpy(d).to(device).float().permute(0, 3, 1, 2) for d in val_inputs]
        val_y_true = [torch.from_numpy(d).to(device).float().permute(0, 3, 1, 2) for d in val_y_true]
        val_y_pred = model(*val_inputs)

        # Compute Jacobian determinant
        flow = val_y_pred[1].detach().cpu().numpy()[0]  # displacement field
        flow = np.moveaxis(flow, 0, -1)  # [H,W,C]
        jac_det = vxm.py.utils.jacobian_determinant(flow)
        jacobians.append(np.mean(jac_det <= 0))  # % non-positive determinant

    results.append({
        'regularization': regu,
        'val_loss': best_val_loss,
        'train_loss': train_loss_of_best_val_loss,
        'nonpositive_jacobian_mean': np.mean(jacobians)
    })

    # visualize 5 val deformation fields
    val_pair_indices = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
    val_fixed_generator = vxm.generators.fixed_scan_pairs(val_files, val_pair_indices, add_feat_axis=True)
    for i in range(5):
        val_inputs, val_y_true = next(val_fixed_generator)
        val_inputs = [torch.from_numpy(d).to(device).float().permute(0, 3, 1, 2) for d in val_inputs]
        val_y_true = [torch.from_numpy(d).to(device).float().permute(0, 3, 1, 2) for d in val_y_true]
        val_y_pred = model(*val_inputs)

        flow = val_y_pred[1].detach().cpu().numpy()[0]  # displacement field
        flow = np.moveaxis(flow, 0, -1)  # [H,W,C]

        plt.imshow(flow[..., 0], cmap='coolwarm')  # show x-displacement
        plt.colorbar()
        plt.title(f'Flow (x-dir) for regularization={regu}, sample {i}')
        plt.savefig(os.path.join(model_dir_regu, f'flow_sample_{i}.png'))
        plt.close()

    # Save loss curves
    epochs_range = range(args.initial_epoch, args.epochs)
    train_losses = [np.mean(l) for l in epoch_total_loss_per_epoch]
    val_losses = val_loss_per_epoch  # already a list

    plt.figure()
    plt.plot(epochs_range, train_losses, label='Train Loss')
    plt.plot(epochs_range, val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Loss Curves - Regularization: {regu}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir_regu, 'loss_curve.png'))
    plt.close()

results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(args.model_dir, "regularizations", "summary.csv"), index=False)