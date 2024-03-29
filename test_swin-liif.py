import argparse
import os
import math
from functools import partial

import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import datasets
import models
import utils

import numpy as np


def batched_predict(model, inp, coord, cell, bsize):
    with torch.no_grad():
        model.gen_feat(inp)
        n = coord.shape[1]
        ql = 0
        preds = []
        while ql < n:
            qr = min(ql + bsize, n)
            pred = model.query_rgb(coord[:, ql: qr, :], cell[:, ql: qr, :])
            preds.append(pred)
            ql = qr
        pred = torch.cat(preds, dim=1)
    return pred


def eval_psnr(loader, model, data_norm=None, eval_type=None, eval_bsize=None,
              verbose=False):
    model.eval()

    if data_norm is None:
        data_norm = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }
    t = data_norm['inp']
    inp_sub = torch.FloatTensor(t['sub']).view(1, -1, 1, 1).cuda()
    inp_div = torch.FloatTensor(t['div']).view(1, -1, 1, 1).cuda()
    t = data_norm['gt']
    gt_sub = torch.FloatTensor(t['sub']).view(1, 1, -1).cuda()
    gt_div = torch.FloatTensor(t['div']).view(1, 1, -1).cuda()

    if eval_type is None:
        metric_fn = utils.calc_psnr
        ssim_fn = utils.calculate_ssim2
        
    elif eval_type.startswith('div2k'):
        scale = int(eval_type.split('-')[1])
        metric_fn = partial(utils.calc_psnr, dataset='div2k', scale=scale)
        ssim_fn = utils.calculate_ssim2
        
    elif eval_type.startswith('benchmark'):
        scale = int(eval_type.split('-')[1])
        metric_fn = partial(utils.calc_psnr, dataset='benchmark', scale=scale)
        ssim_fn = utils.calculate_ssim2
        
    else:
        raise NotImplementedError

    val_res = utils.Averager()
    val_ssim = utils.Averager()

    pbar = tqdm(loader, leave=False, desc='val')
    for batch in pbar:
        for k, v in batch.items():
            batch[k] = v.cuda()

        inp = (batch['inp'] - inp_sub) / inp_div
        if eval_bsize is None:
            with torch.no_grad():
                # print('inp',inp.shape)
                pred = model(inp, batch['coord'], batch['cell'])
                # print('pred',pred.shape) # pred torch.Size([1, 278784, 3]) 278784=512^2
                # print('gt', batch['gt'].shape)
        else:
            pred = batched_predict(model, inp,
                batch['coord'], batch['cell'], eval_bsize)
        pred = pred * gt_div + gt_sub
        pred.clamp_(0, 1)
        
        # 获得将pred还原的尺寸
        h, w = batch['hr'].shape[-2:]
        shape = [batch['inp'].shape[0], h, w, 3]
        pred = pred.view(*shape).permute(0, 3, 1, 2).contiguous()
        
        # 对pred进行裁剪为gt的尺寸
        # print('gt shape', batch['gt'].shape)
        pred = pred[:, :, :batch['gt'].shape[2], :batch['gt'].shape[3]]
        # print('pred shape', pred.shape)
        
        res = metric_fn(pred, batch['gt'])
        
        pred = (pred[0]).cpu().numpy()
        pred = np.transpose(pred[[2, 1, 0], :, :], (1, 2, 0))
        pred = (pred * 255.0).round().astype(np.uint8)
        
        gt = (batch['gt'][0]).cpu().numpy()
        gt = np.transpose(gt[[2, 1, 0], :, :], (1, 2, 0))
        gt = (gt * 255.0).round().astype(np.uint8)
        # print(pred.shape, gt.dtype)
        
        ssim = ssim_fn(pred, gt, crop_border=scale, input_order='HWC')
        
        val_res.add(res.item(), inp.shape[0])
        val_ssim.add(ssim.item(), inp.shape[0])

        if verbose:
            pbar.set_description('psnr val {:.4f}'.format(val_res.item()))
            pbar.set_description('ssim val {:.4f}'.format(val_ssim.item()))

    return val_res.item(), val_ssim.item()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/test-swin/test-set5-2.yaml')
    parser.add_argument('--model', default='H:/spyder/RDST_model/_train_swin-dense-liif-small-df2k/epoch-last.pth')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    spec = config['test_dataset']
    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    loader = DataLoader(dataset, batch_size=spec['batch_size'],
        num_workers=0, pin_memory=True)

    model_spec = torch.load(args.model)['model']
    # for swin
    # model_spec['args']['encoder_spec']
    model = models.make(model_spec, load_sd=True).cuda()

    res, ssim = eval_psnr(loader, model,
        data_norm=config.get('data_norm'),
        eval_type=config.get('eval_type'),
        eval_bsize=config.get('eval_bsize'),
        verbose=True)
    # print('result psnr: {:.4f}'.format(res))
    print('result ssim: {:.4f}'.format(ssim))
    