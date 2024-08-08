import argparse
import os
import math
from functools import partial
import matplotlib.pyplot as plt
import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import datasets
import models
import utils


def batched_predict(model, inp, coord, cell, wave_inp, wave_inp_l, wave_cell, bsize):
    with torch.no_grad():
        model.gen_feat(inp, wave_inp, wave_inp_l)

        #coord = coord.reshape(bs, -1, 2)
        n = coord.shape[1]
        ql = 0
        preds = []
        while ql < n:
            qr = min(ql + bsize, n)
            #pred = model.query_rgb(coord[:, ql: qr, :], wave_coord, cell[:, ql: qr, :], wave_cell[:, ql: qr, :])
            pred = model.query_rgb(coord[:, ql: qr, :], cell, wave_cell)
            preds.append(pred)
            ql = qr
        pred = torch.cat(preds, dim=1)
    return pred


def eval_psnr(loader, model, data_norm=None, eval_type=None, eval_bsize=None, window_size=0, scale_max=4, fast=False,
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
    elif eval_type.startswith('div2k'):
        scale = int(eval_type.split('-')[1])
        metric_fn = partial(utils.calc_psnr, dataset='div2k', scale=scale)
    elif eval_type.startswith('benchmark'):
        scale = int(eval_type.split('-')[1])
        metric_fn = partial(utils.calc_psnr, dataset='benchmark', scale=scale)
    else:
        raise NotImplementedError

    val_res = utils.Averager()

    pbar = tqdm(loader, leave=False, desc='val')
    for batch in pbar:
        for k, v in batch.items():
            if v is not None:
               batch[k] = v.cuda()
        bs = batch['inp'].shape[0]
        inp = (batch['inp'] - inp_sub) / inp_div
        #wave_inp = (batch['wave_lr'] - inp_sub) / inp_div
        #wave_inp_l = (batch['wave_lr_l'] - inp_sub) / inp_div
        # SwinIR Evaluation - reflection padding
        if window_size != 0:
            _, _, h_old, w_old = inp.size()
            h_pad = (h_old // window_size + 1) * window_size - h_old
            w_pad = (w_old // window_size + 1) * window_size - w_old
            inp = torch.cat([inp, torch.flip(inp, [2])], 2)[:, :, :h_old + h_pad, :]
            inp = torch.cat([inp, torch.flip(inp, [3])], 3)[:, :, :, :w_old + w_pad]
            
            coord = utils.make_coord((scale*(h_old+h_pad), scale*(w_old+w_pad))).unsqueeze(0).cuda()
            cell = torch.ones_like(coord)
            cell[:, :, 0] *= 2 / inp.shape[-2] / scale
            cell[:, :, 1] *= 2 / inp.shape[-1] / scale
        else:
            h_pad = 0
            w_pad = 0
            
            coord = batch['sample_coord']
            cell = batch['cell']
            
        if eval_bsize is None:
            with torch.no_grad():
                pred = model(inp, coord, cell)
        else:
            if fast:
                pred = model(inp, coord, cell*max(scale/scale_max, 1))
            else:
                coords = batch['coords']
                bs = coords.shape[0]
                coords = coords.reshape(bs, -1, 2)
                with torch.no_grad():
                   pred = batched_predict(model, inp, coords, cell*max(scale/scale_max, 1), eval_bsize) # cell clip for extrapolation
            
        pred = pred * gt_div + gt_sub
        pred.clamp_(0, 1)
        
        if eval_type is not None and fast == False: # reshape for shaving-eval
            # gt reshape
            ih, iw = batch['inp'].shape[-2:]
            s = math.sqrt(coords.shape[1] / (ih * iw))
            shape = [batch['inp'].shape[0], round(ih * s), round(iw * s), 3]
            #batch['gt'] = batch['gt'].view(*shape) \
            #    .permute(0, 3, 1, 2).contiguous()
            
            # prediction reshape
            ih += h_pad
            iw += w_pad
            s = math.sqrt(coords.shape[1] / (ih * iw))
            shape = [batch['inp'].shape[0], round(ih * s), round(iw * s), 3]
            pred = pred.view(*shape) \
                .permute(0, 3, 1, 2).contiguous()
            pred = pred[..., :batch['gt'].shape[-2], :batch['gt'].shape[-1]]
            #plt.imshow(pred[0].cpu().permute(1,2,0),cmap="gray")
            #plt.show()
        
        sample_coord = batch['sample_coord']
        sample_coord = sample_coord.unsqueeze(2)

        gt = F.grid_sample(batch['gt'], sample_coord.flip(-1), \
                mode='nearest', align_corners=False).permute(0, 2, 3, 1)

        gt = gt.reshape(bs, -1, 3)      
        
        #pred = pred.reshape(bs, -1, 3)
        #gt = batch['gt'].reshape(bs, -1, 3)
        res = metric_fn(pred, gt)
        val_res.add(res.item(), inp.shape[0])
        '''
        if eval_type is not None and fast == False: # reshape for shaving-eval
            # gt reshape
            ih, iw = batch['wave_lr'].shape[-2:]
            s = math.sqrt(batch['wave_coord'].shape[1] / (ih * iw))
            shape = [batch['wave_lr'].shape[0], round(ih * s), round(iw * s), 3]
            batch['wave_gt'] = batch['wave_gt'].view(*shape) \
                .permute(0, 3, 1, 2).contiguous()
            
            # prediction reshape
            ih += h_pad
            iw += w_pad
            s = math.sqrt(coord.shape[1] / (ih * iw))
            shape = [batch['wave_lr'].shape[0], round(ih * s), round(iw * s), 3]
            pred = pred.view(*shape) \
                .permute(0, 3, 1, 2).contiguous()
            pred = pred[..., :batch['wave_gt'].shape[-2], :batch['wave_gt'].shape[-1]]
            
        res = metric_fn(pred, batch['wave_gt'])
        val_res.add(res.item(), wave_inp.shape[0])        
        '''
        if verbose:
            pbar.set_description('val {:.4f}'.format(val_res.item()))
            
    return val_res.item()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='/ssd/data/miccai01/xiaoduan/lter10/configs/test/test-div2k-2.yaml')
    parser.add_argument('--model', default='/ssd/data/miccai01/xiaoduan/lter10/save/train_edsr_baseline/epoch-last.pth')
    parser.add_argument('--window', default='0')
    parser.add_argument('--scale_max', default='4')
    parser.add_argument('--fast', default=False)
    parser.add_argument('--gpu', default='3')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    spec = config['test_dataset']
    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    loader = DataLoader(dataset, batch_size=spec['batch_size'],
        num_workers=8, pin_memory=True, collate_fn=dataset.collate_fn)

    model_spec = torch.load(args.model)['model']
    model = models.make(model_spec, load_sd=True).cuda()

    res = eval_psnr(loader, model,
        data_norm=config.get('data_norm'),
        eval_type=config.get('eval_type'),
        eval_bsize=config.get('eval_bsize'),
        window_size=int(args.window),
        scale_max = int(args.scale_max),
        fast = args.fast,
        verbose=True)
    print('result: {:.4f}'.format(res))