import argparse
import os
from PIL import Image

import torch
from torchvision import transforms

import models
from utils import make_coord
from test_liif import batched_predict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='F:/benchmark/Urban100/HR/img078.png')
    parser.add_argument('--model', default='H:/spyder/RDST_model/_train_swin-dense-liif-small-df2k/epoch-last.pth')
    parser.add_argument('--output', default='Urban100_img078.png')
    parser.add_argument('--scale', default=2)
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    img = Image.open(args.input).convert('RGB')
    
    # crop
    patch = 48
    hcrop = 100
    wcrop = 320
    box = (wcrop, hcrop, wcrop+patch, hcrop+patch)
    img = img.crop(box)
    img.save('crop.png')
    
    img = transforms.ToTensor()(img)
    model = models.make(torch.load(args.model)['model'], load_sd=True).cuda()
    
    h = w = patch
    scale = args.scale
    h *= scale
    w *= scale
    coord = make_coord((h, w)).cuda()
    cell = torch.ones_like(coord)
    cell[:, 0] *= 2 / h
    cell[:, 1] *= 2 / w
    
    # inference
    pred = batched_predict(model, ((img - 0.5) / 0.5).cuda().unsqueeze(0), coord.unsqueeze(0), cell.unsqueeze(0), bsize=30000)[0]
    pred = (pred * 0.5 + 0.5).clamp(0, 1).view(h, w, 3).permute(2, 0, 1).cpu()
    # bicubic
    # pred = transforms.ToTensor()(transforms.Resize((h, w), transforms.InterpolationMode.BICUBIC)(transforms.ToPILImage()(img)))
    
    transforms.ToPILImage()(pred).save(args.output)