import functools
import random
import math
from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from datasets import register
from utils import to_pixel_samples


@register('sr-implicit-paired')
class SRImplicitPaired(Dataset):

    def __init__(self, dataset, inp_size=None, augment=False, sample_q=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.augment = augment
        self.sample_q = sample_q

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_lr, img_hr = self.dataset[idx]

        s = img_hr.shape[-2] // img_lr.shape[-2] # assume int scale
        if self.inp_size is None:
            h_lr, w_lr = img_lr.shape[-2:]
            img_hr = img_hr[:, :h_lr * s, :w_lr * s]
            crop_lr, crop_hr = img_lr, img_hr
        else:
            w_lr = self.inp_size
            x0 = random.randint(0, img_lr.shape[-2] - w_lr)
            y0 = random.randint(0, img_lr.shape[-1] - w_lr)
            crop_lr = img_lr[:, x0: x0 + w_lr, y0: y0 + w_lr]
            w_hr = w_lr * s
            x1 = x0 * s
            y1 = y0 * s
            crop_hr = img_hr[:, x1: x1 + w_hr, y1: y1 + w_hr]

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)

        hr_coord, hr_rgb = to_pixel_samples(crop_hr.contiguous())

        if self.sample_q is not None:
            sample_lst = np.random.choice(
                len(hr_coord), self.sample_q, replace=False)
            hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]

        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / crop_hr.shape[-2]
        cell[:, 1] *= 2 / crop_hr.shape[-1]

        return {
            'inp': crop_lr,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb
        }


# def resize_fn(img, size):
    # return transforms.ToTensor()(
        # transforms.Resize(size, Image.BICUBIC)(
            # transforms.ToPILImage()(img)))InterpolationMode.BICUBIC
def resize_fn(img, size):
    return transforms.ToTensor()(
        transforms.Resize(size, transforms.InterpolationMode.BICUBIC)(
            transforms.ToPILImage()(img)))


@register('sr-implicit-downsampled')
class SRImplicitDownsampled(Dataset):

    def __init__(self, dataset, inp_size=None, scale_min=1, scale_max=None,
                 augment=False, sample_q=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.scale_min = scale_min
        if scale_max is None:
            scale_max = scale_min
        self.scale_max = scale_max
        self.augment = augment
        self.sample_q = sample_q

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.dataset[idx]
        s = random.uniform(self.scale_min, self.scale_max)
        # print('s', s)

        if self.inp_size is None:
            h_lr = math.floor(img.shape[-2] / s + 1e-9)
            w_lr = math.floor(img.shape[-1] / s + 1e-9)
            img = img[:, :round(h_lr * s), :round(w_lr * s)] # assume round int
            img_down = resize_fn(img, (h_lr, w_lr))
            crop_lr, crop_hr = img_down, img
        else:
            w_lr = self.inp_size
            w_hr = round(w_lr * s)
            # print(w_lr, w_hr) # 这里根据s随机选择一个尺度扩大，再round
            x0 = random.randint(0, img.shape[-2] - w_hr)
            y0 = random.randint(0, img.shape[-1] - w_hr)
            # print(x0, y0, img.shape) # 随机裁剪
            # 这里的img是tensor
            crop_hr = img[:, x0: x0 + w_hr, y0: y0 + w_hr]
            crop_lr = resize_fn(crop_hr, w_lr)
            # print(crop_hr.shape, crop_lr.shape) # 利用resize_fn进行下采样得到输入
        
        # 随机增强
        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)
        
        '''关键函数'''
        hr_coord, hr_rgb = to_pixel_samples(crop_hr.contiguous())
        # print(hr_coord.shape, hr_rgb.shape)
        # 维度变换后的变量是之前变量的浅拷贝，指向同一区域，
        # 即 view 操作会连带原来的变量一同变形，这是不合法的，所以也会报错。
        # （这个解释有部分道理，也即 contiguous 返回了 tensor 的深拷贝 contiguous copy 数据）
        # 这里为了保证函数内的操作合法，进行连续化，并没有改变形状

        if self.sample_q is not None:
            # print(len(hr_coord))
            sample_lst = np.random.choice(
                len(hr_coord), self.sample_q, replace=False)
            # 从a(一维的)中随机抽取数字，并组成指定大小(size)的数组
            # If a is an int, the random sample is generated as if a were np.arange(a)
            # replace:True表示可以取相同数字
            # print(sample_lst)
            hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]
        # print(hr_coord.shape, hr_rgb.shape)
        # 将尺寸减小至sample_q
        # 2304=48*48

        cell = torch.ones_like(hr_coord)
        # print(cell[:, 0].shape, crop_hr.shape[-2])
        # cell = cell * 2 / 180(hr size) why?
        cell[:, 0] *= 2 / crop_hr.shape[-2]
        cell[:, 1] *= 2 / crop_hr.shape[-1]
        # print(cell.shape)
        
        return {
            'inp': crop_lr,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb
        }


@register('sr-implicit-uniform-varied')
class SRImplicitUniformVaried(Dataset):

    def __init__(self, dataset, size_min, size_max=None,
                 augment=False, gt_resize=None, sample_q=None):
        self.dataset = dataset
        self.size_min = size_min
        if size_max is None:
            size_max = size_min
        self.size_max = size_max
        self.augment = augment
        self.gt_resize = gt_resize
        self.sample_q = sample_q

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_lr, img_hr = self.dataset[idx]
        p = idx / (len(self.dataset) - 1)
        w_hr = round(self.size_min + (self.size_max - self.size_min) * p)
        img_hr = resize_fn(img_hr, w_hr)

        if self.augment:
            if random.random() < 0.5:
                img_lr = img_lr.flip(-1)
                img_hr = img_hr.flip(-1)

        if self.gt_resize is not None:
            img_hr = resize_fn(img_hr, self.gt_resize)

        hr_coord, hr_rgb = to_pixel_samples(img_hr)

        if self.sample_q is not None:
            sample_lst = np.random.choice(
                len(hr_coord), self.sample_q, replace=False)
            hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]

        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / img_hr.shape[-2]
        cell[:, 1] *= 2 / img_hr.shape[-1]

        return {
            'inp': img_lr,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb
        }




@register('sr-implicit-paired-swin1')
class SRImplicitPaired_swin1(Dataset):

    def __init__(self, dataset, inp_size=None, augment=False, sample_q=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.augment = augment
        self.sample_q = sample_q

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_lr, img_hr = self.dataset[idx]

        s = img_hr.shape[-2] // img_lr.shape[-2] # assume int scale
        if self.inp_size is None:
            # swin start
            inp = img_lr
            window_size = 8
            _, h_old, w_old = inp.size()
            h_pad = (h_old // window_size + 1) * window_size - h_old
            w_pad = (w_old // window_size + 1) * window_size - w_old
            inp = torch.cat([inp, torch.flip(inp, [1])], 1)[:, :h_old + h_pad, :]
            img_lr = torch.cat([inp, torch.flip(inp, [2])], 2)[:, :, :w_old + w_pad]
            # swin end
            
            # swin start
            h_lr, w_lr = img_lr.shape[-2:]
            img_hr = torch.cat([img_hr, torch.flip(img_hr, [1])], 1)
            img_hr = torch.cat([img_hr, torch.flip(img_hr, [2])], 2)
            img_hr = img_hr[:, :h_lr * s, :w_lr * s]
            # swin end
            
            crop_lr, crop_hr = img_lr, img_hr
        else:
            w_lr = self.inp_size
            x0 = random.randint(0, img_lr.shape[-2] - w_lr)
            y0 = random.randint(0, img_lr.shape[-1] - w_lr)
            crop_lr = img_lr[:, x0: x0 + w_lr, y0: y0 + w_lr]
            w_hr = w_lr * s
            x1 = x0 * s
            y1 = y0 * s
            crop_hr = img_hr[:, x1: x1 + w_hr, y1: y1 + w_hr]

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)

        hr_coord, hr_rgb = to_pixel_samples(crop_hr.contiguous())

        if self.sample_q is not None:
            sample_lst = np.random.choice(
                len(hr_coord), self.sample_q, replace=False)
            hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]

        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / crop_hr.shape[-2]
        cell[:, 1] *= 2 / crop_hr.shape[-1]

        return {
            'inp': crop_lr,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb
        }




@register('sr-implicit-paired-swin2')
class SRImplicitPaired_swin2(Dataset):

    def __init__(self, dataset, inp_size=None, augment=False, sample_q=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.augment = augment
        self.sample_q = sample_q

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_lr, img_hr = self.dataset[idx]
        gt = img_hr

        s = img_hr.shape[-2] // img_lr.shape[-2] # assume int scale
        if self.inp_size is None:
            # swin start
            inp = img_lr
            # print('lr', img_lr.shape) # lr torch.Size([3, 256, 256])
            window_size = 8
            _, h_old, w_old = inp.size()
            h_pad = (h_old // window_size + 1) * window_size - h_old
            w_pad = (w_old // window_size + 1) * window_size - w_old
            inp = torch.cat([inp, torch.flip(inp, [1])], 1)[:, :h_old + h_pad, :]
            img_lr = torch.cat([inp, torch.flip(inp, [2])], 2)[:, :, :w_old + w_pad]
            # swin end 
            # swin start
            h_lr, w_lr = img_lr.shape[-2:]
            img_hr = torch.cat([img_hr, torch.flip(img_hr, [1])], 1)
            img_hr = torch.cat([img_hr, torch.flip(img_hr, [2])], 2)
            img_hr = img_hr[:, :h_lr * s, :w_lr * s]
            # print('hr', img_hr.shape) # hr torch.Size([3, 528, 528])
            # print('gt', gt.shape) # gt torch.Size([3, 512, 512])
            # swin end
            
            crop_lr, crop_hr = img_lr, img_hr
        else:
            w_lr = self.inp_size
            x0 = random.randint(0, img_lr.shape[-2] - w_lr)
            y0 = random.randint(0, img_lr.shape[-1] - w_lr)
            crop_lr = img_lr[:, x0: x0 + w_lr, y0: y0 + w_lr]
            w_hr = w_lr * s
            x1 = x0 * s
            y1 = y0 * s
            crop_hr = img_hr[:, x1: x1 + w_hr, y1: y1 + w_hr]

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)

        hr_coord, hr_rgb = to_pixel_samples(crop_hr.contiguous())

        if self.sample_q is not None:
            sample_lst = np.random.choice(
                len(hr_coord), self.sample_q, replace=False)
            hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]

        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / crop_hr.shape[-2]
        cell[:, 1] *= 2 / crop_hr.shape[-1]

        return {
            'inp': crop_lr,
            'coord': hr_coord,
            'cell': cell,
            'gt': gt,
            'hr': img_hr
        }




@register('sr-implicit-downsampled-swin2')
class SRImplicitDownsampled_swin2(Dataset):

    def __init__(self, dataset, inp_size=None, scale_min=1, scale_max=None,
                 augment=False, sample_q=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.scale_min = scale_min
        if scale_max is None:
            scale_max = scale_min
        self.scale_max = scale_max
        self.augment = augment
        self.sample_q = sample_q

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.dataset[idx]
        gt = img
        
        s = random.uniform(self.scale_min, self.scale_max) # 测试时s的值唯一

        if self.inp_size is None:
            # 先确定下采样后的lr满足窗
            h_lr = math.floor(img.shape[-2] / s + 1e-9)
            w_lr = math.floor(img.shape[-1] / s + 1e-9)
            
            window_size = 8
            h_old, w_old = h_lr, w_lr
            h_pad = (h_old // window_size + 1) * window_size - h_old
            w_pad = (w_old // window_size + 1) * window_size - w_old
            
            # 求对应原图大小，然后扩大原图尺寸，再对原图pad
            h_lr = h_old + h_pad
            w_lr = w_old + w_pad
            
            img_hr = torch.cat([img, torch.flip(img, [1])], 1)
            img_hr = torch.cat([img, torch.flip(img, [2])], 2)
            img = img_hr[:, :round(h_lr * s), :round(w_lr * s)] # assume round int
            
            # 获取lr，变得简单
            img_down = resize_fn(img, (h_lr, w_lr))
            crop_lr, crop_hr = img_down, img

            
        else:
            w_lr = self.inp_size
            w_hr = round(w_lr * s)
            x0 = random.randint(0, img.shape[-2] - w_hr)
            y0 = random.randint(0, img.shape[-1] - w_hr)
            crop_hr = img[:, x0: x0 + w_hr, y0: y0 + w_hr]
            crop_lr = resize_fn(crop_hr, w_lr)
        
        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)
        hr_coord, hr_rgb = to_pixel_samples(crop_hr.contiguous())

        if self.sample_q is not None:
            sample_lst = np.random.choice(
                len(hr_coord), self.sample_q, replace=False)
            hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]

        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / crop_hr.shape[-2]
        cell[:, 1] *= 2 / crop_hr.shape[-1]
        
        return {
            'inp': crop_lr,
            'coord': hr_coord,
            'cell': cell,
            'gt': gt,
            'hr': img
        }


if __name__ == '__main__':
    import image_folder
    
    def make_coord(shape, ranges=None, flatten=True):
        """ Make coordinates at grid centers.
        """
        coord_seqs = []
        # (0,h) (1,w) 不复杂
        for i, n in enumerate(shape):
            # print(i, n)
            if ranges is None:
                v0, v1 = -1, 1
            else:
                v0, v1 = ranges[i]
            r = (v1 - v0) / (2 * n)
            seq = v0 + r + (2 * r) * torch.arange(n).float()
            # torch.arange(n) 返回一维张量，0-n
            coord_seqs.append(seq)
            # print(r, seq.shape, len(coord_seqs)) # 不同的尺寸有不同的r
        
        # torch.meshgrid用于生成坐标，函数输入两个数据类型相同的一维张量，一个是行，一个是列
        # 形式：x, y = torch.meshgrid(a, b)
        # 第一个输出张量填充第一个输入张量中的元素，各行元素相同；
        # 第二个输出张量填充第二个输入张量中的元素，各列元素相同。
        ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
        # print(ret.shape)
        if flatten:
            ret = ret.view(-1, ret.shape[-1])
        # print(ret)
        return ret
    
    def to_pixel_samples(img):
        """ Convert the image to coord-RGB pairs.
            img: Tensor, (3, H, W)
        """
        # print(img.shape[-2:]) # 仅仅是一个形状，不带有信息
        coord = make_coord(img.shape[-2:])
        rgb = img.view(3, -1).permute(1, 0) # 将通道交换至最后，即（h*w，c）
        return coord, rgb
    
    '''
    dataset = image_folder.ImageFolder(root_path='F:\DIV2K\DIV2K_train_HR', repeat=20, cache='none')
    # print(len(dataset), dataset[0].shape)
    # 和普通dataset一样
    
    dataset = SRImplicitDownsampled(dataset, inp_size=48, scale_min=1, scale_max=4, augment=True, sample_q=2304)
    print(len(dataset), dataset[0]['inp'].shape)
    # 每一次索引之后得到一个字典
    '''
    dataset = image_folder.PairedImageFolders(root_path_1='F:/benchmark/Set5/LR_bicubic/X2', root_path_2='F:/benchmark/Set5/HR')
    dataset = SRImplicitPaired_swin2(dataset)
    print(len(dataset), dataset[0]['inp'].shape)