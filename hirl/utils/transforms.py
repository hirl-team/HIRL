import random
import numpy as np
from PIL import ImageFilter, ImageOps, Image, ImageDraw
from torchvision import transforms


class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )

class Solarization(object):
    """
    Apply Solarization to the PIL image.
    """
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


class PermutePatch(object):
    """
    Apply Patch permutation to the PIL image.
    """
    def __init__(self, psz):
        self.psz = psz

    def __call__(self, img):
        imgs = []
        imgwidth, imgheight = img.size
        for i in range(0, imgheight, self.psz):
            for j in range(0, imgwidth, self.psz):
                box = (j, i, j+self.psz, i+self.psz)
                imgs.append(img.crop(box))
        random.shuffle(imgs)
        new_img = Image.new('RGB', (imgwidth, imgheight))
        k = 0
        for i in range(0, imgheight, self.psz):
            for j in range(0, imgwidth, self.psz):
                new_img.paste(imgs[k], (j, i))
                k += 1
        return new_img

class HideAndSeek(object):
    """
    Apply Patch permutation to the PIL image.
    """
    def __init__(self, ratio, psz):
        self.ratio = ratio
        self.psz = psz

    def __call__(self, img):
        imgwidth, imgheight = img.size 
        numw, numh = imgwidth // self.psz, imgheight // self.psz
        mask_num = int(numw * numh * self.ratio)
        mask_patch = np.random.choice(np.arange(numw * numh), mask_num, replace=False)
        mask_w, mask_h = mask_patch % numh, mask_patch // numh
        draw = ImageDraw.Draw(img)
        for mw, mh in zip(mask_w, mask_h):
            draw.rectangle((mw * self.psz, 
                            mh * self.psz,
                            (mw + 1) * self.psz,
                            (mh + 1) * self.psz), fill="black")
        return img

class DataAugmentationDINO(object):
    def __init__(self, global_crops_scale, local_crops_scale, global_crops_number, local_crops_number):
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        self.global_crops_number = global_crops_number
        # transformation for the first global crop
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(1.0),
            normalize,
        ])
        ## Note: an extra solarization is applied on other global crops | why doing so?
        # transformation for the rest of global crops
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(0.1),
            Solarization(0.2),
            normalize,
        ])
        # transformation for the local crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(96, scale=local_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(p=0.5),
            normalize,
        ])

    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        for _ in range(self.global_crops_number - 1):
            crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops


class DataAugmentationiBOT(DataAugmentationDINO):
    """
    This data augmentation is the same as DINO except for 
    allowing more than 2 global crops. So we just move this 
    feature into the parent class DataAugmentationDINO.
    See https://github.com/facebookresearch/dino/blob/main/main_dino.py for 
    more details.
    """
    pass


class DataAugmentationMoCoV3(object):
    def __init__(self, global_crops_scale, local_crops_scale, global_crops_number, local_crops_number):
        color_transform = transforms.Compose([
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        # transformation for global crops
        self.global_crops_number = global_crops_number
        self.global_transform1 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale),
            color_transform,
            GaussianBlur(1.0),
            transforms.RandomHorizontalFlip(),
            normalize
        ])
        self.global_transform2 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale),
            color_transform,
            GaussianBlur(0.1),
            Solarization(0.2),
            transforms.RandomHorizontalFlip(),
            normalize
        ])

        # transformation for the local crops
        self.local_crops_number = local_crops_number
        self.local_transform = transforms.Compose([
            transforms.RandomResizedCrop(96, scale=local_crops_scale),
            color_transform,
            GaussianBlur(p=0.5),
            transforms.RandomHorizontalFlip(),
            normalize
        ])

    def __call__(self, image):
        crops = []
        crops.append(self.global_transform1(image))
        for _ in range(self.global_crops_number - 1):
            crops.append(self.global_transform2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transform(image))
        return crops


class SingleCropTransform(object):
    """
    MoCo v2's augmentation: similar to SimCLR https://arxiv.org/abs/2002.05709
    """
    def __init__(self, size_crop, min_scale_crop, max_scale_crop, color_scale=0.5, color_first=True):
        
        color_transform = [transforms.RandomApply([transforms.ColorJitter(0.8 * color_scale, 0.8 * color_scale, \
                0.8 * color_scale, 0.2 * color_scale)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=0.5, radius_min=0.1, radius_max=2.0),]
        
        base_augmentations = []
        if color_first:
            base_augmentations.extend(color_transform)
            base_augmentations.extend([transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                                       transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        else:
            base_augmentations.append(transforms.RandomHorizontalFlip())
            base_augmentations.extend(color_transform)
            base_augmentations.extend([transforms.ToTensor(),
                                       transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        
        
        self.trans = transforms.Compose([
            transforms.RandomResizedCrop(size_crop, scale=(min_scale_crop, max_scale_crop)),] + base_augmentations)

    def __call__(self, x):
        q = self.trans(x)
        k = self.trans(x)
        return [q, k]

class MultiCropsTransform(object):
    """
    The code is modified from 
    https://github.com/maple-research-lab/AdCo/blob/b8f749db3e8e075f77ec17f859e1b2793844f5d3/data_processing/MultiCrop_Transform.py
    
    Args:
        size_crops: (list of int) the crop size for each group of transform
        nmb_crops: (list of int) the number of crops for each group of transform
        min_scale_crops: (list of int) min random resized crop scale for each group of transform
        max_scale_crops: (list of int) max random resized crop scale for each group of transform
        color_scale: (float) scale for color jitter
        color_first: (bool) if set true, apply color transform before RandomHorizontalFlip
    """
    def __init__(self, size_crops, nmb_crops, min_scale_crops, max_scale_crops, color_scale=0.5, color_first=True):
        assert len(size_crops) == len(nmb_crops)
        assert len(min_scale_crops) == len(nmb_crops)
        assert len(max_scale_crops) == len(nmb_crops)
        # transformation for query
        color_transform = [transforms.RandomApply([transforms.ColorJitter(0.8 * color_scale, 0.8 * color_scale, \
                0.8 * color_scale, 0.2 * color_scale)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=0.5, radius_min=0.1, radius_max=2.0),]
        
        base_augmentations = []
        if color_first:
            base_augmentations.extend(color_transform)
            base_augmentations.extend([transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                                       transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        else:
            base_augmentations.append(transforms.RandomHorizontalFlip())
            base_augmentations.extend(color_transform)
            base_augmentations.extend([transforms.ToTensor(),
                                       transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


        trans_q = transforms.Compose([transforms.RandomResizedCrop(size_crops[0], 
            scale=(min_scale_crops[0], max_scale_crops[0]))] + base_augmentations)
        all_trans = [trans_q]

        # transformations for keys
        nmb_crops[0] -= 1  # remove the query from the first crop count
        for i in range(len(size_crops)):
            crop = transforms.RandomResizedCrop(size_crops[i], scale=(min_scale_crops[i], max_scale_crops[i]))
            trans_k = transforms.Compose([crop] + base_augmentations)
            all_trans.extend([trans_k] * nmb_crops[i])

        self.all_trans = all_trans
        print("In total, we have %d transformations." % (len(self.all_trans)))

    def __call__(self, x):
        multi_crops = list(map(lambda trans: trans(x), self.all_trans))
        return multi_crops


class MultiAugmentationWrapper(object):
    def __init__(self, trans, num_views=2) -> None:
        self.trans = trans
        self.num_views = num_views
    
    def __call__(self, x):
        return [self.trans(x) for _ in range(self.num_views)]