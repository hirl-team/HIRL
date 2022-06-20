import math
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


class RandomResizedCropAndInterpolationWithTwoPic(object):
    """
    Crop the given PIL Image to random size and aspect ratio with random interpolation,
        producing two crops if `second_size` is specified.
    Adapted from: https://github.com/microsoft/unilm/blob/master/beit/transforms.py
    """

    interpolations = {"random": (Image.BILINEAR, Image.BICUBIC), "bicubic": Image.BICUBIC, "lanczos": Image.LANCZOS,
                      "hamming": Image.HAMMING, "bilinear": Image.BILINEAR}

    def __init__(self, size, second_size=None, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.),
                 interpolation='bilinear', second_interpolation='lanczos'):
        if isinstance(size, tuple):
            self.size = size
        else:
            self.size = (size, size)
        if second_size is not None:
            if isinstance(second_size, tuple):
                self.second_size = second_size
            else:
                self.second_size = (second_size, second_size)
        else:
            self.second_size = None
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("range should be of kind (min, max)")

        self.interpolation = self.interpolations[interpolation]
        self.second_interpolation = self.interpolations[second_interpolation]
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img, scale, ratio):
        """
        Get parameters for a random sized crop.
        """
        area = img.size[0] * img.size[1]

        for attempt in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if w <= img.size[0] and h <= img.size[1]:
                i = random.randint(0, img.size[1] - h)
                j = random.randint(0, img.size[0] - w)
                return i, j, h, w

        # Fallback to central crop
        in_ratio = img.size[0] / img.size[1]
        if in_ratio < min(ratio):
            w = img.size[0]
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = img.size[1]
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = img.size[0]
            h = img.size[1]
        i = (img.size[1] - h) // 2
        j = (img.size[0] - w) // 2
        return i, j, h, w

    def __call__(self, img):
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        if isinstance(self.interpolation, (tuple, list)):
            interpolation = random.choice(self.interpolation)
        else:
            interpolation = self.interpolation
        if self.second_size is None:
            return F.resized_crop(img, i, j, h, w, self.size, interpolation)
        else:
            return F.resized_crop(img, i, j, h, w, self.size, interpolation), \
                   F.resized_crop(img, i, j, h, w, self.second_size, self.second_interpolation)


class BlockwiseMasking(object):
    """
    Block-wise masking adapted from BEiT: https://github.com/microsoft/unilm/blob/master/beit/masking_generator.py
    """
    def __init__(self, input_size, num_masking_patches, min_num_patches=4, max_num_patches=None,
                 min_aspect=0.3, max_aspect=None):
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2
        self.height, self.width = input_size

        self.num_patches = self.height * self.width
        self.num_masking_patches = num_masking_patches
        self.min_num_patches = min_num_patches
        self.max_num_patches = num_masking_patches if max_num_patches is None else max_num_patches
        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))

    def get_shape(self):
        return self.height, self.width

    def _mask(self, mask, max_mask_patches):
        delta = 0
        for attempt in range(10):
            target_area = random.uniform(self.min_num_patches, max_mask_patches)
            aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < self.width and h < self.height:
                top = random.randint(0, self.height - h)
                left = random.randint(0, self.width - w)
                num_masked = mask[top: top + h, left: left + w].sum()
                # Overlap
                if 0 < h * w - num_masked <= max_mask_patches:
                    for i in range(top, top + h):
                        for j in range(left, left + w):
                            if mask[i, j] == 0:
                                mask[i, j] = 1
                                delta += 1
                if delta > 0:
                    break

        return delta

    def __call__(self):
        mask = np.zeros(shape=self.get_shape(), dtype=np.int)
        mask_count = 0
        while mask_count < self.num_masking_patches:
            max_mask_patches = self.num_masking_patches - mask_count
            max_mask_patches = min(max_mask_patches, self.max_num_patches)
            delta = self._mask(mask, max_mask_patches)
            if delta == 0:
                break
            else:
                mask_count += delta

        return mask


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


class DataAugmentationBEiT(object):
    """
    Adapted from the official codebase of BEiT: https://github.com/microsoft/unilm/tree/master/beit
    """
    def __init__(self, args):
        try:
            import dall_e
        except ModuleNotFoundError:
            raise ModuleNotFoundError("DALL-E is not found. Please install it with `pip install DALL-E`")

        from dall_e.utils import map_pixels
        self.common_transform = transforms.Compose([
            transforms.ColorJitter(0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(p=0.5),
            RandomResizedCropAndInterpolationWithTwoPic(size=224, second_size=112,
                                                        interpolation="bicubic", second_interpolation="lanczos")
        ])
        self.patch_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        self.visual_token_transform = transforms.Compose([
            transforms.ToTensor(),
            map_pixels
        ])
        self.masked_position_generator = BlockwiseMasking((14, 14), num_masking_patches=75, min_num_patches=16)

    def __call__(self, image):
        patch_image, token_image = self.common_transform(image)
        output = [self.patch_transform(patch_image), self.visual_token_transform(token_image),
                  self.masked_position_generator()]
        return output


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
