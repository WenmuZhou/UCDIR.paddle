import paddle
import os
import random
import numpy as np
from PIL import Image
from PIL import ImageFilter
import paddle.vision.transforms as transforms
from paddle.vision.transforms import functional as F


class RandomApply(transforms.BaseTransform):
    """Apply randomly a list of transformations with a given probability.

    .. note::
        In order to script the transformation, please use ``torch.nn.ModuleList`` as input instead of list/tuple of
        transforms as shown below:

        >>> transforms = transforms.RandomApply(torch.nn.ModuleList([
        >>>     transforms.ColorJitter(),
        >>> ]), p=0.3)
        >>> scripted_transforms = torch.jit.script(transforms)

        Make sure to use only scriptable transformations, i.e. that work with ``torch.Tensor``, does not require
        `lambda` functions or ``PIL.Image``.

    Args:
        transforms (sequence or torch.nn.Module): list of transformations
        p (float): probability
    """

    def __init__(self, transforms, p=0.5):
        super().__init__()
        self.transforms = transforms
        self.p = p

    def _apply_image(self, img):
        if self.p < random.random():
            return img
        for t in self.transforms:
            img = t(img)
        return img


class GaussianBlur(object):

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

class RandomGrayscale(transforms.BaseTransform):
    """Randomly convert image to grayscale with a probability of p (default 0.1).
    If the image is torch Tensor, it is expected
    to have [..., 3, H, W] shape, where ... means an arbitrary number of leading dimensions

    Args:
        p (float): probability that image should be converted to grayscale.

    Returns:
        PIL Image or Tensor: Grayscale version of the input image with probability p and unchanged
        with probability (1-p).
        - If input image is 1 channel: grayscale version is 1 channel
        - If input image is 3 channel: grayscale version is 3 channel with r == g == b

    """

    def __init__(self, p=0.1):
        super().__init__()
        self.p = p

    def _apply_image(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be converted to grayscale.

        Returns:
            PIL Image or Tensor: Randomly grayscaled image.
        """
        if random.random() < self.p:
            return F.to_grayscale(img)
        return img



def folder_content_getter(folder_path):

    cate_names = list(np.sort(os.listdir(folder_path)))

    image_path_list = []
    image_cate_list = []

    for cate_name in cate_names:
        sub_folder_path = os.path.join(folder_path, cate_name)
        if os.path.isdir(sub_folder_path):
            image_names = list(np.sort(os.listdir(sub_folder_path)))
            for image_name in image_names:
                image_path = os.path.join(sub_folder_path, image_name)
                image_path_list.append(image_path)
                image_cate_list.append(cate_names.index(cate_name))

    return image_path_list, image_cate_list


class EvalDataset(paddle.io.Dataset):

    def __init__(self, datasetA_dir, datasetB_dir):
        self.datasetA_dir = datasetA_dir
        self.datasetB_dir = datasetB_dir
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
            0.229, 0.224, 0.225])
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224), 
            transforms.ToTensor(), 
            normalize])
        self.image_paths_A, self.image_cates_A = folder_content_getter(
            datasetA_dir)
        self.image_paths_B, self.image_cates_B = folder_content_getter(
            datasetB_dir)
        self.domainA_size = len(self.image_paths_A)
        self.domainB_size = len(self.image_paths_B)

    def __getitem__(self, index):
        index_A = np.mod(index, self.domainA_size)
        index_B = np.mod(index, self.domainB_size)
        image_path_A = self.image_paths_A[index_A]
        image_path_B = self.image_paths_B[index_B]
        image_A = self.transform(Image.open(image_path_A).convert('RGB'))
        image_B = self.transform(Image.open(image_path_B).convert('RGB'))
        target_A = self.image_cates_A[index_A]
        target_B = self.image_cates_B[index_B]
        return image_A, index_A, target_A, image_B, index_B, target_B

    def __len__(self):
        return max(self.domainA_size, self.domainB_size)


class TrainDataset(paddle.io.Dataset):

    def __init__(self, datasetA_dir, datasetB_dir, aug_plus):
        self.datasetA_dir = datasetA_dir
        self.datasetB_dir = datasetB_dir
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
            0.229, 0.224, 0.225])
        if aug_plus:
            self.transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
                    RandomApply([
                        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                    ], p=0.8),
                    RandomGrayscale(p=0.2),
                    RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
                    RandomGrayscale(p=0.2),
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize
                ]
            )
        self.image_paths_A, self.image_cates_A = folder_content_getter(
            datasetA_dir)
        self.image_paths_B, self.image_cates_B = folder_content_getter(
            datasetB_dir)
        self.domainA_size = len(self.image_paths_A)
        self.domainB_size = len(self.image_paths_B)

    def __getitem__(self, index):
        if index >= self.domainA_size:
            index_A = random.randint(0, self.domainA_size - 1)
        else:
            index_A = index
        if index >= self.domainB_size:
            index_B = random.randint(0, self.domainB_size - 1)
        else:
            index_B = index
        image_path_A = self.image_paths_A[index_A]
        image_path_B = self.image_paths_B[index_B]
        x_A = Image.open(image_path_A).convert('RGB')
        q_A = self.transform(x_A)
        k_A = self.transform(x_A)
        x_B = Image.open(image_path_B).convert('RGB')
        q_B = self.transform(x_B)
        k_B = self.transform(x_B)
        target_A = self.image_cates_A[index_A]
        target_B = self.image_cates_B[index_B]
        return [q_A, k_A], index_A, [q_B, k_B], index_B, target_A, target_B

    def __len__(self):
        return max(self.domainA_size, self.domainB_size)
