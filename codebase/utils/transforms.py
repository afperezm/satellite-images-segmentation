import numpy as np
import torch

from skimage import transform


class RandomHorizontalFlip(object):
    """
    Horizontally flip the image in a sample randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        assert isinstance(p, float)
        self.p = p

    def __call__(self, sample):

        image, label = sample['image'], sample['label']

        if np.random.random() < self.p:
            image = np.fliplr(image).copy()

        return {'image': image, 'label': label}


class RandomVerticalFlip(object):
    """
    Vertically flip the image in a sample randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        assert isinstance(p, float)
        self.p = p

    def __call__(self, sample):

        image, label = sample['image'], sample['label']

        if np.random.random() < self.p:
            image = np.flipud(image).copy()

        return {'image': image, 'label': label}


class RandomRotation(object):
    """
    Randomly rotate between -180 and +180 degrees the image in a sample with a uniform probability.

    Args:
        resize (boolean): Expand the image to fit
    """

    def __init__(self, resize=False):

        assert isinstance(resize, bool)

        self.resize = resize

    def __call__(self, sample):

        image, label = sample['image'], sample['label']

        angle = 90 * np.random.randint(-2, 2)

        image = transform.rotate(image, angle, self.resize)

        return {'image': image, 'label': label}


class ToTensor(object):
    """Convert ndarrays in sample to tensors."""

    def __call__(self, sample):

        image, label = sample['image'], sample['label']

        # Swap axis to place number of channels in front
        image = np.transpose(image, (2, 0, 1))

        return {'image': torch.from_numpy(image),
                'label': torch.tensor(label)}
