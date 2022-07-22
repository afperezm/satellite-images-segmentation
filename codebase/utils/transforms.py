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


class AggregateLabel(object):
    """
    Sum label pixels across width and height dimensions and thresholds the result.
    """

    def __call__(self, sample):

        image, label = sample['image'], sample['label']

        label = label.sum(axis=(0, 1)) > 0

        return {'image': image, 'label': label}


class Normalize(object):
    """
    Applies min-max normalization with percentiles cropping.
    """

    def __init__(self, min_value=0.0, max_value=1.0, lower_percent=0, higher_percent=100):

        self.min_value = min_value
        self.max_value = max_value

        self.lower_percent = lower_percent
        self.higher_percent = higher_percent

    def __call__(self, sample):

        image, label = sample['image'], sample['label']

        # Build array of same size than input bands
        out = np.zeros_like(image)

        # Retrieve number of bands or channels
        num_bands = image.shape[2]

        for band_idx in range(num_bands):
            # Compute 5% and 95% percentile values
            lower_percentile = np.percentile(image[:, :, band_idx], self.lower_percent)
            # higher_percentile = np.percentile(image[:, :, band_idx], self.higher_percent)
            higher_percentile = np.mean(image[:, :, band_idx]) + 5 * np.std(image[:, :, band_idx])
            # Apply min-max normalization
            t = self.min_value + (image[:, :, band_idx] - lower_percentile) / (higher_percentile - lower_percentile)
            # Apply new range scaling
            t = t * (self.max_value - self.min_value)
            # Apply threshold for values smaller or higher than the 5% and 95% percentile values
            t[t < self.min_value] = self.min_value
            t[t > self.max_value] = self.max_value
            # Save normalized band on the corresponding channel
            out[:, :, band_idx] = t

        image = out.astype(np.float32)

        return {'image': image, 'label': label}


class Rescale(object):
    """
    Applies rescaling between -1 and 1.
    """

    def __call__(self, sample):

        image, label = sample['image'], sample['label']

        # Rescaling
        image = 2 * image - 1

        return {'image': image, 'label': label}


class ToTensor(object):
    """
    Convert ndarrays in sample to tensors.
    """

    def __call__(self, sample):

        image, label = sample['image'], sample['label']

        # Swap axis to place number of channels in front
        image = np.transpose(image, (2, 0, 1))

        return {'image': torch.from_numpy(image), 'label': torch.tensor(label)}
