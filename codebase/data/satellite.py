import pandas as pd

from codebase.utils.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from utils.data_utils import ImageData


class PermafrostDataset(Dataset):

    train_phase = 'train'
    test_phase = 'test'

    def __init__(self, data_dir, class_id, is_train=True, transform=None):
        super(PermafrostDataset).__init__()

        self.data_dir = data_dir
        self.class_id = class_id
        self.is_train = is_train
        self.transform = transform

        if self.is_train:
            phase = self.train_phase
        else:
            phase = self.test_phase

        self.ground_truth = pd.read_csv(f'{data_dir}/{phase}_wkt.csv',
                                        names=['ImageId', 'ClassType', 'MultipolygonWKT'], skiprows=1)
        self.grid_sizes = pd.read_csv(f'{data_dir}/grid_sizes.csv',
                                      names=['ImageId', 'Xmax', 'Ymin', 'Xmin', 'Ymax'], skiprows=1)

        self.images = list(self.ground_truth[self.ground_truth.ClassType == class_id].ImageId)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        # Select image key
        img_key = self.images[idx]

        # Load image and label
        image_data = ImageData(self.data_dir, img_key, self.ground_truth, self.grid_sizes)
        image_data.create_train_feature()
        image_data.create_label()

        # Retrieve feature and label
        image, label = image_data.train_feature, image_data.label

        # Pick label
        label = label[..., self.class_id:self.class_id + 1].sum(axis=(0, 1))

        sample = {'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample


if __name__ == "__main__":

    train_dataset = PermafrostDataset(data_dir='/home/andresf/data/permafrost-imagery/',
                                      class_id=0,
                                      is_train=False,
                                      transform=transforms.Compose([
                                          ToTensor()
                                      ]))
    train_dataloader = DataLoader(train_dataset, batch_size=4)

    for epoch in range(2):
        print(f"epoch - {epoch}")
        for batch_idx, data in enumerate(train_dataloader):
            images, labels = data['image'], data['label']
            print(f"batch - {batch_idx}")
            print(images.shape)
            print(images.sum())
            print(labels.shape)
            print(labels.sum())
