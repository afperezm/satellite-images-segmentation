import argparse
import math
import time
import torch

from codebase.data.satellite import PermafrostDataset
from codebase.models.classification import resnet18
from codebase.utils.transforms import RandomHorizontalFlip, RandomVerticalFlip, RandomRotation, ToTensor
from codebase.utils.metrics import binary_accuracy
from torchvision import transforms
from torch.utils.data import DataLoader

DEVICE = None
PARAMS = None


def main():
    # https://towardsdatascience.com/from-pytorch-to-pytorch-lightning-a-gentle-introduction-b371b7caaf09
    # https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

    data_dir = PARAMS.data_dir
    batch_size = PARAMS.batch_size
    epochs = PARAMS.epochs
    learning_rate = PARAMS.learning_rate
    scheduler_step = PARAMS.scheduler_step
    checkpoint_dir = PARAMS.checkpoint_dir
    display_freq = PARAMS.display_freq

    # 1. Load and normalize Permafrost dataset
    transform = transforms.Compose([
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        RandomRotation(),
        ToTensor()])

    train_dataset = PermafrostDataset(data_dir=data_dir,
                                      class_id=0,
                                      is_train=True,
                                      transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True, num_workers=1)

    test_dataset = PermafrostDataset(data_dir=data_dir,
                                     class_id=0,
                                     is_train=False,
                                     transform=transform)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size,
                                 shuffle=False, num_workers=1)

    data_loaders = {'train': train_dataloader, 'test': test_dataloader}

    # 2. Define a Convolutional Neural Network

    model = resnet18(num_classes=1, num_channels=7).to(DEVICE)

    # 3. Define a Loss function and optimizer

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=0.1)

    # 4. Train the network

    best_accuracy = 0.0

    since = time.time()

    for epoch in range(epochs):

        print('Epoch {}/{}'.format(epoch + 1, epochs))
        print('-' * 10)

        for phase in ['train', 'val']:

            # Set model mode
            if phase == 'train':
                model.train()
            else:
                model.eval()

            # Init counters
            num_samples = 0
            running_loss = 0.0
            running_accuracy = 0.0

            for batch_idx, sample in enumerate(data_loaders[phase]):
                # Move inputs and labels to GPU
                inputs, labels = sample['image'].to(DEVICE), sample['label'].to(DEVICE)

                # Cast labels
                labels = labels.float()

                # Zero the parameter gradients
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    # Forward
                    outputs = model(inputs)
                    # Compute prediction error and accuracy
                    loss = criterion(outputs, labels.unsqueeze(1))
                    accuracy = binary_accuracy(outputs, labels.unsqueeze(1))

                    # Backpropagation
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Compute metrics
                num_samples += inputs.size(0)
                running_loss += loss.item() * inputs.size(0)
                running_accuracy += accuracy.item() * inputs.size(0)

                # Print metrics
                if phase == 'train' and batch_idx % display_freq == 0:
                    num_batches = math.ceil(len(train_dataset) / batch_size)
                    batch_loss = running_loss / num_samples
                    batch_accuracy = running_accuracy / num_samples
                    print(f"loss: {batch_loss:>7f}  accuracy: {batch_accuracy:>7f}  [{batch_idx + 1:>5d}/{num_batches:>5d}]")

            # Advance learning rate scheduler
            if phase == 'train':
                scheduler.step()

            # Print epoch metrics
            epoch_loss = running_loss / len(train_dataset)
            epoch_accuracy = running_accuracy / len(train_dataset)
            print(f'Finished {phase}ing - loss: {epoch_loss:.4f} - accuracy: {epoch_accuracy:.4f}')

            # Save new better model
            if phase == 'val' and epoch_accuracy > best_accuracy:
                best_accuracy = epoch_accuracy
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, f'{checkpoint_dir}/best_dev.pt')
                print(f"Saved new best validating model with loss {best_accuracy} to: {checkpoint_dir}/best_dev.pt")

        print()

    time_elapsed = time.time() - since

    print(f'Training completed in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best validation accuracy: {best_accuracy:4f}')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", help="Dataset directory", required=True)
    parser.add_argument("--batch_size", help="Batch size", type=int, required=True)
    parser.add_argument("--epochs", help="Number of epochs", type=int, required=True)
    parser.add_argument("--learning_rate", help="Learning rate", default=0.0001, type=float, required=True)
    parser.add_argument("--scheduler_step", help="Period of learning rate decay", default=10, type=int, required=True)
    parser.add_argument("--checkpoint_dir", help="Summary directory", required=True)
    parser.add_argument("--display_freq", help="Display frequency", default=10, type=int)

    return parser.parse_args()


if __name__ == "__main__":
    PARAMS = parse_args()

    # Get cpu or gpu device for training.
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {device} device".format(device=DEVICE.upper()))

    main()
