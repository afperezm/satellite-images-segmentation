import torch


def binary_accuracy(predictions, labels):

    predictions = torch.round(torch.sigmoid(predictions))

    correct_results_sum = (predictions == labels).sum().float()
    accuracy = correct_results_sum / labels.shape[0]

    return accuracy
