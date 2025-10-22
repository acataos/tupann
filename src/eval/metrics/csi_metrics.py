import torch


def csi_denominator_torch(pred, truth, threshold=2, dim=None):
    binary_pred = torch.nan_to_num(pred) > threshold
    binary_truth = torch.nan_to_num(truth) > threshold
    TP = torch.sum(torch.logical_and(binary_pred, binary_truth), dim=dim)
    FN = torch.sum(torch.logical_and(torch.logical_not(binary_pred), binary_truth), dim=dim)
    FP = torch.sum(torch.logical_and(binary_pred, torch.logical_not(binary_truth)), dim=dim)
    return TP + FN + FP


def true_positive_torch(pred, truth, threshold=2, dim=None):
    binary_pred = torch.nan_to_num(pred) > threshold
    binary_truth = torch.nan_to_num(truth) > threshold
    TP = torch.sum(torch.logical_and(binary_pred, binary_truth), dim=dim)
    return TP
