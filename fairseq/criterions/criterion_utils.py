import torch

from fairseq.data import SmileDictionary


def em_accuracy(pred, target, padding_idx):
    """Exactly Matching Accuracy Operator.
    
    Args:
        pred: Predicted PyTorch Tensor.
        target: Targeted Tensor.
        dic, Dictionary with EOS and PAD.
    
    Returns:
        float: the EM accuracy.
    """
    # Pre-check.
    assert pred.shape == target.shape, 'Input tensor should have same shape.'
    # assert dic is not None, 'Dictionary shoud not be None.'
    """
    deprecated
    eos = dic.eos()
    # Replace all token after EOS with PAD.
    for i in range(pred.size(0)):
        row = pred[i, :].tolist()
        idx = row.index(eos) if eos in row else -1
        pred[i, idx + 1:] = pad
    # Compute equal.
    em = sum((target == pred).sum(1) == pred.size(1))
    # Computer accuracy.
    em_acc = float(em) / target.size(0)
    return em_acc
    """

    # New logic.
    # Set all <pad> index from target into pred since we do not need to consider
    # these positions.
    pred.masked_scatter_(target == padding_idx,
                         torch.ones_like(pred) * padding_idx)
    # Compute EM.
    em = sum((target == pred).sum(1) == pred.size(1))
    # Compute EM acc.
    em_acc = float(em) / target.size(0)
    return em_acc
