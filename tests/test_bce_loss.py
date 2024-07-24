import tensorflow as tf
import torch
import torch.nn as nn
import numpy as np


def test_bce_loss_wo_reduction():
    TOL = 0.001
    arr = np.random.uniform(-2.0, 2.0, size=(1, 8, 8, 3))

    tf_arr = tf.convert_to_tensor(arr)
    tf_op = tf.keras.losses.BinaryCrossentropy(
        from_logits=True,
    )(y_true=tf.ones_like(tf_arr), y_pred=tf_arr)

    torch_arr = torch.from_numpy(arr).permute(0, 3, 1, 2)
    torch_op = nn.BCEWithLogitsLoss()(torch_arr, torch.ones_like(torch_arr))

    assert np.abs(tf_op.numpy() - torch_op.item()) < TOL
