import os
import tensorflow as tf
from torchvision.io import read_image
import numpy as np
import torch


IMAGE_SIZE = [256, 256]
TOLERANCE = 0.05
TEST_IMAGE = os.path.join(os.getcwd(), "data/monet_jpg/ffd74c77ea.jpg")


def decode_image(img_path=TEST_IMAGE, framework="torch"):
    channels = 3

    if framework == "tf":
        image = tf.io.decode_jpeg(tf.io.read_file(img_path), channels=channels)
        image = (tf.cast(image, tf.float32) / 127.5) - 1
        image = tf.reshape(image, [*IMAGE_SIZE, 3])
        return image
    elif framework == "torch":
        image = read_image(img_path).to(torch.float32) / 127.5 - 1
        return image


def test_input():
    tf_image = decode_image(framework="tf")
    tf_image = tf_image.numpy()

    torch_image = decode_image(framework="torch")
    torch_image = torch_image.permute(1, 2, 0).numpy()

    error = np.abs(tf_image - torch_image).mean()
    assert error < TOLERANCE
