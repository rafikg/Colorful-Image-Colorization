import pytest
import skimage.color as clr
import numpy as np
from utils import lab_to_rgb, rgb_to_lab

eps = 1e-3


@pytest.fixture
def generate_random_rgb_image():
    random_image = np.random.randint(low=0, high=256, size=(100, 100, 3))
    return random_image


# test reconstruction rgb_to_lab -> lab_to_rgb
def test_rec_rgb_to_lab(generate_random_rgb_image):
    rgb_image = generate_random_rgb_image
    w, h, _ = rgb_image.shape
    rgb_image = rgb_image / 255
    lab = rgb_to_lab(rgb_image)
    rec_rgb = lab_to_rgb(lab)
    val = np.sum(sum(rec_rgb.numpy() - rgb_image)) / (w * h)
    assert val <= eps, "wrong reconstruction rgb_to_lab"


# compare with skimage.rgb2lab
def test_compare_with_skimage(generate_random_rgb_image):
    rgb_image = generate_random_rgb_image
    rgb_image = rgb_image / 255
    my_lab = rgb_to_lab(rgb_image)
    skim_lab = clr.rgb2lab(rgb_image)
    val = np.sum(sum(my_lab.numpy() - skim_lab))
    assert val <= eps, "worng conversion rgb_to_lab"
