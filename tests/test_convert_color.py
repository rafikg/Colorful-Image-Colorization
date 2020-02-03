import pytest
import numpy as np
from utils import lab_to_rgb, rgb_to_lab


@pytest.fixture
def generate_random_rgb_image():
    orig_rgb = np.random.rand(2, 2, 3)
    return orig_rgb


def test_convert_color(generate_random_rgb_image):
    eps = 1e-3
    lab = rgb_to_lab(generate_random_rgb_image)
    rec_rgb = lab_to_rgb(lab)
    val = np.sum(sum(rec_rgb.numpy() - generate_random_rgb_image))
    assert val <= eps
