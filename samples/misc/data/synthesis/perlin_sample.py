from draugr.opencv_utilities import show_image

from neodroidvision.utilities.misc.perlin import generate_perlin_noise

show_image(generate_perlin_noise((100, 100)), wait=True)
