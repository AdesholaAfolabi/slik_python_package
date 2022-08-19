import os
import random


def create_absolute_path(dir_path):
    """
    Creates an absolute path given the provided directory

    :param dir_path: Current path to file
    :return: Absolute path
    """

    return os.path.join(os.getcwd(), dir_path)


def randomly_generate_plain_image_path():
    """
    Randomly generates one of the 10 plain images
    available in the package
    """

    return create_absolute_path(f"../data/image_data/plain{random.randint(0, 9)}.png")


INTRO_VIDEO_PATH = create_absolute_path("../data/video_data/Intro_video.mp4")
DATA_LOAD_IMAGE_PATH = create_absolute_path("../data/image_data/slik_wrangler_logo.jpeg")
SITE_LOGO_PATH = create_absolute_path("../data/image_data/sw.png")
