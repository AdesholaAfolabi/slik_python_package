import os


def create_absolute_path(dir_path):
    """
    Creates an absolute path given the provided directory

    :param dir_path: Current path to file
    :return: Absolute path
    """

    return os.path.join(os.getcwd(), dir_path)


INTRO_VIDEO_PATH = create_absolute_path("../data/video_data/Intro_video.mp4")
DATA_LOAD_IMAGE_PATH = create_absolute_path("../data/image_data/slik_wrangler_logo.jpeg")
