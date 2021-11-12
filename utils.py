import os
import glob
import cv2
from pathlib import Path


def get_files(directory, file_type):
    """Returns a list of all files of the given type in the given directory."""
    directory = Path(directory)
    ori_dir = os.getcwd()
    os.chdir(directory)
    files = [directory / file for file in glob.glob('*.' + file_type)]
    os.chdir(ori_dir)
    return files


def load_image(path, bands=None):
    """Loads a single image file from a provided path. Reduces bands if bands variable is provided."""
    path = Path(path)
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if bands is None:
        return image
    else:
        return image[:, :, :bands]


def save_image(image, path):
    """Stores image in provided path."""
    cv2.imwrite(path, image)
    print('Image saved successfully.')
