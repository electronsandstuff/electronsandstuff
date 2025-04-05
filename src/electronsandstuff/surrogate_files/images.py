from PIL import Image
import os
from typing import Tuple


def n_files(path, max_dir_size=100, depth=1) -> Tuple[int, int]:
    """
    Get the number of files inside of a nested folder assuming the directories except for the last numbered in each folder are
    completely full and everything has a consistent depth.

    Parameters
    ----------
    path : str
        Path to the file archive
    max_dir_size : int, optional
        Maximum number of files allowed in each directory, by default 100
    depth : int, optional
        Current depth (for recursion, don't change), by default 1

    Returns
    -------
    Tuple[int, int]
        The number of files and maximum detected depth of the file structure.
    """
    # Get the files in numeric order
    files = os.listdir(path)
    files = sorted(files, key=lambda x: int(os.path.splitext(x)[0]))

    # Get if there are files or dirs
    has_files = any(os.path.isfile(os.path.join(path, f)) for f in files)
    has_dirs = any(os.path.isdir(os.path.join(path, f)) for f in files)

    # If there are no files
    if not has_dirs and not has_files:
        return 0, depth

    # If we are in a directory, recurse
    elif has_dirs and not has_files:
        # The number of files in the potentially partially full directory
        n_files_last, max_depth = n_files(
            os.path.join(path, files[-1]), max_dir_size=max_dir_size, depth=depth + 1
        )

        # The number of files contained in all of the full folders
        n_filled = (len(files) - 1) * max_dir_size ** (max_depth - depth)

        # Sum it
        return (n_files_last + n_filled), max_depth

    # If we are in a files directory
    elif has_files and not has_dirs:
        return len(files), depth

    # Something went wrong
    else:
        raise ValueError(
            "Detected directory with directories and files, potentially corrupted image archive."
        )


class ImageManager:
    def __init__(self, dirname, depth=3, extension=".png"):
        self.dirname = dirname
        self.depth = depth
        self.extension = extension

    def add_image(self, img: Image):
        # Find our index
        if not os.path.exists(self.dirname):
            idx = 0
            os.makedirs(self.dirname)
        else:
            idx = len(self)

        # Turn into path, save
        rel_fname = self._get_filename(idx, self.depth, self.extension)
        fname = os.path.join(self.dirname, rel_fname)
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        img.save(fname)
        return rel_fname

    def get_image(self, path: str) -> Image:
        return Image.open(os.path.join(self.dirname, path))

    def __len__(self):
        return n_files(self.dirname)[0]

    def _get_filename(self, idx: int, depth: int = 3, extension: str = ".png") -> str:
        """
        Generate filename from file index.

        Parameters
        ----------
        idx : int
            The index of the file being saved.
        depth : int, optional
            How many directories deep to go, can store 100**depth files.
            Default is 1.
        extension : str, optional
            File extension to add. Default is ".png".

        Returns
        -------
        str
            The filename with appropriate directory structure.
        """
        name = (f"%0{depth * 2}d") % idx
        name = "/".join(["".join(x) for x in zip(*(iter(name),) * 2)]) + extension
        return name
