import mmap
import os

from tqdm import tqdm


def count_lines_fast(filename):
    """
    Count lines in a file quickly using mmap. For very large training data sets,
    we want to be able to show progress as each line is read. To show progress,
    we need to know the line count, which itself is time consuming to determine.

    This function uses mmap to count the number of lines in a file, which is
    much faster than counting the lines using the built-in count() function.
    """
    f = open(filename, "r+")
    buf = mmap.mmap(f.fileno(), 0)
    lines = 0
    read_size = 1024 * 1024
    file_size = os.path.getsize(filename)

    with tqdm(
        total=file_size, unit="B", unit_scale=True, desc="Counting lines"
    ) as pbar:
        while buf.tell() < file_size:
            lines += buf.read(read_size).count(b"\n")
            pbar.update(read_size)

    f.close()
    return lines
