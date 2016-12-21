import os


def load(file):
    """
    Load file with the given name under the data_files/ folder.

    Args:
        file (str): a file name (e.g., 'dom_casmurro.txt')

    Returns:
        A string with the file contents.
    """

    base = os.path.dirname(__file__)
    path = os.path.join(base, 'data_files', file)
    with open(path) as F:
        data = F.read()
    return data


def speeches():
    """
    Return list of speeches in speeches.csv.
    """

    import csv

    base = os.path.dirname(__file__)
    path = os.path.join(base, 'data_files', 'speeches.csv')
    with open(path) as F:
        csv_reader = csv.reader(F)
        rows = iter(csv_reader)
        next(rows)  # header
        data = [row[-2] for row in rows]
    return data
