import os


def root_dir(path=''):
    # return the path if it's already absolute
    if path and os.path.isabs(path):
        return path

    res_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    if path:
        res_path = os.path.join(res_path, path)

    return res_path
