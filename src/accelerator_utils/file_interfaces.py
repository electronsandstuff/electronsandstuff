import pandas as pd
import numpy as np
import os, posixpath
from PIL import Image


def get_filename(idx, base_path="files", depth=3, extension=".png"):
    """Generate filename from file index.

    Args:
        idx (_type_):the index
        base_path (str, optional): path to location where files are stored
        depth (int, optional): how many directories deep to go, can store 100**depth files
        extension (str, optional): file extension to add. Defaults to ".png".

    Returns:
        str: the filename
    """
    name = (f"%0{depth*2}d") % idx
    name = "/".join(["".join(x) for x in zip(*(iter(name),)*2)]) + extension
    if base_path:
        name = base_path + '/' + name
    return name


def flatten_table(row_data, name="data", parent="", parent_idx=0, lens={}, files_len=0, base_path="files", extension=".png", file_depth=3):
    """
    Takes row data that looks like a list of dicts with common schema and flattens out elements
    that look like lists into their own tables. Also pulls out files into their own dict with
    filenames that reference back to elements

    Args:
        row_data (list): the rows
        name (str, optional): name of the table being generated
        parent (str, optional): parent to this table
        parent_idx (int, optional): index of the parent to reference in the child
        lens (dict, optional): list of the lengths of tables 

    Returns:
        tuple of dicts: tables (names to rows) and files dict (paths to file binary data)
    """
    tables = {name: []}
    files = {}
    for row_idx, row in enumerate(row_data):
        tables[name].append({})
        if parent:  # Save index of our parent
            tables[name][-1][parent+'_idx'] = parent_idx
        for k, v in row.items():
            if isinstance(v, list) and len(v) and isinstance(v[0], dict):
                my_lens = {k: len(tables.get(k, []))+lens.get(k, 0) for k in set(lens.keys()).union(set(tables.keys()))}
                child_tables, child_files = flatten_table(v, name=k, parent=name, parent_idx=my_lens[name]-1, lens=my_lens, 
                                                          files_len=len(files)+files_len, base_path=base_path, extension=extension, 
                                                          file_depth=file_depth)
                files.update(child_files)
                for k, v in child_tables.items():
                    tables[k] = tables.get(k, []) + v
            elif isinstance(v, bytes):
                tables[name][-1][k] = get_filename(len(files)+files_len, base_path=base_path, extension=extension, depth=file_depth)
                files[tables[name][-1][k]] = v
            elif isinstance(v, np.ndarray) and len(v.shape) == 2:
                tables[name][-1][k] = get_filename(len(files)+files_len, base_path=base_path, extension=extension, depth=file_depth)
                files[tables[name][-1][k]] = v
            else:
                tables[name][-1][k] = v
    return tables, files


def save_homogenous_data(data, path):
    """
    Saves data that looks like a list of dicts with potentially lists and images in elements to
    flattened tables at the specified path

    Args:
        data (list of dicts): the data to save
        path (str): path to save it to (will make directory)
    """
    tables, files = flatten_table(data)
    os.makedirs(path, exist_ok=True)
    for k, v in tables.items():
        pd.DataFrame(v).to_csv(os.path.join(path, k+'.csv'))
    for k, v in files.items():
        fpath = os.path.join(path, k)
        os.makedirs(os.path.dirname(fpath), exist_ok=True)
        if isinstance(v, bytes):
            with open(fpath, 'wb') as f:
                f.write(v)   
        elif isinstance(v, np.ndarray) and len(v.shape) == 2:
            im = Image.fromarray(v)
            im.save(fpath, format='png')
        else:
            raise ValueError(f'Unrecognized type for saving files "{type(v)}"')
            

def load_data_and_images(path):
    """
    Loads all tables and image files from the data stored at path

    Args:
        path (str): path to the directory of data

    Returns:
        tuple: the dict of tables and dict of loaded images
    """
    tables = {}
    files = {}
    for p in os.listdir(path):
        if os.path.splitext(p)[1] == '.csv' and os.path.isfile(os.path.join(path, p)):
            tables[os.path.splitext(p)[0]] = pd.read_csv(os.path.join(path, p))
        else:
            for root, _, fs in os.walk(os.path.join(path, p)):
                for f in fs:
                    if f[0] != '.':
                        files[os.path.relpath(os.path.join(root, f), path).replace(os.sep,posixpath.sep)] = np.asarray(Image.open(os.path.join(root, f)))
    return tables, files


def pixel_coords_from_bbox(upper_left, lower_left, lower_right, image_shape):
    """
    Given bounding box of 2D points, generate grid of points corresponding to pixel coordinates
    for an image of shape image_shape

    Args:
        upper_left (1D np.array 2 elements): coordinates of upper left of image
        lower_left (1D np.array 2 elements): coordinates of lower left of image
        lower_right (1D np.array 2 elements): coordinates of lower right of image
        image_shape (tuple, 2 elements): shape of the image
        
    Returns:
        tuple of np.array: x and y coordinate arrays
    """
    x, y = np.mgrid[0:1:image_shape[0]*1j, 0:1:image_shape[1]*1j]
    mat = np.vstack((
        upper_left - lower_left,
        lower_right - lower_left
    )).T
    x, y = np.reshape(mat@np.vstack((x.ravel(), y.ravel()))+lower_left[:, None], (2, *x.shape))
    return x, y


def pixel_coords_from_row(row, files, prefix="", coords=['r', 'vr']):
    """
    Generates x and y coordinate arrays corresponding to the image stored in the pandas style row (or dict).
    The schema should have:
     - lower_left_x: x coordinate of lower left point of bounding box
     - lower_left_y: y coordinate of lower left point of bounding box
     - lower_right_x: x coordinate of lower right point of bounding box
     - lower_right_y: y coordinate of lower right point of bounding box
     - upper_left_x: x coordinate of upper left point of bounding box
     - upper_left_y: y coordinate of upper left point of bounding box
     - rho: the path to the image
    Args:
        row (dict-like): the dict / pandas row containing bounding box data
        files (dict): the dict of loaded files, to access the image
        prefix (str, optional): a label on the beginning of the density keys in the row. Defaults to "".
        coords (list of strs): a list of two coordinate names for the density image. Defaults to ['r', 'vr'].
    """
    bbox = {}
    for point in ['lower_left', 'lower_right', 'upper_left']:
        bbox[point] = np.array([row[prefix+point+'_'+coord] for coord in coords])
    bbox['upper_right'] = bbox['upper_left'] + bbox['lower_right'] - bbox['lower_left']
    return (x.T for x in pixel_coords_from_bbox(bbox['upper_left'], bbox['lower_left'], bbox['lower_right'], files[row[prefix+'rho']].T.shape))


def get_density_series_bounding_box(df, prefix="", coords=['r', 'vr']):
    """
    Get the axis-aligned bounding box from density images stored as a series of rows in the pandas dataframe

    Args:
        df (pandas dataframe): dataframe with rows that store density image information
        prefix (str, optional): a label on the beginning of the density keys in the row. Defaults to "".
        coords (list of strs): a list of two coordinate names for the density image. Defaults to ['r', 'vr'].

    Returns:
        tuple of dicts: min and max of the coordinates
    """
    df2 = df.copy()
    for coord in coords:
        df2[f'{prefix}upper_right_{coord}'] = df2[f'{prefix}upper_left_{coord}'] + df2[f'{prefix}lower_right_{coord}'] - df2[f'{prefix}lower_left_{coord}']

    mins = {coord: min(df2[f'{prefix}{pt}_{coord}'].min() for pt in ['lower_left', 'lower_right', 'upper_left', 'upper_right']) for coord in coords}
    maxes = {coord: max(df2[f'{prefix}{pt}_{coord}'].max() for pt in ['lower_left', 'lower_right', 'upper_left', 'upper_right']) for coord in coords}
    return mins, maxes
