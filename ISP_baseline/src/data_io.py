# I/O functions
# largely lifted from the src/data/data_io.py
# in the MFISNet repository, added by Olivia Tsang


import logging
import h5py
import numpy as np
from typing import Dict, List, Iterable
import os
import time
import re
import psutil  # to fetch memory usage
import time

MAX_RETRIES = 30
RETRY_SLEEP_DUR = 10

SAMPLE_COMPLETION = "sample_completion"
Q_CART = "q_cart"
D_RS   = "d_rs"
TRUNCATABLE_KEYS = {Q_CART, D_RS, SAMPLE_COMPLETION}

from ISP_baseline.src.noise import add_noise_to_d

### Single-file HDF5 access functions ###

def load_hdf5_to_dict(fp_in, key_replacement: dict = None, retries: int=0) -> Dict:
    """Loads all the fields in a hdf5 file"""
    if retries >= MAX_RETRIES:
        raise IOError(f"(lhtd) Couldn't open load {fp_in} after {MAX_RETRIES} tries")
    key_replacement = key_replacement if key_replacement is not None else {}
    # key replacement function
    krfn = lambda key: key_replacement[key] if key in key_replacement.keys() else key
    # destination dictionary
    data_dict = {}
    try:
        with h5py.File(fp_in, "r") as hf:
            data_dict = {krfn(key): val[()] for (key, val) in hf.items()}
    except BlockingIOError:
        logging.warning(f"File {fp_in} is blocked on attempt {retries}")
        time.sleep(RETRY_SLEEP_DUR)
        return load_hdf5_to_dict(fp_in, key_replacement, retries + 1)
    return data_dict


def save_dict_to_hdf5(
    data_dict: Dict,
    fp_out: str,
    key_replacement: dict = None,
    retries: int=0,
) -> None:
    """Saves a dictionary as a hdf5 file at path fp_out"""
    key_replacement = key_replacement if key_replacement is not None else {}
    # key replacement function
    krfn = lambda key: key_replacement[key] if key in key_replacement.keys() else key

    if retries >= MAX_RETRIES:
        raise IOError(f"(sdth) Couldn't open file {fp_out} after {MAX_RETRIES} tries")
    try:
        with h5py.File(fp_out, "w") as hf:
            for key, val in data_dict.items():
                hf.create_dataset(krfn(key), data=val)
    except BlockingIOError:
        logging.warning(f"File {fp_out} is blocked; {retries} retries remaining")
        time.sleep(RETRY_SLEEP_DUR)
        save_dict_to_hdf5(data_dict, fp_out, key_replacement, retries+1)



### Helper functions to operate on Individual Fields ###

def save_field_to_hdf5(
    key: str, data: np.ndarray, fp_out: str, overwrite: bool = True, retries: int = 0
) -> None:
    """Saves an individual array to the specified field in a given hdf5 file
    Note that this operation may squash the old field
    """
    if not os.path.exists(fp_out):
        raise FileNotFoundError
    if retries >= MAX_RETRIES:
        raise IOError(f"(sfth) Couldn't open file {fp_out} after {MAX_RETRIES} tries")
    try:
        with h5py.File(fp_out, "a") as hf:
            # logging.debug(f"sfth df keys before: {hf.keys()}")
            if key in hf.keys() and not overwrite:
                raise KeyError(
                    f"Attempted to write to key {key} in {fp_out} "
                    "which already exists (and overwrite mode=False)"
                )
            elif key in hf.keys() and overwrite:
                # Need to handle the case where the dataset already exists...
                # See the update_field_in_hdf5 function for ideas maybe?
                # pass
                dset = hf.require_dataset(key, shape=data.shape, dtype=data.dtype)
                dset.write_direct(data)
            else:
                # Create new entry
                hf.create_dataset(key, data=data)
            # logging.debug(f"sfth df keys after:  {hf.keys()}")

    except BlockingIOError:
        logging.warning("File is blocked; on retry # %i", retries)
        time.sleep(RETRY_SLEEP_DUR)
        save_field_to_hdf5(key, data, fp_out, retries + 1)

# Provide an alias for naming consistency but leave the original version
# intact to avoid breaking anything
def save_field_in_hdf5(
    key: str, data: np.ndarray, fp_out: str, overwrite: bool = True, retries: int = 0
) -> None:
    """Saves an individual array to the specified field in a given hdf5 file
    Note that this operation may squash the old field!

    Alias for save_field_to_hdf5 for better consistency
    with the other field-specific helper functions
    """
    save_field_to_hdf5(key, data, fp_out, overwrite=overwrite, retries=retries)


def load_field_in_hdf5(
    key: str, fp_out: str, idx_slice=slice(None), retries: int = 0
) -> np.ndarray:
    """Loads an individual field to the specified field in a given hdf5 file"""
    if not os.path.exists(fp_out):
        raise FileNotFoundError("Can't load field %s from %s" % (key, fp_out))
    if retries >= MAX_RETRIES:
        raise IOError(f"(load_field_in_hdf5) Couldn't open file after {MAX_RETRIES} tries")
    try:
        with h5py.File(fp_out, "r") as hf:
            data_loaded = hf[key][()]
            data = (
                data_loaded[idx_slice]
                if isinstance(data_loaded, np.ndarray)
                else data_loaded
            )
        return data

    except BlockingIOError:
        logging.warning("File is blocked; on retry # %i", retries)
        time.sleep(RETRY_SLEEP_DUR)
        return load_field_in_hdf5(key, fp_out, idx_slice, retries + 1)
    # except:
    #     import pdb; pdb.set_trace()


def update_field_in_hdf5(
    key: str, data: np.ndarray, fp_out: str, idx_slice=slice(None), retries: int = 0
) -> None:
    """Saves an individual array to a slice in the specified field in a given hdf5 file
    Note that this operation may squash the old field
    """
    if not os.path.exists(fp_out):
        raise FileNotFoundError
    if retries >= MAX_RETRIES:
        raise IOError(f"(update_field_in_hdf5) Couldn't open file after {MAX_RETRIES} tries")

    try:
        with h5py.File(fp_out, "a") as hf:
            data_loaded = hf[key][()]
            data_loaded[idx_slice] = data
            dset = hf.require_dataset(
                key, shape=data_loaded.shape, dtype=data_loaded.dtype
            )
            dset.write_direct(data_loaded)
    except KeyError:
        # In case the field was not located, just make a new one...
        save_field_to_hdf5(key, data, fp_out, retries)

    except BlockingIOError:
        logging.warning("File is blocked; on retry # %i", retries)
        time.sleep(RETRY_SLEEP_DUR)
        update_field_in_hdf5(key, data, fp_out, idx_slice, retries + 1)

def get_fields_in_hdf5(
    fp_in: str,
    retries: int=0,
    require_file_exist: bool=True,
) -> list:
    """Helper function to get a list of all the fields in a given hdf5 file

    Parameters:
        fp_in (str): file path of a the desired file to check
        retries (int): number of times to retry in case the file is busy
    Outputs:
        all_keys (list): a list of all the keys encountered
            if the file is not found but require_file_exist
            is set to False this will be an empty list
    """
    if not os.path.exists(fp_in):
        raise FileNotFoundError
    if retries >= MAX_RETRIES:
        raise IOError(f"(get_fields_in_hdf5) Couldn't open file after {MAX_RETRIES} tries")

    all_fields = []
    try:
        with h5py.File(fp_in, "r") as hf:
            all_fields = list(hf.keys())
    except BlockingIOError:
        logging.warning("File is blocked; on retry # %i", retries)
        time.sleep(RETRY_SLEEP_DUR)
        return get_fields_in_hdf5(fp_in, retries + 1)
    except FileNotFoundError:
        if require_file_exist:
            raise # let the error propagate up if we require the file to exist
    return all_fields

### Helper functions for directory-wide HDF5 loading ###
# Define a custom sorting key function
def _get_number_from_filename(filename: str) -> int:
    """Assumes a file has format .*_{number}.h5 and extracts the number"""
    try:
        f = filename.split("_")[-1]
        num = int(f.split(".")[0])
    except:
        raise ValueError(f"_get_number_from_filename: unable to properly parse the filename {filename}")
    return num

def _get_valid_idcs(arr: np.ndarray) -> np.ndarray:
    """Checks the array for NaNs.
    Parameters:
        arr (np.ndarray): expects shape (N_samples, ...)
    Returns:
        out (np.ndarray): 1-dim array with shape (N_samples,) indicating
            whether any entry for a given sample contains a nan
    """
    out = np.logical_not(np.any(np.isnan(arr), axis=tuple(range(1,arr.ndim))))
    return out


##### Loading slices of a dataset, not necessarily file-by-file #####

# Extra helpers
def get_file_start_index(file_path, use_file_name: bool=True):
    """There are multiple scattering object or measurement files in the directory
    So, find out what the starting index is of this file if we were to load the
    entire directory.
    """
    if use_file_name:
        # Assume that the file name follows the convention of including
        # the index corresponding to the first sample it contains
        start_index = _get_number_from_filename(file_path)
    else:
        # In case you don't trust the file names...
        # can load all the files in the directory and tally up the
        # number of samples using the SAMPLE_COMPLETION field
        dir_name, file_name = os.path.split(file_path)[0]
        file_list = os.listdir(dir_name)
        file_list = sorted(file_list, key=_get_number_from_filename)
        sample_count = 0
        for file_i in file_list:
            if fp == file_name:
                break
            file_i = os.path.join(dir_name, file_i)
            file_i_samples = load_field_in_hdf5(SAMPLE_COMPLETION, file_i).shape[0]
            sample_count += file_i_samples
        start_index = sample_count
    return start_index

def find_files_index_range(
    dir_name: str,
    global_idx_start: int=0,
    global_idx_end: int=None,
) -> List[str]:
    """For a given directory, return the list of files (and corresponding slices)
    to extract from each file in order to get the samples from index_start to
    index_end
    """
    file_list = [
        file_name
        for file_name in os.listdir(dir_name)
        if len(re.findall("[0-9]+", file_name.replace(".h5", ""))) >= 1
    ]
    file_list = sorted(file_list, key=_get_number_from_filename)
    file_list = [os.path.join(dir_name, file_name) for file_name in file_list]

    # Calculate the starting indices of each file in the directory
    # Stop once we have all the files within global_idx_start and global_idx_end
    file_index_list = []
    for file_i in file_list:
        file_index_list.append(
            get_file_start_index(file_i, use_file_name=True)
        )

    # Also add the length of the last file for easier index manipulation
    sample_count = (
        load_field_in_hdf5(SAMPLE_COMPLETION, file_list[-1]).shape[0]
        + file_index_list[-1]
    )
    file_index_list.append(sample_count)
    # Update the global slice range in case the end index was previously unspecified
    global_idx_end = global_idx_end if global_idx_end is not None else sample_count
    # Collect the starts/ends etc. as np arrays
    file_index_arr    = np.array(file_index_list)
    file_index_starts = file_index_arr[:-1]
    file_index_ends   = file_index_arr[1:]

    # Calculate which files contain the requested range
    # dense bool array indicating inclusion/exclusion
    valid_files_bools = np.logical_and(
        file_index_ends   >  global_idx_start,
        file_index_starts <= global_idx_end,
    )
    # sparse version
    valid_files = np.argwhere(
        valid_files_bools
    ).flatten()
    valid_starts = file_index_starts[valid_files]
    valid_ends   = file_index_ends[valid_files]
    valid_file_fps = [
        file_i
        for (i, file_i)
        in enumerate(file_list)
        if valid_files_bools[i]
    ]

    # Get the corresponding slices
    # but want these in local terms...
    local_slices = [
        slice(
            max(fs, global_idx_start) - fs,
            min(fe, global_idx_end) - fs,
        )
        for
        (fs, fe) in zip(valid_starts, valid_ends)
    ]
    return valid_file_fps, local_slices

def load_single_dir_slice(
    dir_name: str,
    global_idx_start: int = 0,
    global_idx_end: int = None,
    load_keys: Iterable=None,
    ignore_keys: Iterable=None,
    sample_keys: Iterable=None,
) -> dict:
    """Load all the files in a given directory from the desired slice
    Compared to previous implementations, this is meant to be fairly
    agnostic to the field names, though it still offers control by letting
    the user to specify which keys to ignore and which need to be truncated/concatenated

    This is intended for loading single directories.
    Note: it may be worth considering refactoring some of the code above to use this function,
    since it is much simpler and more generic

    Behavior:
        1. Non-concatable keys will be taken from the first valid file
        2. If fields is unspecified, all fields will be loaded, except those in ignore_keys
        3. Fields that are concatable will be concatenated for the return value

    Parameters:
        dir_name (str): directory whose files this function will look through and load
        global_idx_start (int): starting sample index to load, inclusive
        global_idx_end (int): last sample index to load, exclusive to match python conventions
        load_keys (Iterable): optionally can specify which fields to load; defaults to all
        ignore_keys (Iterable): alternately, can specify the fields not to load
        sample_keys(Iterable): the keys corresponding to samples
            the values should be concatenated/truncated as needed
            Note: this always concatenates/truncates along axis 0
    Outputs:
        res_dd (dict):
    """
    global TRUNCATABLE_KEYS, SAMPLE_COMPLETION

    # Basic setup: fetch the relevant files
    valid_file_fps, local_slices = find_files_index_range(
        dir_name, global_idx_start, global_idx_end,
    )
    all_keys = get_fields_in_hdf5(valid_file_fps[0], require_file_exist=True)

    # Basic setup: fetch argument values or set default values
    ignore_keys = ignore_keys if ignore_keys is not None else set()
    sample_keys = sample_keys if sample_keys is not None \
        else {*TRUNCATABLE_KEYS, SAMPLE_COMPLETION}

    # Finish setting up the keys based on what is present
    load_keys = load_keys if load_keys is not None else [
        key for key in all_keys
        # if key not in ignore_keys
    ]
    load_keys = [key for key in load_keys if key not in ignore_keys]
    if any((key not in all_keys) for key in load_keys):
        logging.warning(
            f"Not all the requested keys from load_keys were found in the file. "
            f"load_keys: {load_keys} vs. all keys present: {all_keys}. Ignoring "
            f"the missing keys..."
        )
        load_keys = filter(lambda k: k in all_keys, load_keys)
    # Filter sample_keys so it only coincides with keys that are present
    # and does not include the keys we want to ignore...
    sample_keys = [
        key for key in sample_keys
        if (key not in ignore_keys) and (key in load_keys)
    ]
    # logging.warning(f"sample_keys (2): {sample_keys}")

    # First get the keys only used in the first file
    first_file_keys = [key for key in load_keys if key not in sample_keys]
    res_dd = {
        key: load_field_in_hdf5(key, valid_file_fps[0])
        for key in first_file_keys
    }
    for key in sample_keys:
        res_dd[key] = []

    # Next, load all the concatable keys
    for valid_file_fp, load_slice in zip(valid_file_fps, local_slices):
        for key in sample_keys:
            res_dd[key].append(
                load_field_in_hdf5(key, valid_file_fp, idx_slice=load_slice)
            )
    # Flatten the lists of numpy arrays into numpy arrays
    for key in sample_keys:
        res_dd[key] = np.concatenate(res_dd[key], axis=0)

    # Finished!
    return res_dd

def nan_handler(
    dd: dict,
    nan_mode: str,
    check_fields: list,
    sample_fields: list=None,
) -> dict:
    """Goes through check_fields to find if any samples have nan values
    Returns a dictionary containing auxiliary information about which samples
    were loaded
    nan_mode expected to be one of ["keep", "zero", "skip"]
    """
    nan_mode = nan_mode.lower() if nan_mode is not None else "keep"
    check_fields = check_fields if check_fields is not None else []
    # sample_fields = sample_fields if sample_fields is not None else []
    sample_fields = sample_fields if sample_fields is not None else [
        *TRUNCATABLE_KEYS,
    ]
    sample_fields = set(filter(lambda f: f in dd.keys(), sample_fields))

    # aux_dd = dict()
    keep_idcs = np.logical_and.reduce([
        _get_valid_idcs(dd[key])
        for key in check_fields
        if  key in dd.keys()
    ])
    nan_idcs = np.logical_not(keep_idcs) # boolean array
    num_tot_samples = keep_idcs.shape[0]
    num_nan_samples = np.sum(nan_idcs)
    num_loaded_samples = num_tot_samples # adjust later

    if nan_mode == "keep":
        # Keep the NaNs
        pass
    elif nan_mode == "zero":
        # Zero out all the fields corresponding to any nan indices
        for key in sample_fields:
            dd[key][nan_idcs] = 0
    elif nan_mode == "skip":
        for key in sample_fields:
            dd[key] = dd[key][keep_idcs]
        num_loaded_samples = num_tot_samples - num_nan_samples
    else:
        raise ValueError(
            f"nan_handler received nan_mode={nan_mode} but expects one of "
            f"['keep', 'zero', 'skip']"
        )

    aux_dd = {
        "orig_idcs": keep_idcs,
        "num_tot_samples": num_tot_samples,
        "num_nan_samples": num_nan_samples,
        "num_loaded_samples": num_loaded_samples,
        "num_valid_samples":  num_tot_samples - num_nan_samples
    }
    return dd, aux_dd

def load_multi_dir_slice(
    dir_list,
    global_idx_start: int=0,
    global_idx_end: int=0,
    load_keys: Iterable=None,
    ignore_keys: Iterable=None,
    sample_keys: Iterable=None,
    freq_dep_keys: Iterable=None,
) -> dict:
    """Load multiple directories with an interface mirroring load_single_dir_slice
    """
    freq_dep_keys = freq_dep_keys if freq_dep_keys is not None else set()
    ignore_keys   = ignore_keys if ignore_keys is not None else set()

    # print(f"load_multi_dir_slice received range: {global_idx_start}:{global_idx_end}")
    # print(f"load_multi_dir_slice load_keys   = {load_keys}")
    # print(f"load_multi_dir_slice sample_keys = {sample_keys}")

    # Load the data first
    dd_single = load_single_dir_slice(
        dir_list[-1],
        global_idx_start=global_idx_start,
        global_idx_end=global_idx_end,
        load_keys=load_keys,
        ignore_keys={*freq_dep_keys, *ignore_keys},
        sample_keys=sample_keys,
    )
    # dd_single_shapes = [
    #     f"{key}{val.shape}"
    #     for key, val in dd_single.items()
    # ]
    # print(f"dd single shapes: {', '.join(dd_single_shapes)}")

    # Only load frequency-dependent keys here
    dd_list = [
        load_single_dir_slice(
            dir_name,
            global_idx_start=global_idx_start,
            global_idx_end=global_idx_end,
            load_keys=freq_dep_keys,
            ignore_keys=ignore_keys,
            sample_keys=sample_keys,
        )
        for dir_name in dir_list
    ]

    # Merge the frequency-dependent key entries
    dict_fdk = {
        fdk: np.stack(
            [dd[fdk] for dd in dd_list],
            axis=1,
        )
        for fdk in freq_dep_keys
    }

    dd_all = {
        **{k:v for (k,v) in dd_single.items() if k not in freq_dep_keys},
        **dict_fdk,
    }
    return dd_all

def get_multifreq_dset_dirs(
    dset: str,
    kbar_str_list: list,
    base_dir: str=None,
    dir_fmt: str=None,
):
    """Gets the dataset directories in the multi-frequency setting"""
    base_dir = base_dir if base_dir is not None else ""
    dir_fmt = dir_fmt if dir_fmt is not None else "{0}_train_measurements_nu_{1}"
    dir_list = [
        os.path.join(base_dir, dir_fmt.format(dset, kbar_str))
        for kbar_str in kbar_str_list
    ]
    return dir_list


def load_cart_multifreq_dataset(
    dir_list: list,
    global_idx_start: int=0,
    global_idx_end: int=None,
    add_noise: bool=False,
    noise_to_sig_ratio: float=0,
    noise_seed: int | list=None,
    noise_seed_mode: str = "sequential",
    noise_norm_mode: str = "inf",
):
    """Load just the cartesian parts of the multifrequency dataset
    """
    loaded_dd = load_multi_dir_slice(
        dir_list,
        global_idx_start=global_idx_start,
        global_idx_end=global_idx_end,
        load_keys=[Q_CART, D_RS, SAMPLE_COMPLETION, "x_vals"],
        sample_keys=[Q_CART, D_RS, SAMPLE_COMPLETION],
        freq_dep_keys=[D_RS],
    )
    if add_noise and noise_to_sig_ratio != 0:
        # raise NotImplementedError(f"Noise not supported currently")
        logging.info(
            f"Adding noise at level {noise_to_sig_ratio} with scaling {noise_norm_mode}"
        )
        drs = loaded_dd[D_RS]
        drs_noisy = add_noise_to_d(
            drs,
            noise_to_sig_ratio,
            noise_seed=noise_seed,
            seed_mode=noise_seed_mode,
            norm_mode=noise_norm_mode,

        )
        loaded_dd[D_RS] = drs_noisy
    return loaded_dd

def save_single_dir_slice(
    data_dict: dict,
    dir_name: str,
    file_format: str="scattering_objs_{0}.h5",
    shard_size: int = 1000,
    sample_keys: Iterable=None,
):
    """Save the contents of the data_dict to a single directory slice
    """
    # collection of all keys present
    all_keys = set(data_dict.keys())
    # Keys corresponding to samples that will get split up by file
    sample_keys = sample_keys if sample_keys is not None \
        else all_keys.intersection({*TRUNCATABLE_KEYS, SAMPLE_COMPLETION})
    # Keys common to every file
    common_keys = {key for key in all_keys if key not in sample_keys}

    a_sample_key = list(sample_keys)[0]
    num_samples = data_dict[a_sample_key].shape[0]
    logging.info(f"(save_single_dir_slice) all_keys={all_keys}")
    logging.info(f"(save_single_dir_slice) common_keys={common_keys}")
    logging.info(f"(save_single_dir_slice) sample_keys={sample_keys}")
    logging.info(f"(save_single_dir_slice) num_samples={num_samples}")

    os.makedirs(dir_name, exist_ok=True)
    shard_idx_starts = np.arange(0, num_samples, shard_size, dtype=int)
    shard_names = [
        os.path.join(
            dir_name,
            file_format.format(start_idx)
        )
        for start_idx in shard_idx_starts
    ]
    logging.info(f"(save_single_dir_slice) shard_names={shard_names}")
    for i, start_idx in enumerate(shard_idx_starts):
        shard_name  = shard_names[i]
        start_idx   = start_idx
        end_idx     = min(num_samples, start_idx+shard_size)
        shard_slice = slice(start_idx, end_idx)

        # Prepare the dictionary for this particular shard
        shard_dd = {
            # Every file gets this
            **{k:v for (k,v) in data_dict.items() if k in common_keys},
            # Shard-specific range
            **{k:v[shard_slice] for (k,v) in data_dict.items() if k in sample_keys},
        }
        save_dict_to_hdf5(
            shard_dd,
            shard_name,
        )
    # logging.info(f"(save_single_dir_slice) finished")
    return shard_names
