# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import glob
import os
import pathlib

import torch

from Data.Generated.CIF import process_cif

from tqdm import tqdm

import numpy as np

from torch import multiprocessing as mp


def hkl(hkl_max=10):
    """
    Calculate {h, k, l} planes based on hkl_max wanted accuracy.
    """
    # First we must define a complete set of (hkl) planes. To cover all cases, we need a automatic algorithm to generate
    # the matrix. hkl groups start from 000 to nnn, here n is the upper limit needed to be set. The larger the n, the
    # more planes we are dealing. from n we define hklMax as the first parameters we want.

    # we use "vstack" to generate hkl matrix one row by one row.
    # we have hkl_info and hkl_add
    hkl_info = np.array([[1, 0, 0]])
    hkl_add = np.zeros((2, 3))
    # Here we start our loop to increase hkl rows one by one sequentially
    hkl_idx = 0
    hkl_h = 1
    while hkl_h <= hkl_max:
        if hkl_info[hkl_idx, 1] == hkl_info[hkl_idx, 2] and hkl_info[hkl_idx, 0] != hkl_info[hkl_idx, 1]:
            hkl_add[0] = hkl_info[hkl_idx]
            hkl_add[0, 1] = hkl_add[0, 1] + 1
            hkl_add[0, 2] = 0
            hkl_info = np.vstack([hkl_info, hkl_add[0]])
        elif hkl_info[hkl_idx, 1] > hkl_info[hkl_idx, 2]:
            hkl_add[0] = hkl_info[hkl_idx]
            hkl_add[0, 2] = hkl_add[0, 2] + 1
            hkl_info = np.vstack([hkl_info, hkl_add[0]])
        elif hkl_info[hkl_idx, 0] == hkl_info[hkl_idx, 1] and hkl_info[hkl_idx, 0] == hkl_info[hkl_idx, 2]:
            hkl_add[0] = hkl_info[hkl_idx]
            hkl_add[0, 0] = hkl_add[0, 0] + 1
            hkl_add[0, 1] = 0
            hkl_add[0, 2] = 0
            if hkl_h != hkl_max:
                hkl_info = np.vstack([hkl_info, hkl_add[0]])
            hkl_h += 1
        hkl_idx += 1
    # Above, "hkl_info" has been calculated

    # Then we need to switch hkl positions to guarantee 100, 010, 001
    # The process is simple, switch 01, 12, 02, then displace one by one abc -> cab -> bca then reduce identical row
    # First, switch 01, 12, 02
    hkl_exp = hkl_info[0, :]

    for i in range(0, hkl_info.shape[0]):
        hkl_switch01 = np.zeros((2, 3))
        hkl_switch01[0, 0] = hkl_info[i, 1]
        hkl_switch01[0, 1] = hkl_info[i, 0]
        hkl_switch01[0, 2] = hkl_info[i, 2]
        hkl_switch12 = np.zeros((2, 3))
        hkl_switch12[0, 0] = hkl_info[i, 0]
        hkl_switch12[0, 1] = hkl_info[i, 2]
        hkl_switch12[0, 2] = hkl_info[i, 1]
        hkl_switch02 = np.zeros((2, 3))
        hkl_switch02[0, 0] = hkl_info[i, 2]
        hkl_switch02[0, 1] = hkl_info[i, 1]
        hkl_switch02[0, 2] = hkl_info[i, 0]
        hkl_displace201 = np.zeros((2, 3))
        hkl_displace201[0, 0] = hkl_info[i, 2]
        hkl_displace201[0, 1] = hkl_info[i, 0]
        hkl_displace201[0, 2] = hkl_info[i, 1]
        hkl_displace120 = np.zeros((2, 3))
        hkl_displace120[0, 0] = hkl_info[i, 1]
        hkl_displace120[0, 1] = hkl_info[i, 2]
        hkl_displace120[0, 2] = hkl_info[i, 0]
        hkl_exp = np.vstack([hkl_exp, hkl_info[i, :]])
        hkl_exp = np.vstack([hkl_exp, hkl_switch01[0]])
        hkl_exp = np.vstack([hkl_exp, hkl_switch12[0]])
        hkl_exp = np.vstack([hkl_exp, hkl_switch02[0]])
        hkl_exp = np.vstack([hkl_exp, hkl_displace201[0]])
        hkl_exp = np.vstack([hkl_exp, hkl_displace120[0]])

    # Then, reduce identical row
    hkl_redu = np.zeros((1, 3))
    hkl_redu[0] = hkl_exp[0]
    # Loop for extract
    for i in range(1, hkl_exp.shape[0]):
        # Loop for line by line comparison
        vstack_judge = True
        if_loop_judge = False
        for j in range(0, hkl_redu.shape[0]):
            if np.array_equal(hkl_exp[i], hkl_redu[j]):
                vstack_judge = False
            if_loop_judge = True
        if vstack_judge and if_loop_judge:
            hkl_redu = np.vstack([hkl_redu, hkl_exp[i]])

    # Now, we put negative signs in the matrix
    # for hkl_exp, we extract every line and then vstack to hkl_exp2\
    hkl_exp2 = hkl_redu[0, :]
    for i in range(1, hkl_redu.shape[0]):
        # 1st case: 2 0s
        if hkl_redu[i, 0]*hkl_redu[i, 1] == 0 and hkl_redu[i, 0]*hkl_redu[i, 2] == 0 \
                and hkl_redu[i, 1]*hkl_redu[i, 2] == 0:
            hkl_exp2 = np.vstack([hkl_exp2, hkl_redu[i, :]])
        # 2nd case: 1 0s
        elif hkl_redu[i, 0] == 0 or hkl_redu[i, 2] == 0 or hkl_redu[i, 1] == 0:
            hkl_one0_1 = np.zeros((2, 3))
            if hkl_redu[i, 2] == 0:
                hkl_one0_1[0, 0] = hkl_redu[i, 0]
                hkl_one0_1[0, 1] = -hkl_redu[i, 1]
                hkl_one0_1[0, 2] = hkl_redu[i, 2]
            elif hkl_redu[i, 0] == 0:
                hkl_one0_1[0, 0] = hkl_redu[i, 0]
                hkl_one0_1[0, 1] = hkl_redu[i, 1]
                hkl_one0_1[0, 2] = -hkl_redu[i, 2]
            elif hkl_redu[i, 1] == 0:
                hkl_one0_1[0, 0] = hkl_redu[i, 0]
                hkl_one0_1[0, 1] = hkl_redu[i, 1]
                hkl_one0_1[0, 2] = -hkl_redu[i, 2]
            hkl_exp2 = np.vstack([hkl_exp2, hkl_redu[i, :]])
            hkl_exp2 = np.vstack([hkl_exp2, hkl_one0_1[0, :]])
        # 3rd case: none 0
        else:
            hkl_none0_1 = np.zeros((2, 3))
            hkl_none0_2 = np.zeros((2, 3))
            hkl_none0_3 = np.zeros((2, 3))
            hkl_none0_1[0, 0] = hkl_redu[i, 0]
            hkl_none0_1[0, 1] = -hkl_redu[i, 1]
            hkl_none0_1[0, 2] = hkl_redu[i, 2]
            hkl_none0_2[0, 0] = hkl_redu[i, 0]
            hkl_none0_2[0, 1] = hkl_redu[i, 1]
            hkl_none0_2[0, 2] = -hkl_redu[i, 2]
            hkl_none0_3[0, 0] = hkl_redu[i, 0]
            hkl_none0_3[0, 1] = -hkl_redu[i, 1]
            hkl_none0_3[0, 2] = -hkl_redu[i, 2]
            hkl_exp2 = np.vstack([hkl_exp2, hkl_redu[i, :]])
            hkl_exp2 = np.vstack([hkl_exp2, hkl_none0_1[0, :]])
            hkl_exp2 = np.vstack([hkl_exp2, hkl_none0_2[0, :]])
            hkl_exp2 = np.vstack([hkl_exp2, hkl_none0_3[0, :]])

    # Then we calculate the multiplicity of each hkl planes. The rules are simply, no 0 -> 4, one 0 -> 2, two 0 -> 1
    hkl_multi = np.zeros(( hkl_exp2.shape[0], 1))
    for i in range(0,  hkl_exp2.shape[0]):
        if hkl_exp2[i, 0] != 0 and hkl_exp2[i, 1] != 0 and hkl_exp2[i, 2] != 0:
            hkl_multi[i] = 1
        elif hkl_exp2[i, 0] * hkl_exp2[i, 1] == 0 and hkl_exp2[i, 0] * hkl_exp2[i, 2] == 0 \
                and hkl_exp2[i, 1] * hkl_exp2[i, 2] == 0:
            hkl_multi[i] = 1
        else:
            hkl_multi[i] = 1
    hkl_final = np.hstack([hkl_exp2, hkl_multi])

    if not os.path.exists(f'{generated_path}/'):
        os.makedirs(f'{generated_path}/')

    np.save(f'{generated_path}/hkl_{hkl_max}.npy', hkl_final)

    return hkl_final


generated_path = str(pathlib.Path(__file__).parent.resolve())


# tqdm starmap compatibility https://stackoverflow.com/a/67845088/22002059
def star(args):
    return process_cif(*args)


def generate(in_dir='./CIFs_open_access/', out_dir='./XRDs_open_access/'):
    """
    Generate XRD data from CIF files located in directory in_dir.
    """

    # Generate hkl matrix up to precision 10
    hkl_max = 10
    hkl_path = f'{generated_path}/hkl_{hkl_max}.npy'

    if os.path.exists(hkl_path):
        print('Loading in hkl matrix...', end=" ")
        hkl_info = np.load(hkl_path)
    else:
        print('Computing hkl matrix...', end=" ")
        hkl_info = hkl()
    print('- Done ✓')

    root = in_dir.strip('/').split('/')[-1]

    if os.path.exists(out_dir) and \
            len([name for name in os.listdir(out_dir) if os.path.isfile(os.path.join(out_dir, name))]) > 0:
        return

    if not os.path.exists(f'{generated_path}/Preprocessed_{root}'):
        os.makedirs(f'{generated_path}/Preprocessed_{root}')

    hkl_info = torch.as_tensor(hkl_info).share_memory_()  # Faster hkl_info data transfer to parallel workers

    for path, _, files in os.walk(in_dir):
        with mp.Pool(os.cpu_count()) as pool:
            list(tqdm(pool.imap_unordered(star,
                                          [(path, file, f'{generated_path}/Preprocessed_{root}', hkl_info)
                                           for file in files if file.endswith('.cif')
                                           and not os.path.exists(f'{generated_path}/Preprocessed_{root}/{file}')]),
                      desc=f'Generating synthetic XRDs from crystal data in {path}. '
                           f'This can take a moment.', total=len(files)))

    # Non-parallel version
    # for path, _, files in os.walk(in_dir):
    #     for file in tqdm(files, desc=f'Generating synthetic XRDs from crystal data in {path}. This can take a moment.'):
    #         if file.endswith('.cif'):
    #             # Save preprocessed data
    #             if not os.path.exists(f'{generated_path}/Preprocessed_{root}/{file}'):
    #                 process_cif(path, file, f'{generated_path}/Preprocessed_{root}', hkl_info)

    preprocessed_files = glob.glob(f'{generated_path}/Preprocessed_{root}/*.txt')

    # TODO Delete below and retrieve via index in Dataset - Downside: current Bluehive generated data becomes obsolete
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Format processed data for training
    for name in tqdm(sorted(preprocessed_files), desc='Converting generated XRD data to training format...'):
        with open(name) as f:
            xrd = f.readlines()

        with open(out_dir + 'labels7.csv', "a") as labels7:
            data = np.zeros((1, 7), dtype=int)
            # One-hot
            data[0, int(xrd[1].split()[1]) - 1] = 1
            # Save data
            np.savetxt(labels7, data, fmt="%d", delimiter=",")
        with open(out_dir + 'labels230.csv', "a") as labels230:
            data = np.zeros((1, 230), dtype=int)
            # One-hot
            data[0, int(xrd[2].split()[1]) - 1] = 1
            # Save data
            np.savetxt(labels230, data, fmt="%d", delimiter=",")
        with open(out_dir + 'features.csv', "a") as features:
            data = np.zeros((1, 8500), dtype=int)
            for i in range(0, 8500):
                data[0, i] = float(xrd[i+3+500].split()[1]) * 1000
            # Save data
            np.savetxt(features, data, fmt="%d", delimiter=",")


if __name__ == '__main__':
    generate()
