import os

import requests
from tqdm import tqdm


def download(path, name):
    # Create new CIFs directory if it doesn't exist
    if not os.path.exists(path + name):
        os.makedirs(path + name)

    with open(path + 'DOWNLOAD.txt', "r") as f:
        # Iterate through download destinations
        for url in tqdm(f.readlines(), desc='Downloading open-access CIFs'):
            # Download CIF file data
            cif = requests.get(url).content
            # Check if non-empty
            if cif:
                open(path + name + ''.join(url.split('/')[-1].split('.cif')[:-1]) + '.cif', 'wb').write(cif)
