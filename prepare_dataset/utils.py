import os

import requests
import tqdm
import zstandard as zstd


def download_single_file(url, save_dir, file_name):
    print('INFO: Entered download_single_file.')
    print(url)
    print(save_dir)
    print(file_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, file_name)
    if os.path.exists(save_path):
        return save_path

    with requests.get(url, stream=True) as read_file, open(save_path, 'wb') as write_file:
        total_length = int(read_file.headers.get("content-length"))
        with tqdm.tqdm(
            total=total_length,
            unit="B",
            unit_scale=True,
            desc=file_name,
        ) as pbar:
            for chunk in read_file.iter_content(chunk_size=8192):
                if chunk:
                    write_file.write(chunk)
                    pbar.update(len(chunk))
    return save_path

def extract_single_zst_file(input_path, save_dir, file_name):
    total_length = os.stat(input_path).st_size
    with tqdm.tqdm(
        total=total_length,
        unit="B",
        unit_scale=True,
        desc=file_name,
    ) as pbar:
        dctx = zstd.ZstdDecompressor()
        read_size = 131075
        write_size = int(read_size * 4)
        save_path = os.path.join(save_dir, file_name)
        with open(input_path, 'rb') as in_f, open(save_path, 'wb') as out_f:
            for chunk in dctx.read_to_iter(in_f, read_size=read_size, write_size=write_size):
                out_f.write(chunk)
                pbar.update(read_size)
