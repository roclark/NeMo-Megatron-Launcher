# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import csv
import shutil

import hydra
from glob import glob
from time import sleep


def split_files(files_list: list, shards_per_file: int) -> list:
    """Function to split the list of files into chunks of `shards_per_file` size.

    Arguments:
        files_list: List of all files to be concatenated.
        shards_per_file: Number of individual files to concatenate into a single, larger file.
    """
    file_chunks = [files_list[i:i+shards_per_file]
                   for i in range(0, len(files_list), shards_per_file)]
    return file_chunks


@hydra.main(config_path="conf", config_name="config")
def main(cfg) -> None:
    """Function to concatenate the Slim Pajama dataset files.

    Arguments:
        cfg: main config file.
    """
    data_dir = cfg.get("data_dir")
    shards_per_file = cfg.get("shards_per_file")
    rm_extracted = cfg.get("rm_extracted")
    assert data_dir is not None, "data_dir must be a valid path."

    if cfg.get("cluster_type") == "k8s":
        wrank = int(os.environ.get("OMPI_COMM_WORLD_RANK", 0))
        wsize = int(os.environ.get("OMPI_COMM_WORLD_SIZE", 0))
    else:
        wrank = int(os.environ.get("RANK", 0))
        wsize = int(os.environ.get("WORLD_SIZE", 0))

    # Read the list of files on the first rank only and have all other ranks
    # read from that list to prevent race conditions where other ranks begin
    # processing before others and alter the training list.
    if wrank == 0:
        if os.path.exists(os.path.join(data_dir, "train_files.txt")):
            os.remove(os.path.join(data_dir, "train_files.txt"))
        files_list = sorted(glob(f"{data_dir}/example_train_*.jsonl"))

        with open(os.path.join(data_dir, "concat_files.txt"), "w") as concat_file:
            csv_data = csv.writer(concat_file, quoting=csv.QUOTE_ALL)
            csv_data.writerow(files_list)
    else:
        while not os.path.exists(os.path.join(data_dir, "concat_files.txt")):
            sleep(1)
        with open(os.path.join(data_dir, "concat_files.txt"), "r") as concat_file:
            files_list = csv.reader(concat_file)
            files_list = list(files_list)[0]

    files_to_concatenate = split_files(files_list, shards_per_file)

    for num, chunk in enumerate(files_to_concatenate):
        # Divide the work between ranks
        if wrank != num % wsize:
            continue

        chunk_filename = os.path.join(data_dir, f"train_chunk_{num}.jsonl")
        with open(chunk_filename, "w") as chunk_file:
            for shard in chunk:
                with open(shard, "r") as shard_file:
                    shutil.copyfileobj(shard_file, chunk_file)
                if rm_extracted:
                    os.remove(shard)


if __name__ == "__main__":
    main()
