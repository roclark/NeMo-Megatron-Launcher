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

import multiprocessing
import os

import hydra
import csv
import nemo_launcher.utils.file_utils as utils
from glob import glob
from time import sleep


@hydra.main(config_path="conf", config_name="config")
def main(cfg) -> None:
    """Function to extract the Slim Pajama dataset files.

    Arguments:
        cfg: main config file.
    """
    data_dir = cfg.get("data_dir")
    rm_downloaded = cfg.get("rm_downloaded")
    assert data_dir is not None, "data_dir must be a valid path."

    if cfg.get("cluster_type") == "k8s":
        wrank = int(os.environ.get("OMPI_COMM_WORLD_RANK", 0))
        wsize = int(os.environ.get("OMPI_COMM_WORLD_SIZE", 0))
    else:
        wrank = int(os.environ.get("RANK", 0))
        wsize = int(os.environ.get("WORLD_SIZE", 0))

    proc_list = []

    # Read the list of files on the first rank only and have all other ranks
    # read from that list to prevent race conditions where other ranks begin
    # processing before others and alter the training list.
    if wrank == 0:
        files_list = sorted(glob(f"{data_dir}/example_train_*.jsonl.zst"))

        with open(os.path.join(data_dir, "train_files.txt"), "w") as train_file:
            csv_data = csv.writer(train_file, quoting=csv.QUOTE_ALL)
            csv_data.writerow(files_list)
    else:
        while not os.path.exists(os.path.join(data_dir, "train_files.txt")):
            sleep(1)
        with open(os.path.join(data_dir, "train_files.txt"), "r") as train_file:
            files_list = csv.reader(train_file)
            files_list = list(files_list)[0]

    files_list_groups = utils.split_list(files_list, wsize)
    files_to_extract = files_list_groups[wrank]

    for filename in files_to_extract:
        filepath = os.path.join(data_dir, filename)

        if not os.path.exists(filepath):
            continue

        output_file = filepath.replace(".zst", "")
        proc = multiprocessing.Process(
            target=utils.extract_single_zst_file,
            args=(filepath, data_dir, output_file, rm_downloaded),
        )
        proc_list.append(proc)
        proc.start()

    for proc in proc_list:
        proc.join()

    #if cfg.get("cluster_type") == "bcm":
    #    file_number = int(os.environ.get("SLURM_ARRAY_TASK_ID"))
    #    downloaded_path = os.path.join(data_dir, f"{file_number:02d}.jsonl.zst")
    #    output_file = f"{file_number:02d}.jsonl"
    #    utils.extract_single_zst_file(
    #        downloaded_path, data_dir, output_file, rm_downloaded
    #    )
    #elif cfg.get("cluster_type") in ["bcp", "k8s"]:
    #    file_numbers = cfg.get("file_numbers")
    #    # Downloading the files
    #    files_list = utils.convert_file_numbers(file_numbers)
    #    # Assumes launched via mpirun:
    #    #   mpirun -N <nnodes> -npernode 1 ...
    #    wrank = int(os.environ.get("OMPI_COMM_WORLD_RANK", 0))
    #    wsize = int(os.environ.get("OMPI_COMM_WORLD_SIZE", 0))
    #    files_list_groups = utils.split_list(files_list, wsize)
    #    files_to_extract = files_list_groups[wrank]
    #    proc_list = []
    #    for file_number in files_to_extract:
    #        downloaded_path = os.path.join(data_dir, f"{file_number:02d}.jsonl.zst")
    #        output_file = f"{file_number:02d}.jsonl"
    #        # TODO: Consider multiprocessing.Pool instead.
    #        proc = multiprocessing.Process(
    #            target=utils.extract_single_zst_file,
    #            args=(downloaded_path, data_dir, output_file, rm_downloaded),
    #        )
    #        proc_list.append(proc)
    #        proc.start()

    #    for proc in proc_list:
    #        proc.join()


if __name__ == "__main__":
    main()
