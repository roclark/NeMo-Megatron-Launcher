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
import re
import subprocess
import csv
from time import sleep

import hydra
import nemo_launcher.utils.file_utils as utils  # TODO: check if this in python path
import psutil
from glob import glob


@hydra.main(config_path="conf", config_name="config")
def main(cfg):
    launcher_scripts_path = cfg.get("launcher_scripts_path")
    data_config = cfg.get("data_config")
    data_dir = cfg.get("data_dir")
    rm_extracted = cfg.get("rm_extracted")
    tokenizer_type = cfg.get("tokenizer_type")
    tokenizer_library = cfg.get("tokenizer_library")
    tokenizer_model = cfg.get("tokenizer_model")
    assert data_dir is not None, "data_dir must be a valid path"

    # Vocab
    vocab_dir = cfg.get("vocab_save_dir")
    assert vocab_dir is not None, "vocab_save_dir must be a valid path."
    if "gpt" in tokenizer_type.lower():
        vocab_path = os.path.join(launcher_scripts_path, vocab_dir, "vocab.json")
    else:
        vocab_path = os.path.join(launcher_scripts_path, vocab_dir, "vocab.txt")

    # Merges
    merges_dir = cfg.get("merges_save_dir")
    assert merges_dir is not None, "merges_save_dir must be a valid path."
    merges_path = os.path.join(launcher_scripts_path, merges_dir, "merges.txt")

    # This compile doesn't seem to do anything. It compiles
    # "helpers.cpython-38-x86_64-linux-gnu.so", but since that file already
    # exists, it doesn't do anything. Force make via: touch helpers.cpp
    megatron_dir = "/opt/NeMo/nemo/collections/nlp/data/language_modeling/megatron"
    compiled_helpers_lib = os.path.join(megatron_dir, "compiled_helpers_lib")
    compilecmd = (
        f"cd /opt/NeMo; git rev-parse HEAD; "
        f"cd {megatron_dir}; "
        f"touch helpers.cpp; make;"
    )

    code_path = (
        "/opt/NeMo/scripts/nlp_language_modeling/preprocess_data_for_megatron.py"
    )
    runcmd = (
        f"cd {megatron_dir}; "
        f'export PYTHONPATH="/opt/NeMo/.:$PYTHONPATH"; '
        f'export TRANSFORMERS_CACHE="/temp_root/.cache/"; '
        f"python3 {code_path} "
    )

    #if cfg.get("cluster_type") == "bcm":
    #    file_number = int(os.environ.get("SLURM_ARRAY_TASK_ID"))
    #    extracted_path = os.path.join(data_dir, f"{file_number:02d}.jsonl")

    #    model_type = "t5"
    #    if "bert" in data_config:
    #        model_type = "bert"
    #    elif "gpt3" in data_config:
    #        model_type = "gpt3"
    #    elif "llama" in data_config:
    #        model_type = "llama"
    #    elif "falcon" in data_config:
    #        model_type = "falcon"

    #    output_prefix = os.path.join(data_dir, f"my-{model_type}_{file_number:02d}")

    #    flags = (
    #        f"--input {extracted_path} "
    #        f"--output-prefix {output_prefix} "
    #        f"--vocab {vocab_path} "
    #        f"--dataset-impl mmap "
    #        f"--tokenizer-library megatron "
    #        f"--tokenizer-type {tokenizer_type} "
    #        f"--tokenizer-library {tokenizer_library} "
    #        f"--tokenizer-model {tokenizer_model} "
    #        f"--workers $SLURM_CPUS_ON_NODE "
    #    )

    #    if model_type == "bert":
    #        # Used for bert binary head (Next sentence predition)
    #        flags += "--split-sentences "
    #    else:
    #        flags += f"--merge-file {merges_path} " f"--append-eod "

    #    os.system(compilecmd)
    #    runcmd += f"{flags} "
    #    os.system(runcmd)
    #    if rm_extracted:
    #        os.remove(extracted_path)
    if cfg.get("cluster_type") == "k8s":
        # Assumes launched via mpirun:
        #   mpirun -N <nnodes> -npernode 1 ...
        wrank = int(os.environ.get("OMPI_COMM_WORLD_RANK", 0))
        wsize = int(os.environ.get("OMPI_COMM_WORLD_SIZE", 0))
        lrank = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", 0))
    else:
        wrank = int(os.environ.get("RANK", 0))
        wsize = int(os.environ.get("WORLD_SIZE", 0))
        lrank = int(os.environ.get("LOCAL_RANK", 0))

    # Read the list of files on the first rank only and have all other ranks
    # read from that list to prevent race conditions where other ranks begin
    # processing before others and alter the training list.
    if wrank == 0:
        if os.path.exists(os.path.join(data_dir, "train_files.txt")):
            os.remove(os.path.join(data_dir, "train_files.txt"))
        if os.path.exists(os.path.join(data_dir, "concat_files.txt")):
            os.remove(os.path.join(data_dir, "concat_files.txt"))
        files_list = sorted(glob(f"{data_dir}/*.jsonl"))

        with open(os.path.join(data_dir, "preproc_files.txt"), "w") as preproc_file:
            csv_data = csv.writer(preproc_file, quoting=csv.QUOTE_ALL)
            csv_data.writerow(files_list)
    else:
        while not os.path.exists(os.path.join(data_dir, "preproc_files.txt")):
            sleep(1)
        with open(os.path.join(data_dir, "preproc_files.txt"), "r") as preproc_file:
            files_list = csv.reader(preproc_file)
            files_list = list(files_list)[0]

    if lrank == 0:
        # Compile once per node. Should be one container instance per node.
        os.system(compilecmd)
        os.system(f"touch {compiled_helpers_lib}")
    else:
        while not os.path.exists(compiled_helpers_lib):
            sleep(1)

    files_list_groups = utils.split_list(files_list, wsize)
    files_to_preproc = files_list_groups[wrank]

    if cfg.get("cluster_type") == "bcm":
        ncpus = "$SLURM_CPUS_ON_NODE"
    else:
        ncpus = psutil.cpu_count(logical=False)

    for filename in files_to_preproc:
        basename = os.path.basename(filename)
        file_num = re.findall('\d+', basename)

        # Invalid filename
        if len(file_num) != 1:
            continue
        file_num = int(file_num[0])

        model_type = "t5"
        if "bert" in data_config:
            model_type = "bert"
        elif "gpt3" in data_config:
            model_type = "gpt3"
        elif "llama" in data_config:
            model_type = "llama"
        elif "falcon" in data_config:
            model_type = "falcon"

        output_prefix = os.path.join(data_dir, f"my-{model_type}_{file_num:02d}")

        flags = (
            f"--input {filename} "
            f"--output-prefix {output_prefix} "
            f"--vocab {vocab_path} "
            f"--dataset-impl mmap "
            f"--tokenizer-library megatron "
            f"--tokenizer-type {tokenizer_type} "
            f"--tokenizer-library {tokenizer_library} "
            f"--tokenizer-model {tokenizer_model} "
            f"--workers {ncpus} "
        )

        if model_type == "bert":
            # Used for bert binary head (Next sentence predition)
            flags += "--split-sentences "
        else:
            flags += f"--merge-file {merges_path} " f"--append-eod "

        proc = subprocess.Popen(runcmd + flags, shell=True)
        proc.wait()
        if rm_extracted:
            os.remove(filename)


if __name__ == "__main__":
    main()
