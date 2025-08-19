"""
Executes a bunch of python scripts one after the other.

Quick Use - define [METHOD_NAME] and [DATASET_NAME] in the condiguration files:
python batch_computation.py \
    --method [METHOD_NAME] \
    --dataset [DATASET_NAME]

by Stéphane Vujasinovic
"""
# - IMPORTS ---
import subprocess
import argparse
import colorful as cf


# - FUNCTIONS ---
def arguments_parser():
    """
    Return parser arguments
    """
    parser = argparse.ArgumentParser(
        description="Run a batch of python script at once")
    parser.add_argument(
        "--method", type=str, default='XMem', help='To define the method')
    parser.add_argument(
        "--dataset", type=str, default='d17-val', help='To define the dataset')
    parser.add_argument(
        "--ext", type=str, default='NPZ', help='which extension to use')
    parser.add_argument(
        "--mask_H", type=int, default=5,
        help="""Mask the total entropy of the frame with a mask
        around the prediction of an object. Based on the
        size of the object to delimit the region used in
        the entropy calculation. Expressed as a percentage."""
    )

    # Load/Set arguments
    return parser.parse_args()


# - CONSTANTS ---
# This is an ordered list, be carefull with it.
PYTHON_BATCH = [#"compute_count_nbr_of_objx.py",
                #"compute_confusion.py",
                #"compute_entropy_for_ensemble.py",
                "compute_entropy.py",
                "analyze_iou_vs_entropy.py",
                "compute_summary_statistics.py",
                #"compute_IoU_ratio.py"
                ]
NEED_H_MASK = ["analyze_iou_vs_entropy.py",
               "compute_summary_statistics.py"]

BENCHMARK_SCRIPTS = {"d17-val": "benchmarks/eval_davis_predictions.py",
                     "lvos-val": "benchmarks/eval_lvos_predictions.py"}

# - MAIN ---
if __name__ == "__main__":
    args = arguments_parser()
    dataset_name = args.dataset
    method_name = args.method
    ext = args.ext
    mask_H_value = args.mask_H

    # - BENCHMARK EVALUATION ---
    # Run the benchmark script
    benchmark_scipt = BENCHMARK_SCRIPTS[dataset_name]
    command = [
        'python', benchmark_scipt,
        '--method', method_name,
        '--dataset', dataset_name]
    colored_cmd = (
        "python "
        f"{cf.violet(f'{benchmark_scipt}')} "
        f"--dataset {cf.violet(dataset_name)} "
        f"--method {cf.violet(method_name)}"
        )

    # Run script
    print('\n-------------------------------------------------------------'
          '\nRunning -', colored_cmd,
          '\n-------------------------------------------------------------'
          )
    subprocess.run(command)

    # - CUSTOM EVALUATION for iVOTS---
    # Run the results computation scripts
    for python_script in PYTHON_BATCH:
        # Prepare the command
        command = ['python', python_script,
                   '--dataset', dataset_name,
                   '--method', method_name,
                   '--ext', ext]
        if python_script in NEED_H_MASK:
            command += ['--mask_H', str(mask_H_value)]

        # Aesthetics
        colored_cmd = (
            "python "
            f"{cf.turquoise(python_script)} "
            f"--dataset {cf.turquoise(dataset_name)} "
            f"--method {cf.turquoise(method_name)} "
            f"--ext {ext}"
            )
        if python_script in NEED_H_MASK:
            colored_cmd += f" {cf.orange('--mask_H')} {cf.orange(mask_H_value)}"

        # Run script
        print('\n-------------------------------------------------------------'
              '\nRunning -', colored_cmd,
              '\n-------------------------------------------------------------'
              )
        subprocess.run(command)
