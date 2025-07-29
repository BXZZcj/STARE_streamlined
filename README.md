# STARE (streamlined) with [DEVO](https://github.com/tum-vision/DEVO) as a test case

## Introduction

The difference between this branch and the `main` branch is that on the `main` branch, users can only run STARE with a simple, author-written VOT test case. In contrast, this branch allows users to run STARE with a proper visual odometry model. The visual odometry model chosen as the test case here is DEVO.

Note:

1.  To run this DEVO test case, users must first configure the DEVO environment on their own.
2.  The integration of DEVO in this repository may not be rigorous. That is, the configuration of the DEVO algorithm within this repository (e.g., parameters for event representation, pre- and post-processing for model inputs/outputs) might not achieve DEVO's peak performance. We have this concern because we have not conducted rigorous testing on the DEVO version integrated into STARE (streamlined). However, this does not affect the validity of STARE itself, as we use DEVO here as a test case simply to demonstrate that an visual odometry task can run properly within STARE. Users are free to modify the code, whether for the integrated DEVO, other visual odometry algorithms, or other perception tasks.
3.  The integration process for DEVO follows the "How to Use" tutorial in the `main` branch.

<br>

## Installation

1.  Clone [DEVO](https://github.com/tum-vision/DEVO) locally and set up its environment.
2.  Install STARE within the DEVO environment.

```bash
cd /path/to/STARE_streamlined
pip install -e .
```

<br>

## How to Use

1.  Download the [test case dataset](https://drive.google.com/file/d/1EVgDqUcROx1Qz3rr60j8l86lObBGJbPn/view?usp=sharing) (only one sequence, for demonstration purposes) and unzip it.
2.  Change the `devo_project_root` in `/path/to/STARE_streamlined/src/stare/configs/demo_devo.yaml` to the path of the DEVO project.
3.  Change the `model.checkpoint` in `/path/to/STARE_streamlined/src/stare/configs/demo_devo.yaml` to the path of the DEVO checkpoint (e.g., devo-t.ckpt).
4.  Change the `dataset.path` in `/path/to/STARE_streamlined/src/stare/configs/demo_devo.yaml` to the path of the test case dataset.
5.  Run `cd /path/to/STARE_streamlined` and then `python -m main`. The results will be saved in `/path/to/STARE_streamlined/../test_case_devo_results`.
