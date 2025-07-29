# STARE (streamlined) with [RVT](https://github.com/uzh-rpg/RVT) as a test case

## Introduction

The difference between this branch and the `main` branch is that on the `main` branch, users can only run STARE with a simple, author-written VOT test case. In contrast, this branch allows users to run STARE with a proper object detection model. The object detection model chosen as the test case here is RVT.

Note:

1.  To run this RVT test case, users must first configure the RVT environment on their own.
2.  The integration of RVT in this repository may not be rigorous. That is, the configuration of the RVT algorithm within this repository (e.g., parameters for event representation, pre- and post-processing for model inputs/outputs) might not achieve RVT's peak performance. We have this concern because we have not conducted rigorous testing on the RVT version integrated into STARE (streamlined). However, this does not affect the validity of STARE itself, as we use RVT here as a test case simply to demonstrate that an object detection task can run properly within STARE. Users are free to modify the code, whether for the integrated RVT, other object detection algorithms, or other perception tasks.
3.  The integration process for RVT follows the "How to Use" tutorial in the `main` branch.

<br>

## Installation

1.  Clone [RVT](https://github.com/uzh-rpg/RVT) locally and set up its environment.
2.  Install STARE within the RVT environment.

```bash
cd /path/to/STARE_streamlined
pip install -e .
```

<br>

## How to Use

1.  Download the [test case dataset](https://drive.google.com/file/d/1gNLtaR8GM3FenOsCynGFOyPVheZL7oTM/view?usp=sharing) (only one sequence, for demonstration purposes) and unzip it.
2.  Change the `rvt_project_root` in `/path/to/STARE_streamlined/src/stare/configs/demo_rvt.yaml` to the path of the RVT project.
3.  Change the `model.checkpoint` in `/path/to/STARE_streamlined/src/stare/configs/demo_rvt.yaml` to the path of the RVT checkpoint (e.g., rvt-t.ckpt).
4.  Change the `dataset.path` in `/path/to/STARE_streamlined/src/stare/configs/demo_rvt.yaml` to the path of the test case dataset.
5.  Run `cd /path/to/STARE_streamlined` and then `python -m main`. The results will be saved in `/path/to/STARE_streamlined/../test_case_rvt_results`.
