# STARE (streamlined)

## Introduction

STARE, an acronym for **ST**ream-based **lA**tency-awa**R**e **E**valuation, is a performance evaluation framework for event-based perception models, proposed in the paper [Bridging the Latency Gap with a Continuous Stream Evaluation Framework in Event-Driven Perception](https://www.researchsquare.com/article/rs-6135923/v1). STARE was designed to simulate the real-time online performance of event-based perception models in real-world deployment scenarios. The core of its implementation revolves around two key points:

1.  Scheduling the model to **immediately sample events for the next inference loop** once it completes the last one.
2.  Incorporating the model's **inference latency** into the evaluation of its perception accuracy.

This repository provides a streamlined implementation of STARE, hence the name STARE (streamlined). Its purpose is to make it easy for users to integrate STARE into various perception tasks.
The original paper demonstrated the effectiveness of STARE using the single-object tracking task and drew further conclusions. The code to reproduce the data from the original paper is available in this repository: [STARE](https://github.com/IIVEventGroup/STARE). That repository also implements STARE, but it is tightly coupled with the [pytracking](https://github.com/visionml/pytracking) framework, which is commonly used for object tracking, making it inconvenient for users to perform evaluations on other tasks. This is why we developed STARE (streamlined).

<br>

## Installation

STARE (streamlined) provides only the core, simplified implementation of the STARE framework. It does not require special Python libraries; it only needs common ones like `torch` and `numpy`.
Since users will ultimately use STARE for specific tasks such as object tracking, object detection, or SLAM, we recommend that users first set up the environment for their specific task and then install STARE within that environment. The dependencies required by STARE itself will not conflict with the special dependencies needed for other environments.

We recommend the following steps for setting up the environment:

1.  Set up the environment for the required specific task. For example, if a user wants to run an object detection task with [RVT](https://github.com/uzh-rpg/RVT), they should first set up the RVT environment.
2.  Activate the environment from the step above.
3.  Install STARE into that environment:

```bash
cd /path/to/STARE_streamlined
pip install -e .
```

To help users quickly adapt to STARE (streamlined), we have written a simple object tracking task as a test case and integrated it into STARE (see `How to Use` below for details). For this test case, users do not need to install complex dependencies for a specific task. They just need to create a new conda environment and run step 3 above to install STARE.

<br>

## How to Use

Users need to manually make the following changes in `/path/to/STARE_streamlined/src` to adapt STARE to a specific task:

1.  **For parsing datasets**: Users should add a new module in the `data` folder to read and parse the specific dataset. This new module should implement a class that inherits from the `BaseDataset` base class in `/path/to/data/base_dataset.py`. Then, import this class in `data/__init__.py`.
2.  **For running models**: Users should add a new module in the `models` folder to initialize and run the specific task's algorithm. This new module should implement a class that inherits from the `BasePerceptionModel` base class in `/path/to/models/base_model.py`. Then, import this class in `models/__init__.py`.
3.  **For event representation**: Users should add a new function in `/path/to/utils/convert_event_repr.py`. This function is responsible for converting a segment of raw events into the specific event representation required for the current task and returning it. The function must accept at least three parameters: `events: np.ndarray` (the raw events to be converted), `resolution: Tuple[int, int]` (the height & width of the raw events), and `device: torch.device` (the device the returned event representation should be on, which will be passed as "cuda" by default). Other required parameters can be defined by the user.
    Then, the user should add `"event_representation_name": function_name` to the `REPRESENTATION_MAP` variable in `/path/to/utils/convert_event_repr.py`.
4.  **For metrics required for latency-aware evaluation**: Users should add a new function in `/path/to/core/metrics.py`. This function is responsible for calculating the final metrics based on the model's output and the ground truth, returning it as a scalar. This function must accept at least one parameter: `matched_pairs: List[Tuple[torch.Tensor, torch.Tensor]]`, which is a list where each element is a tuple of `(model output, ground truth)` aligned by timestamp. Other required parameters can be defined by the user.
    Then, the user should add `"metrics_name": function_name` to the `METRIC_MAP` variable in `/path/to/core/metrics.py`.
    In fact, users can add multiple metric functions if they need to calculate various metrics. However, each metric function can only return a single scalar.
5.  **For experiment configuration**: Add a new YAML file in the `configs` folder to set experiment parameters. Subsequently, users only need to modify this YAML file to switch between different experimental setups, including but not limited to: model configurations, event representation parameters, metric types and parameters, etc.

To intuitively demonstrate the required changes, we have written a simple object tracking task test case to show how to integrate a new task into STARE. We have added detailed comments in the Python files and YAML configuration file involved in this test case.
Please note: this test case does not actually load the weights of an object tracking model and run it. We have manually written the computational logic that takes an event representation as input and outputs a target object bounding box, just to get STARE running and to make it easier for users to read the code and debug.

The steps to run this test case are as follows:

1.  Download the [test case dataset](https://drive.google.com/file/d/1oY7BHheLXazzakZjoQaoG3nHHZBac1mk/view?usp=sharing) (only one sequence, for demonstration purposes) and unzip it.
2.  Change the `dataset.path` in `/path/to/STARE_streamlined/src/stare/configs/demo_vot.yaml` to the path of the test case dataset.
3.  Run `cd /path/to/STARE_streamlined` and then `python -m main`. The results will be saved in `/path/to/STARE_streamlined/../test_case_results`.

To demonstrate how to integrate tasks other than object tracking, we have also tested the event-based object detection model [RVT](https://github.com/uzh-rpg/RVT) and the event-based odometry model [DEVO](https://github.com/tum-vision/DEVO?tab=readme-ov-file). You can switch to the following branches in the current repository to view them:

1.  RVT: `git switch demo_rvt`
2.  DEVO: `git switch demo_devo`

<br><br>

## Support

**If you encounter any issues while using our code or dataset, please feel free to contact us.**

<br><br>

## License

- The released code is under [GPL-3.0 license](https://www.gnu.org/licenses/gpl-3.0.en.html) following the [DEVO](https://github.com/tum-vision/DEVO).
- The released test case dataset is under [CC-BY 4.0 license](https://creativecommons.org/licenses/by/4.0/).

<br><br>

## Acknowledgments

- Thanks for the great works including [RVT](https://github.com/uzh-rpg/RVT) and [DEVO](https://github.com/tum-vision/DEVO).
