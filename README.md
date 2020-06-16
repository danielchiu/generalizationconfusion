# Generalization by Noticing Confusion
This is the official implementation of the paper [Generalization by Noticing Confusion](https://arxiv.org/abs/2006.07737

## Requirements

- Python >= 3.6
- PyTorch >= 1.0
- Numpy
- Google Cloud TPU with XLA (not strictly necessary; CUDA is also usable)

## Usage
### TPU Setup
Follow the instructions in the README here: https://github.com/pytorch/xla.
### Locations of Files
`main.py` contains the training and the evaluation code. Here is a description of some of the options:
- Options
  - `noise-rate`: the percentage of data with randomly changed labels.
  - `noise-type`: type of random corruptions (i.e., corrupted_label, Gaussian, random_pixel, shuffled_pixel)
  - `sat-es`: the number of epochs before label corrections begins
  - `sat-alpha`: the momentum term ![formula](https://render.githubusercontent.com/render/math?math=\alpha) of our approach (either self-adaptive training or self-adaptive mixup)
  - `mixup-alpha`: specifies the distribution that the mixup coefficients are drawn from, specificially a symmetric Beta distribution with parameter ![formula](https://render.githubusercontent.com/render/math?math=\alpha)
  - `mixup-gamma`: specifies the cutoff for the mixup mixing coefficient beyond which label correction occurs.

#### Results on CIFAR datasets under uniform label noise
- CIFAR10

|Noise Rate         |0.2    |0.4    |0.6    |0.8    |
|-------------------|-------|-------|-------|-------|
|Test Accuracy(%)   |95.48  | 94.15 |91.21  |80.25  |

- CIFAR100

|Noise Rate         |0.2    |0.4    |0.6    |0.8    |
|-------------------|-------|-------|-------|-------|
|Test Accuracy(%)   |78.03  |72.67  |65.12  |38.96  |


### Imbalanced Class Training
With imbalanced class training, we show that self-adaptive training learns imbalanced classes far worse than standard cross-entropy training does. 

#### Exact Commands
Here are the exact commands to run for the first experiment and the second one. 

##### Uniform Label Noise

  ```bash
  $ git checkout master
  $ bash scripts/cifar10/run_mixup.sh [TRIAL_NAME] [NOISE_RATE] [MIXUP_ALPHA]
  ```
  
Here, TRIAL_NAME is used for experiment naming, NOISE_RATE is the proportion of labels which are flipped (e.g. 0.4), and MIXUP_ALPHA is the parameter for the symmetric beta distribution from which the mixing coefficients are selected. 
  
##### Imbalanced Classes

```bash
$ git checkout imbalatest
$ bash scripts/cifar10/run_sat.sh [TRIAL_NAME] [CLASS_RATIO]
```
Here, TRIAL_NAME is used for experiment naming and CLASS_RATIO is the ratio of number of examples of the classes for the training. Everything else is default. 

## Reference
For technical details, please refer to the paper.
```
@article{chiu2020generalization,
        title = {Generalization by Recognizing Confusion},
        author = {Daniel Chiu and Franklyn Wang and Scott Duke Kominers},
        journal = {arXiv preprint arXiv:2006.07737},
        year = {2020}
}
```
