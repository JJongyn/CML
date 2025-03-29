# Cooperative Meta-Learning with Gradient Augmentation (CML)

This repository provides the code for the _UAI 2024_ (Main Track) paper titled **Cooperative Meta-Learning with Gradient Augmentation** (CML) by ***J. Shin***, S. Han, and J. Kim,.

## Overview
<img width="990" alt="cml" src="https://github.com/JJongyn/CML/assets/92678942/f084b6e5-10c0-47fa-90ad-a290ec398417">


## Requirements

- Python >= 3.9
- Pytorch == 1.12

```setup
pip install -r requirements.txt
```

## Training & Evaluation

To train and evalute the 4-conv model with CML in the paper, run this command:

```train
./run_cml.sh
```

If you want to train standard MAML, run this command:

```train
./run_maml.sh
```

## Train 

You can use **train_cml.py** to meta-train your model in CML framework. For example, to run Miniimagenet 5-way 5-shot, run this command:
```
train_cml.py --folder=~/data --dataset=miniimagenet --model=4-conv_cml --num-ways=5  --num-shots=5  --extractor-step-size=0.5  --classifier-step-size=0.5 --loss-scaling=1 --output-folder=./result --save-name=CML
```
* You can download the dataset from option **--download**
## Evaluation 

You can use **test_cml.py** to meta-test your model in CML framework. run this command:
```
test_cml.py --folder=~/data --dataset=miniimagenet --model=4-conv_cml --num-ways=5  --num-shots=5 --extractor-step-size=0.5 --classifier-step-size=0.5 --output-folder=./result --save-name=CML --use-colearner
```
* If you want to test co-learner, you can use the option **--use-colearner**.
* Note that it must be the same as the path to the model saved by train (ouput-folder, save-name)


### References

This code is based on the implementations of [BOIL](https://github.com/jhoon-oh/BOIL).

### Citation
If you use this project in your research, please cite our paper:

```bibtex
@article{shin2024cooperative,
  title={Cooperative Meta-Learning with Gradient Augmentation},
  author={Jongyun Shin and Seunjin Han and Jangho Kim},
  journal={arXiv preprint arXiv:2406.04639},
  year={2024}
}
