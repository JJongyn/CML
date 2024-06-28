# Cooperative Meta-Learning with Gradient Augmentation (CML)

This repository provides the code for the UAI 2024 (Main Track) paper titled Cooperative Meta-Learning with Gradient Augmentation (CML)

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
