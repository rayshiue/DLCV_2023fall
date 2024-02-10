# Video-Question-Answering-on-STAR-Situated-Reasoning

## Create environment
```
conda create -n ThreeMissingOne python=3.8 -y
conda activate ThreeMissingOne
```

## Install Requirements and Download Dataset
```
bash setup.sh
```

## Finetune
```
bash finetune.sh
```
For TAs, how I reproduce the performance of the Flipped-VQA:
1. Calculate the correct base learning rate:\
the author apply 0.09 as blr with 4 GPUs; therefore, we have to set 0.09*4=0.36 as blr when using single GPU.
2. Remove distributed training:\
I've no idea why the performance increase when I remove distributed training with same hyperparameters.
3. Set constant learning rate with warm up:\
I reproduce the performance with constant learning rate.


## Inference
```
bash inference.sh
```

## Ensemble
The ensemble result of my highest score is in ./results/score=64.25.json
```
python3 utils_voting.py
```