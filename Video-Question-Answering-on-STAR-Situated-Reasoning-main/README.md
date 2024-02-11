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
For TAs, how to reproduce our results:
1. Calculate the correct base learning rate:\
The author set blr as 0.09 with 4 GPUs; therefore, we set blr as 0.09*4=0.36 for a single GPU.
2. Remove distributed training:\
The performance is improved after disabling distributed training.
3. Set constant learning rate with warm up:\
To reproduce our results, please use a constant learning rate.


## Inference
```
bash inference.sh
```

## Ensemble
Our predictions of highest score: ./results/score=64.25.json
```
python3 utils_voting.py
```
