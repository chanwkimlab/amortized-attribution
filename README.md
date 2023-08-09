# XAI Amortization

## Installation

```bash
conda create -n xai-amortization python=3.9
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install transformers==4.30.2 evaluate==0.4.0 datasets==2.13.1 accelerate==0.20.3 scikit-learn==1.3.0

pip install git+https://github.com/huggingface/transformers
pip install datasets==2.14.3
pip install jupyterlab
```

```bash
conda create -n xai-amortization2 python=3.9
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install transformers==4.32.0 evaluate==0.4.0 datasets==2.14.3 accelerate==0.20.3 scikit-learn==1.3.0
pip install git+https://github.com/huggingface/transformers
pip install jupyterlab wandb ipdb
ipython kernel install --name "xai-amortization2" --user
```


# <https://stackoverflow.com/questions/68806265/huggingface-trainer-logging-train-data>


* deterministic sampling 
  * override `get_train_sampler` function
  * (done) not necessary if use `predict` function
* save model output (callback, metrics, loss function)
  * callback (x) does not have output
  * metrics (x) only run during evalute loop
  * loss function (x) dealing with randomness / predict calls this
  * (done) use `predict` function
* generate masks (change `transform`` function) also deterministic for training
  * (done) by adding transform function.
* custom loss function (define new `Trainer` class)
  * (done) defined new model class


* model output
  * use hidden states
* evaluate multiple times
  * first evaluate. 


## Note

This repository was inspired by the following repositories:
    <https://github.com/huggingface/transformers/tree/149cb0cce2df3f932de58c6d05cec548600553e2/examples/pytorch/image-classification>
