## Introduction

Authors: @ChengKang520 and @the-null

Aimed at making LLMs understand time series (e.g. time series classification tasks, time series regression tasks, time series forecasting tasks, time series trend analysis, action planning tasks and so on) by fine-tuning a single GPU.

A fine-tuned LLMABBA model encapsulates two core components: a symbolic approximation encoder and a language generator. During a symbols approximation process, we encode the time series input with the LLMs' tokens and pass it to the Fixed-point Adaptive Piecewise Linear Continuous Approximation (FAPCA) to extract relevant chain of logics (or patterns). Such contextualized inputs are then passed to the language generator.

Read more about LLMABBA at https://arxiv.org/abs/2411.18506.


## Experiment


```bash
$ python -m venv LLMABBA_Env
$ source LLMABBA_Env/bin/activate
$ pip install -e peft-0.10.0/
```

### Classification


```bash
lora_r=64
ABBA_tol=0.005
ABBA_init='agg'
ABBA_alpha=0.01
python main-ts-classification.py \
--model_name "llama2-7B" \
--data_name 'ptbdb' \
--batch_size=4 \
--lr=5e-4 \
--lora_r=${lora_r} \
--weight_decay=0.00001 \
--epochs=100 \
--ABBA_tol=${ABBA_tol} \
--ABBA_init=${ABBA_init} \
--ABBA_alpha=${ABBA_alpha} \
--ABBA_scl=1 \
--MAX_LENGTH 128
```

### Regression


### Forecasting


### Trend analysis



## Practical applications




