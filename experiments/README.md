## Introduction

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
cd ts_classification/
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

```bash
cd ts_regression/
data_name=('HouseholdPowerConsumption2' 'BeijingPM25Quality' 'BeijingPM10Quality' 'LiveFuelMoistureContent' 'FloodModeling1'
             'FloodModeling2' 'FloodModeling3' 'AustraliaRainfall' 'PPGDalia' 'IEEEPPG' 'BIDMC32RR' 'BIDMC32HR' 'BIDMC32SpO2'
             'NewsHeadlineSentiment' 'NewsTitleSentiment' 'Covid3Month')
lora_r=16
ABBA_tol=0.005
ABBA_init='agg'
ABBA_alpha=0.01

time2=$(date "+%Y-%m-%d %H:%M:%S")
echo $time2
for i in {1..7}
do
  echo ${data_name[$i]}
  python main-ts-regression.py \
  --model_name "llama2-7B" \
  --data_name ${data_name[$i]} \
  --batch_size=2 \
  --lr=2e-4 \
  --lora_r=${lora_r} \
  --weight_decay=0.00001 \
  --epochs=20 \
  --ABBA_tol=${ABBA_tol} \
  --ABBA_init=${ABBA_init} \
  --ABBA_alpha=${ABBA_alpha} \
  --ABBA_scl=1 \
  --MAX_LENGTH 4096
done
echo $time2
```

To reproduce the [DrCIF](https://arxiv.org/abs/2305.01429) and
[FreshPRINCE](https://arxiv.org/abs/2201.12048) on [Time Series Extrinsic Regression (TSER)](https://arxiv.org/abs/2006.12672) benchmark, we used the [tsml-eval](https://tsml-eval.readthedocs.io/en/v0.4.0/) that is a package containing tools for running experiments with time series machine learning algorithms and evaluating the results. 

```bash
ml Python/3.10.8-GCCcore-12.2.0
pip install tsml-eval==0.3.0

data_name=('AppliancesEnergy' 'HouseholdPowerConsumption1_nmv' 'HouseholdPowerConsumption2_nmv' 'BenzeneConcentration_nmv'
             'BeijingPM25Quality_nmv' 'BeijingPM10Quality_nmv' 'LiveFuelMoistureContent' 'FloodModeling1'
             'FloodModeling2' 'FloodModeling3' 'AustraliaRainfall' 'PPGDalia_eq' 'IEEEPPG' 'BIDMC32RR' 'BIDMC32HR' 'BIDMC32SpO2'
             'NewsHeadlineSentiment' 'NewsTitleSentiment' 'Covid3Month')
             
model_name=('MultiROCKET' 'InceptionE' 'TSF' 'Ridge' 'RotF' 'DrCIF' 'FreshPRINCE' 'CNN')

time3=$(date "+%Y-%m-%d %H:%M:%S")
echo $time3
for i in {0..18}
do
  python tsml-eval/tsml_eval/publications/2023/tser_archive_expansion/run_experiments.py ${data_name[$i]} results/ ${model_name} ${data_name[$i]} 0
done
time3=$(date "+%Y-%m-%d %H:%M:%S")
echo $time3
```


### Forecasting

Please pull the request if you have some problems (such as ABBA cannot recognize the generated symbols). 
```bash
cd ts_forecasting/
data_name='ETTh1'
lora_r=16
ABBA_tol=0.000040837
ABBA_init='agg'
ABBA_alpha=0.000040837
seq_len=168
pred_len=96

python main-ts-prediction-168.py \
--model_name "llama2-7B" \
--data_name ${data_name} \
--data ${data_name} \
--batch_size=4 \
--lr=2e-4 \
--lora_r=${lora_r} \
--weight_decay=0.00001 \
--epochs=20 \
--ABBA_tol=${ABBA_tol} \
--ABBA_init=${ABBA_init} \
--ABBA_alpha=${}ABBA_alpha \
--ABBA_scl=1 \
--seq_len=${seq_len} \
--pred_len=${pred_len}
```


## Practical applications

Please wait for the update. Thanks.






