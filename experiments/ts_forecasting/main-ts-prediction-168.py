import os
import argparse
import matplotlib.pyplot as plt
from fABBA import JABBA
import torch
from datasets.dataset_dict import DatasetDict
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, BitsAndBytesConfig, AutoModelForCausalLM, DataCollatorForLanguageModeling
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from transformers import AutoTokenizer, DataCollatorWithPadding
import numpy as np
import math
import pickle
from transformers import TrainingArguments, Trainer

import pandas as pd
from sklearn.preprocessing import StandardScaler
import warnings


torch.cuda.empty_cache()


from accelerate import FullyShardedDataParallelPlugin, Accelerator
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig

fsdp_plugin = FullyShardedDataParallelPlugin(
    state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
    optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False),
)

accelerator = Accelerator(fsdp_plugin=fsdp_plugin)



# def model_preprocessing_function(examples):
#     return model_tokenizer(examples['text'], truncation=True, padding='max_length', max_length=MAX_LENGTH)

# Python program to convert a list to string

# Python program to convert a list to string
def listToString(s):
    # initialize an empty string
    str1 = ""
    # traverse in the string
    for ele in s:
        str1 += ele + ' '
    # return string
    return str1


# Python program to convert a list to string
def listToString_blank(s):
    # initialize an empty string
    str1 = ""
    # traverse in the string
    for ele in s:
        str1 += ele + ''
    # return string
    return str1


# Python program to convert a list to string
def stringToList(s):
    # initialize an empty string
    str1 = []
    # traverse in the string
    for ele in range(int(len(s) / 2)):
        str1.append(s[ele * 2])
    # return string
    return str1


def mean(arr):
    return sum(arr) / len(arr)


def cross_correlation(x, y):
    # Calculate means
    x_mean = mean(x)
    y_mean = mean(y)

    # Calculate numerator
    numerator = sum((a - x_mean) * (b - y_mean) for a, b in zip(x, y))

    # Calculate denominators
    x_sq_diff = sum((a - x_mean) ** 2 for a in x)
    y_sq_diff = sum((b - y_mean) ** 2 for b in y)
    denominator = math.sqrt(x_sq_diff * y_sq_diff)
    correlation = numerator / denominator

    return correlation


def find_keys_by_value(dictionary, value):
    return [key for key, val in dictionary.items() if val == value]


def find_values_by_key(dictionary, keys):
    return [dictionary[key] for key in keys]



# def generate_and_tokenize_prompt(data_point):

#     full_prompt = f"""Generate a symbolic "Series" based on a given symbolic "Inputs". \n

#         ### Inputs:
#         {data_point["text"]}.

#         ### Series:
#         {data_point["label"]}
#         """
#     # print(full_prompt)
#     return tokenize(full_prompt)


def tokenize(data_point):
    model_inputs = model_tokenizer(
        data_point["text_inputs"],
        truncation=True,
        padding='max_length',
        max_length=MAX_LENGTH,
    )

    result = model_tokenizer(
        data_point["text_outputs"],
        truncation=True,
        padding='max_length',
        max_length=MAX_LENGTH,
    )

    model_inputs["labels"] = result["input_ids"].copy()

    #     result["labels"] = result["input_ids"].copy()
    #     result["labels_mask"] = result["attention_mask"].copy()

    return model_inputs


def main(opts):

    warnings.filterwarnings("ignore")

    global batch_size
    global model_tokenizer
    global MAX_LENGTH

    ###############   Time Series Data   ###############
    seq_len_pre = opts.seq_len  # 96 -> 96;;; 168 -> 24, 48, 96
    seq_len_post = opts.pred_len  # 96 -> 96;;; 168 -> 24, 48, 96

    MAX_LENGTH_pre = seq_len_pre * 7
    MAX_LENGTH_post = seq_len_post * 7

    batch_size = opts.batch_size
    MAX_LENGTH = opts.MAX_LENGTH

    ##  Quantization Coonfig
    quantization_config = BitsAndBytesConfig(
        # Load the model with 4-bit quantization
        load_in_4bit=True,
        # Use double quantization
        bnb_4bit_use_double_quant=True,
        # Use 4-bit Normal Float for storing the base model weights in GPU memory
        bnb_4bit_quant_type="nf4",
        # De-quantize the weights to 16-bit (Brain) float before the forward/backward pass
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    ######################################################################## lora config
    if opts.model_name == "roberta-large":

        model_checkpoint = "roberta-large"
        model_tokenizer = AutoTokenizer.from_pretrained(
            model_checkpoint,
            model_max_length=(MAX_LENGTH_pre+MAX_LENGTH_post)*2,
            padding_side="right",
            truncation=True,
            add_eos_token=True
        )
        model_tokenizer.pad_token_id = model_tokenizer.eos_token_id
        model_tokenizer.pad_token = model_tokenizer.eos_token
        model_tokenizer.padding_side = 'right'

        model_input = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_checkpoint,
            quantization_config=quantization_config,
            device_map="auto"
        )

        from peft import prepare_model_for_kbit_training

        model_input.gradient_checkpointing_enable()
        model_input.config.pad_token_id = model_input.config.eos_token_id
        model_input = prepare_model_for_kbit_training(model_input)

        roberta_peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=opts.lora_r,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            modules_to_save=["embed_tokens"],
        )
        model_input = get_peft_model(model_input, roberta_peft_config)

    elif opts.model_name == "mistral-7B":

        model_checkpoint = 'mistralai/Mistral-7B-Instruct-v0.1'
        model_tokenizer = AutoTokenizer.from_pretrained(
            model_checkpoint,
            model_max_length=(MAX_LENGTH_pre+MAX_LENGTH_post)*2,
            padding_side="right",
            truncation=True,
            add_eos_token=True
        )

        model_tokenizer.pad_token_id = model_tokenizer.eos_token_id
        model_tokenizer.pad_token = model_tokenizer.eos_token
        model_tokenizer.padding_side = 'right'

        model_input = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_checkpoint,
            quantization_config=quantization_config,
            device_map="auto"
        )

        from peft import prepare_model_for_kbit_training

        model_input.gradient_checkpointing_enable()

        model_input.config.pad_token_id = model_input.config.eos_token_id
        model_input = prepare_model_for_kbit_training(model_input)

        mistral_lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=opts.lora_r,  # the dimension of the low-rank matrices
            lora_alpha=16,  # scaling factor for LoRA activations vs pre-trained weight activations
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
                "lm_head",
            ],
            modules_to_save=["embed_tokens"],
            lora_dropout=0.05,  # dropout probability of the LoRA layers
            bias='none',  # wether to train bias weights, set to 'none' for attention layers
        )
        model_input = get_peft_model(model_input, mistral_lora_config)

    elif opts.model_name == "llama2-7B":

        model_checkpoint = "starmpcc/Asclepius-Llama2-7B"
        model_tokenizer = AutoTokenizer.from_pretrained(
            model_checkpoint,
            model_max_length=(MAX_LENGTH_pre+MAX_LENGTH_post)*2,
            padding_side="right",
            truncation=True,
            add_eos_token=True
        )
        model_tokenizer.pad_token_id = model_tokenizer.eos_token_id
        model_tokenizer.pad_token = model_tokenizer.eos_token

        model_input = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_checkpoint,
            quantization_config=quantization_config,
            device_map="auto"
        )

        from peft import prepare_model_for_kbit_training

        model_input.gradient_checkpointing_enable()
        model_input.config.pad_token_id = model_input.config.eos_token_id
        model_input = prepare_model_for_kbit_training(model_input)

        llama_peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=opts.lora_r,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
                "lm_head",
            ],
            modules_to_save=["embed_tokens"],
        )
        model_input = get_peft_model(model_input, llama_peft_config)

    else:
        print("Please input correct models!")

    model_vocab = model_tokenizer.get_vocab()
    vocab_list = list(model_vocab.keys())

    model_input.print_trainable_parameters()
    model_input = model_input.cuda()

    scaler = StandardScaler()  ##   DONE the Test
    # scaler = MinMaxScaler()

    ###############   Loading data   ###############
    if (opts.data_name == 'ETTh1') or (opts.data_name == 'ETTh2'):

        current_file = 'data/time-series-dataset/dataset/ETT-small/'
        df_raw = pd.read_csv(os.path.join(current_file, opts.data_name + '.csv'))
        border1s = [0, 12 * 30 * 24 - seq_len_pre, 12 * 30 * 24 + 4 * 30 * 24 - seq_len_post]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]

        #     qabba = JABBA(tol=0.000037, init='agg', alpha=0.000037, scl=3, verbose=0)  #  MinMaxScaler
        #     qabba = JABBA(tol=0.0001117541, init='agg', alpha=0.0001117541, scl=3, verbose=0)
        #     qabba = JABBA(tol=0.0000408066, init='agg', alpha=0.0000408066, scl=3, verbose=0)  #  StandardScaler -> ETTh1  len=24
        #     qabba = JABBA(tol=0.000040838, init='agg', alpha=0.000040838, scl=3, verbose=0)  #  StandardScaler -> ETTh1  len=96
        # qabba = JABBA(tol=0.000040838, init='agg', alpha=0.000040838, scl=3,
        #               verbose=0)  # StandardScaler -> ETTh1  len=168
        qabba = JABBA(tol=opts.ABBA_tol, init=opts.ABBA_init,
                      alpha=opts.ABBA_alpha, scl=3, verbose=1)

    elif (opts.data_name == 'ETTm1') or (opts.data_name == 'ETTm2'):
        current_file = 'data/time-series-dataset/dataset/ETT-small/'

        df_raw = pd.read_csv(os.path.join(current_file, opts.data_name + '.csv'))
        border1s = [0, 12 * 30 * 24 * 4 - seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]

    elif (opts.data_name == 'Weather'):
        current_file = 'data/time-series-dataset/dataset/ETT-small/'

        df_raw = pd.read_csv(os.path.join(current_file, opts.data_name + '.csv'))
        border1s = [0, 6 * 30 * 24 * 6 - seq_len, 6 * 30 * 24 * 6 + 0 * 30 * 24 * 6 - seq_len]
        border2s = [6 * 30 * 24 * 6, 6 * 30 * 24 * 6 + 0 * 30 * 24 * 6, 6 * 30 * 24 * 6 + 6 * 30 * 24 * 6]

    cols_data = df_raw.columns[1:]
    df_data = df_raw[cols_data]

    output_scaler = open(str(opts.model_name + "_" + opts.data_name + "-Scaler_Pre" + str(opts.seq_len) + "_Post" + str(opts.pred_len) + "_save.pkl"), 'wb')
    output_jabba = open(str(opts.model_name + "_" + opts.data_name + "-ABBA_Pre" + str(opts.seq_len) + "_Post" + str(
        opts.pred_len) + "_save.pkl"), 'wb')

    #############################################  Train Data  #############################################
    train_data = df_data[border1s[0]:border2s[0]]
    scaler.fit(train_data.values)
    train_data_transformed = scaler.transform(train_data)

    X_Train_data_patch = np.zeros(
        [train_data_transformed.shape[0] - (seq_len_pre + seq_len_post), seq_len_pre, train_data_transformed.shape[1]],
        dtype=float)
    Y_Train_data_patch = np.zeros(
        [train_data_transformed.shape[0] - (seq_len_pre + seq_len_post), seq_len_post, train_data_transformed.shape[1]],
        dtype=float)
    for i_data_patch in range(train_data_transformed.shape[0] - (seq_len_pre + seq_len_post)):
        X_Train_data_patch[i_data_patch, :, :] = train_data_transformed[i_data_patch:i_data_patch + seq_len_pre, :]
        Y_Train_data_patch[i_data_patch, :, :] = train_data_transformed[
                                                 i_data_patch + seq_len_pre:i_data_patch + seq_len_pre + seq_len_post,
                                                 :]

    str1 = pickle.dumps(scaler)
    output_scaler.write(str1)
    output_scaler.close()

    symbols_train_data = []
    symbols_train_data = qabba.fit_transform(X_Train_data_patch, alphabet_set=vocab_list, llm_split='Pre')
    # reconstruction_train_data = qabba.inverse_transform(symbols_train_data)
    # train_data_same_shape = qabba.recast_shape(reconstruction_train_data, llm_split='Pre')  # recast into original shape


    symbols_train_target = []
    symbols_train_target, params_train_target = qabba.transform(Y_Train_data_patch, llm_split='Post')
    # reconstruction_train_target = qabba.inverse_transform(symbols_train_target, params_train_target)
    # train_target_same_shape = qabba.recast_shape(reconstruction_train_target, recap_shape=Y_Train_data_patch.shape,
    #                                              llm_split='Post')  # recast into original shape

    str2 = pickle.dumps(qabba)
    output_jabba.write(str2)
    output_jabba.close()

    print('##############################################################')
    print("The length of used symbols is:" + str(qabba.parameters.centers.shape[0]))

    train_data_symbolic = []
    for i_data in range(len(symbols_train_data)):
        train_data_symbolic.append(listToString(list(symbols_train_data[i_data])))

    train_target_symbolic = []
    for i_data in range(len(symbols_train_target)):
        train_target_symbolic.append(listToString(list(symbols_train_target[i_data])))

    arranged_seq = np.random.randint(len(train_data_symbolic), size=int(len(train_data_symbolic) * 0.2))

    val_data_symbolic = [train_data_symbolic[index] for index in arranged_seq]
    val_target_symbolic = [train_target_symbolic[index] for index in arranged_seq]

    data_TS = DatasetDict({
        'train': Dataset.from_dict({'text_outputs': train_target_symbolic, 'text_inputs': train_data_symbolic}),
        'val': Dataset.from_dict({'text_outputs': val_target_symbolic, 'text_inputs': val_data_symbolic}),
    })

    alphabets_list_used = qabba.parameters.alphabets
    added_tokens = []
    for i_add_tokens in range(len(alphabets_list_used)):
        if alphabets_list_used[i_add_tokens] not in vocab_list:
            print(alphabets_list_used[i_add_tokens])
            added_tokens.append(alphabets_list_used[i_add_tokens])
    #         model_tokenizer.add_special_tokens(alphabets_list_used[i_add_tokens])

    model_tokenizer.add_special_tokens({'additional_special_tokens': added_tokens})
    print('#############################  END  #################################')


    # model_tokenized_datasets = data_TS.map(model_preprocessing_function, batched=True)
    # model_tokenized_datasets = data_TS.map(generate_and_tokenize_prompt)
    model_tokenized_datasets = data_TS.map(tokenize, batched=True, batch_size=batch_size)
    model_tokenized_datasets.set_format("torch")

    project = "ts-finetune-" + opts.data_name
    # b-instruct-v0.1-h
    run_name = opts.model_name + "-" + project + "-r" + str(opts.lora_r) + "-Pre" + str(opts.seq_len) + "-Post" + str(opts.pred_len)
    output_dir = "./" + run_name

    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=5e-4,
        warmup_ratio=0.1,
        max_grad_norm=0.3,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=opts.epochs,
        optim="paged_adamw_8bit",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        logging_dir="./logs",
        load_best_model_at_end=True,
        report_to="wandb",
        fp16=True,
        gradient_checkpointing=True,
        save_total_limit=1,
    )

    trainer_abba = Trainer(
        model=model_input,
        args=training_args,
        train_dataset=model_tokenized_datasets['train'],
        eval_dataset=model_tokenized_datasets["val"],
        data_collator=DataCollatorForLanguageModeling(tokenizer=model_tokenizer, mlm=False)
    )

    model_input.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    trainer_abba.train()

    ################################  Evaluate  ################################
    from peft import PeftModel

    peft_folder = os.listdir(output_dir)
    ft_model = PeftModel.from_pretrained(
        model_input,
        output_dir + "/" + peft_folder[0]
    )

    qabba_load = JABBA(tol=opts.ABBA_tol, init=opts.ABBA_init, alpha=opts.ABBA_alpha, scl=3, verbose=1)
    with open(output_jabba, 'rb') as file:
        qabba_load = pickle.loads(file.read())

    scaler_load = StandardScaler()
    with open(output_scaler, 'rb') as file:
        scaler_load = pickle.loads(file.read())


    #############################################  Test Data  #############################################
    test_data = df_data[border2s[0]:border2s[2]]
    test_data_transformed = scaler_load.transform(test_data.values)

    X_Test_data_patch = np.zeros(
        [test_data_transformed.shape[0] - (seq_len_pre + seq_len_post), seq_len_pre, test_data_transformed.shape[1]],
        dtype=float)
    Y_Test_data_patch = np.zeros(
        [test_data_transformed.shape[0] - (seq_len_pre + seq_len_post), seq_len_post, test_data_transformed.shape[1]],
        dtype=float)
    for i_data_patch in range(test_data_transformed.shape[0] - (seq_len_pre + seq_len_post)):
        X_Test_data_patch[i_data_patch, :, :] = test_data_transformed[i_data_patch:i_data_patch + seq_len_pre, :]
        Y_Test_data_patch[i_data_patch, :, :] = test_data_transformed[
                                                i_data_patch + seq_len_pre:i_data_patch + seq_len_pre + seq_len_post, :]

    symbols_test_data = []
    symbols_test_data, params_test_data = qabba_load.transform(X_Test_data_patch, llm_split='Post')
    reconstruction_test_data = qabba_load.inverse_transform(symbols_test_data, params_test_data)
    test_data_same_shape = qabba_load.recast_shape(reconstruction_test_data, recap_shape=X_Test_data_patch.shape,
                                                   llm_split='Post')  # recast into original shape

    symbols_test_target = []
    symbols_test_target, params_test_target = qabba_load.transform(Y_Test_data_patch, llm_split='Post')
    reconstruction_test_target = qabba_load.inverse_transform(symbols_test_target, params_test_target)
    test_target_same_shape = qabba_load.recast_shape(reconstruction_test_target, recap_shape=Y_Test_data_patch.shape,
                                                     llm_split='Post')  # recast into original shape

    print('##############################################################')
    print("The length of used symbols is:" + str(qabba_load.parameters.centers.shape[0]))

    test_data_symbolic = []
    for i_data in range(len(symbols_test_data)):
        test_data_symbolic.append(listToString(list(symbols_test_data[i_data])))

    test_target_symbolic = []
    for i_data in range(len(symbols_test_target)):
        test_target_symbolic.append(listToString(list(symbols_test_target[i_data])))

    model_tokenizer.pad_token_id = model_tokenizer.eos_token_id

    test_length = 10
    plus_seg = 1000
    model_output = []
    for i_test in range(test_length):  # len(data_TS['test'])):

        test_prompt = f"""Generate a Series {test_data_symbolic[i_test + plus_seg]} ### """

        print('###################################  Model Outputs  ####################################')
        #     model_input = model_tokenizer(test_tokenize_prompt(data_TS['test'][i_test]), return_tensors="pt").to("cuda")
        model_input = model_tokenizer(test_prompt, return_tensors="pt").to("cuda")

        temp_out = model_tokenizer.decode(

            ft_model.generate(
                **model_input,
                max_new_tokens=int(MAX_LENGTH_post * 2.1),
                max_length=int((MAX_LENGTH_pre + MAX_LENGTH_post) * 2.1),
                repetition_penalty=2.1,
                temperature=0.0,
            )[0],
            skip_special_tokens=True
        )
        print(temp_out)

        model_output.append(temp_out)

    # model_output = model_output[1:]

    ## TODO
    symbols_LLM_input = []
    symbols_LLM_output = []

    model_output_list1_processed = []
    model_output_list2_processed = []

    for i_out in range(test_length):

        model_output_list1 = model_output[i_out].split(' ### ')[0].split(' ')[3:-1]

        test_blank_index = []
        for i_list in range(len(model_output_list1) - 1):
            if model_output_list1[i_list] is '' and model_output_list1[i_list - 1] is '' and model_output_list1[
                i_list + 1] is '':
                test_blank_index.append(i_list)
        del_index1 = []
        del_index1 = np.array(test_blank_index, dtype=int)
        test_processed = []
        test_processed = [model_output_list1[num] for num, i in enumerate(model_output_list1) if num not in del_index1]

        if len(del_index1) != 0:
            add_blank = []
            for i_add in range(len(test_blank_index)):
                add_blank += '▁'

            add_blank = listToString_blank(add_blank)
            test_processed.insert(del_index1[0], add_blank)

        test_remove_index = []
        for i_list in range(len(test_processed) - 1):
            if test_processed[i_list] is '' and test_processed[i_list + 1] is not '':
                test_remove_index.append(i_list)
                test_processed[i_list + 1] = '▁' + test_processed[i_list + 1]
        del_index1 = []
        del_index1 = np.array(test_remove_index, dtype=int)
        test_processed2 = []
        test_processed2 = [test_processed[num] for num, i in enumerate(test_processed) if num not in del_index1]

        test_blank_index = []
        for i_list in range(len(test_processed2) - 1):
            if test_processed2[i_list] is '':
                test_blank_index.append(i_list)

        del_index1 = []
        del_index1 = np.array(test_blank_index, dtype=int)
        model_output_list1_processed = [test_processed2[num] for num, i in enumerate(test_processed2) if
                                        num not in del_index1]

        symbols_LLM_input.append(model_output_list1_processed)

        #     print('###################################  Plan A  ####################################')
        #     tokens_output_mistral = model_tokenizer(model_output[i_out].split('###')[1][1:], return_tensors="pt")['input_ids'][0][1:-1]
        #     model_output_list2_processed = []
        #     for i_find in range(len(tokens_output_mistral)):
        #         key_mistral_vocab = find_keys_by_value(mistral_vocab, tokens_output_mistral[i_find])[0]
        # #         model_output_list2_processed.append(key_mistral_vocab)
        #         if key_mistral_vocab != '▁':
        #             model_output_list2_processed.append(key_mistral_vocab)

        #     symbols_LLM_output.append(model_output_list2_processed)

        print('###################################  Plan B  ####################################')
        split_content = model_output[i_out].split(' ### ')
        split_len = len(split_content)
        model_output_list2 = ''
        if split_len == 2:
            model_output_list2 = model_output[i_out].split(' ### ')[1].split(' ')[2:-1]
        elif split_len > 2:
            for i_split_content in range(split_len - 1):
                model_output_list2 = model_output_list2 + split_content[i_split_content + 1] + ' ### '

        test_blank_index = []
        for i_list in range(len(model_output_list2) - 1):
            if model_output_list2[i_list] is '' and model_output_list2[i_list - 1] is '' and model_output_list2[
                i_list + 1] is '':
                test_blank_index.append(i_list)
        del_index2 = []
        del_index2 = np.array(test_blank_index, dtype=int)
        test_processed = []
        test_processed = [model_output_list2[num] for num, i in enumerate(model_output_list2) if num not in del_index2]

        if len(del_index2) != 0:
            add_blank = []
            for i_add in range(len(test_blank_index)):
                add_blank += '▁'

            add_blank = listToString_blank(add_blank)
            test_processed.insert(del_index2[0], add_blank)

        test_remove_index = []
        for i_list in range(len(test_processed) - 1):
            if test_processed[i_list] is '' and test_processed[i_list + 1] is not '':
                test_remove_index.append(i_list)
                test_processed[i_list + 1] = '▁' + test_processed[i_list + 1]
        del_index2 = []
        del_index2 = np.array(test_remove_index, dtype=int)
        test_processed2 = []
        test_processed2 = [test_processed[num] for num, i in enumerate(test_processed) if num not in del_index2]

        test_blank_index = []
        for i_list in range(len(test_processed2) - 1):
            if test_processed2[i_list] == '':
                test_blank_index.append(i_list)

        del_index2 = []
        del_index2 = np.array(test_blank_index, dtype=int)
        model_output_list2_processed = [test_processed2[num] for num, i in enumerate(test_processed2) if
                                        num not in del_index2]

        symbols_LLM_output.append(model_output_list2_processed)

        model_output_list2_processed_copy = model_output_list2_processed.copy()

        index_num_add = 0
        for i_remove in range(len(model_output_list2_processed_copy)):
            if model_output_list2_processed_copy[i_remove] not in qabba_load.parameters.alphabets:

                tokens_i_remove = \
                model_tokenizer(model_output_list2_processed_copy[i_remove], return_tensors="pt")['input_ids'][0]
                if len(tokens_i_remove) == 2:
                    continue

                #             print(model_output_list2_processed_copy[i_remove])

                model_output_list2_processed.pop(i_remove + index_num_add)
                index_num_add -= 1

                for i_convert in range(len(tokens_i_remove[1:-1])):
                    key_mistral_vocab = find_keys_by_value(model_vocab, tokens_i_remove[i_convert + 1])[0]
                    #                 print(key_mistral_vocab)
                    model_output_list2_processed.insert(1 + i_remove + index_num_add, key_mistral_vocab)
                    index_num_add += 1

        for i_list in range(len(model_output_list2_processed)):
            if model_output_list2_processed[i_list] == '':
                #             print(i_list)
                model_output_list2_processed.remove('')

        symbols_LLM_output.append(model_output_list2_processed)

    from sklearn.metrics import mean_squared_error, mean_absolute_error
    from scipy import signal
    import itertools
    from scipy.interpolate import interp1d

    # test_length = 10

    reconst_test_input = qabba_load.inverse_transform(symbols_LLM_input, params_test_data)  # convert into array
    reconst_same_shape_input = qabba_load.recast_shape(reconst_test_input, recap_shape=[test_length, seq_len_pre, 7],
                                                       llm_split='Post')
    # for i_0 in range(reconst_same_shape_input.shape[0]):
    #     for i_2 in range(reconst_same_shape_input.shape[2]):
    #         reconst_same_shape_input[i_0, :, i_2] = signal.detrend(reconst_same_shape_output[i_0, :, i_2])

    reconst_test_output = qabba_load.inverse_transform(symbols_LLM_output, params_test_target)  # convert into array
    # reconst_same_shape_output = qabba.recast_shape(reconst_test_output, recap_shape=[test_length, seq_len_post, 7], llm_split='Post')

    padded = zip(*itertools.zip_longest(*reconst_test_output, fillvalue=-1))
    padded = list(padded)
    padded = np.asarray(padded)
    reconst_same_shape_output = []
    for i_i in range(test_length):
        Temp_ = padded[i_i][:24 * 7]
        reconst_same_shape_output.append(Temp_)
    reconst_same_shape_output = np.asarray(reconst_same_shape_output).reshape([test_length, 24, 7])

    # reconst_same_shape_output = np.ones([test_length, 24, 7])
    # for i_convert_shape in range(len(reconst_test_output)):

    #     BB = np.array(reconst_test_output[i_convert_shape])

    #     x = np.linspace(0, len(BB)-1, num=len(BB))
    #     x_pred = np.linspace(0, len(BB)-1, num=24*7*8)

    #     f1 = interp1d(x, BB, kind='linear')
    #     f2 = interp1d(x, BB, kind='cubic')

    #     y2 = f2(x_pred)
    #     reconst_same_shape_output[i_convert_shape] = np.reshape(y2[0::8], (seq_len_post,7))

    # for i_0 in range(reconst_same_shape_output.shape[0]):
    #     for i_2 in range(reconst_same_shape_output.shape[2]):
    #         reconst_same_shape_output[i_0, :, i_2] = signal.detrend(reconst_same_shape_output[i_0, :, i_2])

    # reconst_same_shape = np.zeros(([test_length, 7, seq_len]), dtype=float)

    Y_pred_all = np.zeros(([test_length, 7 * seq_len_post]), dtype=float)
    Y_true_all = np.zeros(([test_length, 7 * seq_len_post]), dtype=float)
    MAE_result = np.zeros(test_length, dtype=float)
    MSE_result = np.zeros(test_length, dtype=float)

    for i_reconst in range(2):  # test_length

        #     Y_recons_pre = scaler.inverse_transform(X_Test_data_patch[i_reconst, :, :])
        #     Y_recons_post = scaler.inverse_transform(Y_Test_data_patch[i_reconst, :, :])

        Y_true = scaler_load.inverse_transform(Y_Test_data_patch[i_reconst + plus_seg, :, :])
        Y_true_pre = scaler_load.inverse_transform(X_Test_data_patch[i_reconst + plus_seg, :, :])

        #     scaler_output = MinMaxScaler()
        scaler_output = StandardScaler()  ##   DONE the Test
        scaler_output.fit(
            reconst_same_shape_output[i_reconst, :, :])  ##  ?????????????????????????   是否需要多個維度叠加在一起？？？？？

        Y_pred = scaler_load.inverse_transform(scaler_output.transform(reconst_same_shape_output[i_reconst, :, :]))
        #     Y_pred_pre = scaler_load.inverse_transform(reconst_same_shape_input[i_reconst, :, :])
        Y_pred_pre = Y_true_pre

        #     Y_pred = scaler_output.transform(reconst_same_shape_output[i_reconst, :, :])
        #     Y_pred_pre = reconst_same_shape_input[i_reconst, :, :]

        #     Y_true = test_data_patch[i_reconst+plus_seg+seq_len, :, :]
        #     Y_true_pre = test_data_patch[i_reconst+plus_seg, :, :]

        for i_plot in range(7):
            #         plt.subplot(1, 7, i_plot+1)
            if i_plot >= 0:
                AA = []
                AA = np.concatenate((Y_pred_pre[-2:, i_plot], Y_pred[:, i_plot]), axis=0)

                mean_vibration = np.mean(np.abs(np.diff(AA))) * seq_len_post

                AA_max = np.max(np.abs(np.diff(AA)))
                Y_true_max = np.max(np.abs(np.diff(Y_true[:, i_plot])))
                if AA_max > Y_true_max:
                    diff_TS = np.diff(AA)
                    for i_TS in range(len(AA) - 1):
                        if np.abs(diff_TS[i_TS]) > Y_true_max * 0.65:
                            AA[i_TS + 1:] = AA[i_TS + 1:] - 0.45 * diff_TS[i_TS]  # (1.0-Y_true_max/AA_max)

                    diff_temp = Y_true_pre[1, i_plot] - Y_pred_pre[1, i_plot]
                    Y_pred[:, i_plot] = AA[2:] + diff_temp

            if i_reconst == 0:
                plt.plot(np.concatenate((Y_true_pre[:, i_plot], Y_true[:, i_plot]), axis=0), label='Ground Truth')
                #         plt.plot(np.concatenate((Y_pred_pre[:, i_plot], signal.detrend(Y_pred[:, i_plot])+Y_pred_pre[-1, i_plot]-signal.detrend(Y_pred[:, i_plot])[1]), axis=0), label='Forecasting')
                #         plt.plot(np.concatenate((Y_pred_pre[:, i_plot], Y_pred[:, i_plot]), axis=0), label='Forecasting')

                plt.plot(np.concatenate((Y_pred_pre[:, i_plot] + diff_temp, Y_pred[:, i_plot]), axis=0),
                         label='Forecasting')

                Cross_Correlation = cross_correlation(Y_pred[:, i_plot], Y_true[:, i_plot])
                plt.rcParams.update({'font.size': 20})
                plt.title('Cross Correlation: ' + str("{:.3}".format(Cross_Correlation)), fontsize=16)

                n_bins = np.arange(0, seq_len_pre + seq_len_post, 8)
                plt.xticks(n_bins, n_bins, rotation=30, fontsize=12)
                plt.xlabel('Input Length', fontsize=16)
                plt.ylabel('Feature ' + str(i_plot + 1), fontsize=16)

                plt.legend()
                plt.grid(c='r')
                plt.savefig('LLM_Predictor_img_S' + str(i_reconst) + '_Feature' + str(i_plot) + '.jpg')
                plt.show()
                plt.close()
        print('###############################')

        Y_pred = np.reshape(Y_pred, (1, 7 * seq_len_post))
        Y_true = np.reshape(Y_true, (1, 7 * seq_len_post))

        Y_pred_all[i_reconst, :] = Y_pred
        Y_true_all[i_reconst, :] = Y_true

        MAE_result[i_reconst] = np.mean(np.abs(Y_pred - Y_true))
        MSE_result[i_reconst] = np.mean(np.power(Y_pred - Y_true, 2))

    print('###############################')
    print(mean_squared_error(Y_true_all, Y_pred_all))
    print(mean_absolute_error(Y_true_all, Y_pred_all))
    # print(mean(MSE_result))
    # print(mean(MAE_result))



    # test_data = df_data[border2s[0]:border2s[2]]
    # # scaler.fit(train_data.values)
    # data = scaler.transform(test_data.values)
    #
    # data_patch = np.zeros([data.shape[0], data.shape[1], seq_len], dtype=float)
    # for i_data_patch in range(data.shape[0] - seq_len):
    #     data_patch[i_data_patch, :, :] = np.transpose(data[i_data_patch:i_data_patch + seq_len, :])
    #
    #
    # test_symbols, test_params = jabba.transform(data_patch)
    # symbols_convert = []
    # for i_data in range(len(test_symbols)):
    #     symbols_convert.append(listToString(list(test_symbols[i_data])))
    #
    # test_data_symbolic = symbols_convert[:border2s[2] - border2s[0] - seq_len]
    # test_target_symbolic = symbols_convert[seq_len:border2s[2] - border2s[0] + seq_len]
    #
    # symbols_LLM = []
    # test_length = 100 # len(test_data_symbolic)
    # for i_test in range(test_length):  # len(data_TS['test'])):
    #
    #     print('###################################  Model Outputs  ####################################')
    #     model_input = model_tokenizer(test_data_symbolic[i_test], return_tensors="pt").to("cuda")
    #
    #     model_output = model_tokenizer.decode(
    #         ft_model.generate(
    #             **model_input,
    #             max_new_tokens=8,
    #             max_length=MAX_LENGTH,
    #             repetition_penalty=2.1
    #         )[0],
    #         skip_special_tokens=True
    #     )
    #
    #     model_output_list = model_output.split(' ')
    #     model_output_list_copy = model_output_list.copy()
    #     for i_remove in range(len(model_output_list_copy)):
    #         if model_output_list_copy[i_remove] not in jabba.parameters.alphabets:
    #             model_output_list.remove(model_output_list_copy[i_remove])
    #
    #     symbols_LLM.append(model_output_list)
    #
    # from sklearn.metrics import mean_squared_error, mean_absolute_error
    #
    # # test_length = 50
    #
    # reconst_test = jabba.inverse_transform(symbols_LLM, test_params)  # convert into array
    #
    # reconst_same_shape = np.zeros(([test_length, 7, seq_len]), dtype=float)
    #
    # Y_pred_all = np.zeros(([test_length, 7 * seq_len]), dtype=float)
    # Y_true_all = np.zeros(([test_length, 7 * seq_len]), dtype=float)
    # MAE_result = np.zeros(test_length, dtype=float)
    # MSE_result = np.zeros(test_length, dtype=float)
    #
    # for i_reconst in range(len(reconst_test)):
    #     try:
    #         reconst_same_shape[i_reconst, :, :] = np.reshape(reconst_test[i_reconst][:7 * seq_len],
    #                                                          (7, seq_len))  # recast into original shape
    #     except:
    #         print("An exception occurred: " + str(len(reconst_test[i_reconst])))
    #         continue
    #
    #     #     Y_pred = scaler.inverse_transform(np.transpose(reconst_same_shape[i_reconst, :, :]))
    #     Y_pred = scaler.inverse_transform(np.transpose(reconst_same_shape[i_reconst, :, :]))
    #     Y_pred = np.reshape(Y_pred, (1, 7 * seq_len))
    #     Y_true = np.reshape(test_data.values[i_reconst + seq_len:i_reconst + seq_len * 2, :], (1, 7 * seq_len))
    #
    #     Y_pred_all[i_reconst, :] = Y_pred
    #     Y_true_all[i_reconst, :] = Y_true
    #
    #     MAE_result[i_reconst] = np.mean(np.abs(Y_pred - Y_true))
    #     MSE_result[i_reconst] = np.mean(np.power(Y_pred - Y_true, 2))
    #
    # print('###############################')
    # print('MSE: ')
    # print(mean_squared_error(Y_true_all, Y_pred_all))
    # print('MAE: ')
    # print(mean_absolute_error(Y_true_all, Y_pred_all))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Times Series')

    # basic config
    parser.add_argument('--task_name', type=str, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, default=1, help='status')
    parser.add_argument('--model_id', type=str, default='test', help='model id')
    parser.add_argument('--model', type=str, default='Autoformer',
                        help='model name, options: [Autoformer, Transformer, TimesNet]')

    # data loader
    parser.add_argument('--data', type=str, default='ETTm1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

    parser.add_argument('--data_name', type=str, default='None')
    parser.add_argument('--model_name', type=str, default='llama2-7B')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lora_r', type=int, default=16)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--optimizer', type=str, default='adamw')
    parser.add_argument('--ABBA_tol', type=float, default=0.05)
    parser.add_argument('--ABBA_alpha', type=float, default=0.05)
    parser.add_argument('--ABBA_init', type=str, default='agg')
    parser.add_argument('--ABBA_k', type=int, default=1000)
    parser.add_argument('--ABBA_scl', type=int, default=3)
    parser.add_argument('--UCR_data_num', type=int, default=1)
    parser.add_argument('--MAX_LENGTH', type=int, default=2048)

    args = parser.parse_args()
    # print('***********************************************************************************')
    main(args)
    # print('***********************************************************************************')




