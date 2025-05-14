from transformers import DataCollatorForLanguageModeling, DataCollatorWithPadding, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, Trainer, TrainingArguments
from accelerate import FullyShardedDataParallelPlugin, Accelerator
import math
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model, TaskType
from sklearn import preprocessing
import numpy as np
from .xabba import XABBA
import pickle
import os
import torch
from peft import PeftModel
from .utils.fundamentals import *
from .utils.regressor_tools import process_data, fit_regressor, calculate_regression_metrics
from sklearn.model_selection import train_test_split
from datetime import datetime
from datasets.dataset_dict import DatasetDict
from datasets import Dataset
import matplotlib.pyplot as plt

def save_abba(model, filename):
    pickle.dump(model, file = open(filename, "wb"))


def load_abba(filename):
    return pickle.load(open(filename, "rb"))


# Python program to convert a list to string
def ListToString(s):
    # initialize an empty string
    str1 = ""
    # traverse in the string
    for ele in s:
        str1 += ele + ' '
    # return string
    return str1


# Python program to convert a string to list
def StringToList(s):
    # initialize an empty string
    str1 = []
    # traverse in the string
    for ele in range(int(len(s)/2)):
        str1.append(s[ele*2])
    # return string
    return str1


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


class LLMABBA:
    def __init__(self,
                 abba_tol = 0.05,
                 abba_init = 'agg',
                 abba_alpha = 0.01,
                 bits_for_len = 16,
                 bits_for_inc = 16,
                 abba_scl = 3,
                 abba_verbose = 0,
                 lora_r = 16,
                 lora_alpha = 16,
                 target_modules = None,
                 modules_to_save = ["embed_tokens"],
                 lora_dropout = 0.05,
                 quant_process = True
                 ):

        self.abba_tol = abba_tol
        self.abba_init = abba_init
        self.abba_alpha = abba_alpha
        self.bits_for_len = bits_for_len
        self.bits_for_inc = bits_for_inc
        self.abba_scl = abba_scl
        self.abba_verbose = abba_verbose
        self.lora_r = lora_r 
        self.lora_alpha = lora_alpha
        self.quant_process = quant_process
        if target_modules is not None:
            self.target_modules = target_modules
        else:
            self.target_modules = [
                            "q_proj",
                            "k_proj",
                            "v_proj",
                            "o_proj",
                            "gate_proj",
                            "up_proj",
                            "down_proj",
                            "lm_head",
                        ]

        self.modules_to_save = modules_to_save
        self.lora_dropout = lora_dropout


    def tokenize(self, prompt_input):
        result = self.model_tokenizer(
            prompt_input,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
        )
        result["labels"] = result["input_ids"].copy()
        return result

    def build(self, data_point):
        """Build the Prompt Data"""
        full_prompt = f"""{self.prompt}
        ### Symbolic Series: {data_point["text_inputs"]}
        ### Results: {data_point["text_outputs"]}
        """
        return self.tokenize(full_prompt)


    def process(self, project_name, data, task, prompt, model_tokenizer,
                seq_len_pre = 24, scalar="z-score",
                seq_len_post = 24, alphabet_set=None, draw_ts = False)-> None:
        """Load data and process data"""

        if alphabet_set is None:
            vocab_list = model_tokenizer.get_vocab()
            alphabet_set = list(vocab_list.keys())
            print("Using LLM tokens as the alphabet_set")
        else:
            print("Using self-defined alphabet_set")

        self.project_name = project_name
        self.task = task
        self.alphabet_set = alphabet_set
        self.prompt = prompt
        self.model_tokenizer = model_tokenizer
        self.seq_len_pre = seq_len_pre
        self.seq_len_post = seq_len_post

        # Initialize the StandardScaler
        if scalar == "min-max":
            self.scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
        elif scalar == "z-score":
            self.scaler =  preprocessing.StandardScaler()

        # Initialize the ABBA block
        self.xabba = XABBA(tol=self.abba_tol, init=self.abba_init,
                           alpha=self.abba_alpha, scl=self.abba_scl, verbose=0)

        if self.task == "classification":
            # Fit the scaler to the training data and transform it
            X_data = self.scaler.fit_transform(data['X_data'])
            Y_data = data['Y_data']

            symbols = self.xabba.fit_transform(X_data, alphabet_set=self.alphabet_set)
            if draw_ts is True:
                reconstruction = self.xabba.inverse_transform(symbols)
                reconst_same_shape = self.xabba.recast_shape(reconstruction)  # recast into original shape

                plt.plot(X_data[0,:], label='Ground Truth')
                plt.plot(reconst_same_shape[0.:], label='Reconstruction')

                Cross_Correlation = cross_correlation(
                    X_data[0,:],
                    reconst_same_shape[0.:])
                plt.rcParams.update({'font.size': 20})
                # plt.title('Cross Correlation: ' + str("{:.3}".format(Cross_Correlation)), fontsize=20)
                plt.title('A Classification Task Before LLM-ABBA', fontsize=20)
                plt.xlabel('Inout Length', fontsize=20)
                plt.ylabel('Feature', fontsize=20)

                plt.legend()
                plt.grid(True, axis='y')
                plt.tight_layout()
                # plt.savefig('Compression_Reproduction_img_S' + str(i_reconst) + '_Feature' + str(i_plot) + '.jpg')
                plt.show()
                # plt.close()
                ##################################################################

            symbols_convert = []
            for i_data in range(len(symbols)):
                symbols_convert.append(listToString(list(symbols[i_data])))

            train_data_symbolic, val_data_symbolic, train_target_symbolic, val_target_symbolic = train_test_split(
                symbols_convert, Y_data, test_size=0.2)


        elif self.task == "regression":

            # Fit the scaler to the training data and transform it
            min_len = np.inf
            for i in range(len(data['X_data'])):
                x = data['X_data'].iloc[i, :]
                all_len = [len(y) for y in x]
                min_len = min(min(all_len), min_len)
            # print("[{}] Minimum length: {}".format(module, min_len))

            X_data, self.scaler = process_data(data['X_data'], scaler=self.scaler, min_len=min_len)
            Y_data = data['Y_data']

            symbols = self.xabba.fit_transform(X_data, alphabet_set=self.alphabet_set)
            if draw_ts is True:
                reconstruction = self.xabba.inverse_transform(symbols)
                reconst_same_shape = self.xabba.recast_shape(reconstruction)  # recast into original shape

                plt.plot(X_data[0,:], label='Ground Truth')
                plt.plot(reconst_same_shape[0.:], label='Reconstruction')

                Cross_Correlation = cross_correlation(
                    X_data[0,:],
                    reconst_same_shape[0.:])
                plt.rcParams.update({'font.size': 20})
                # plt.title('Cross Correlation: ' + str("{:.3}".format(Cross_Correlation)), fontsize=20)
                plt.title('A Regression Task Before LLM-ABBA', fontsize=20)
                plt.xlabel('Inout Length', fontsize=20)
                plt.ylabel('Feature', fontsize=20)

                plt.legend()
                plt.grid(True, axis='y')
                plt.tight_layout()
                # plt.savefig('Compression_Reproduction_img_S' + str(i_reconst) + '_Feature' + str(i_plot) + '.jpg')
                plt.show()
                # plt.close()
                ##################################################################


            symbols_convert = []
            for i_data in range(len(symbols)):
                symbols_convert.append(listToString(list(symbols[i_data])))

            train_data_symbolic, val_data_symbolic, train_target_symbolic, val_target_symbolic = train_test_split(
                symbols_convert, Y_data, test_size=0.2)

        elif self.task == "forecasting":
            # Fit the scaler to the training data and transform it
            self.scaler.fit(data)
            data_scaled = self.scaler.transform(data)

            X_Train_data_patch = np.zeros(
                [data_scaled.shape[0] - (self.seq_len_pre + self.seq_len_post), self.seq_len_pre,
                 data_scaled.shape[1]],
                dtype=float)

            Y_Train_data_patch = np.zeros(
                [data_scaled.shape[0] - (self.seq_len_pre + self.seq_len_post), self.seq_len_post,
                 data_scaled.shape[1]],
                dtype=float)

            for i_data_patch in range(data_scaled.shape[0] - (self.seq_len_pre + self.seq_len_post)):
                X_Train_data_patch[i_data_patch, :, :] = data_scaled[i_data_patch:i_data_patch + self.seq_len_pre,
                                                         :]
                Y_Train_data_patch[i_data_patch, :, :] = data_scaled[
                                                         i_data_patch + self.seq_len_pre:i_data_patch + self.seq_len_pre + self.seq_len_post,
                                                         :]

            test_blank_index = []
            for i_list in range(len(self.alphabet_set)):
                if ('▁▁' in self.alphabet_set[i_list]):
                    test_blank_index.append(i_list)
            del_index1 = np.array(test_blank_index, dtype=int)
            vocab_list_processed = [vocab_list[num] for num, i in enumerate(vocab_list) if num not in del_index1]
            self.alphabet_set = vocab_list_processed

            symbols_train_data = self.xabba.fit_transform(X_Train_data_patch, alphabet_set=self.alphabet_set, llm_split='Pre')

            symbols_train_target, params_train_target = self.xabba.transform(Y_Train_data_patch, llm_split='Post')

            if draw_ts is True:
                reconstruction_train_data = self.xabba.inverse_transform(symbols_train_data)
                train_data_same_shape = self.xabba.recast_shape(reconstruction_train_data, llm_split='Pre')  # recast into original shape

                reconstruction_train_target = self.xabba.inverse_transform(symbols_train_target, params_train_target)
                train_target_same_shape = self.xabba.recast_shape(reconstruction_train_target, recap_shape=Y_Train_data_patch.shape, llm_split='Post')  # recast into original shape

                ##################################################################
                Y_true_pre = self.scaler.inverse_transform(X_Train_data_patch[0, :, :])
                Y_true_post = self.scaler.inverse_transform(Y_Train_data_patch[0, :, :])

                Y_recons_pre = self.scaler.inverse_transform(train_data_same_shape[0, :, :])
                Y_recons_post = self.scaler.inverse_transform(train_target_same_shape[0, :, :])

                plt.plot(np.concatenate((Y_true_pre[:, 0], Y_true_post[:, 0]), axis=0), label='Ground Truth')
                plt.plot(np.concatenate((Y_recons_pre[:, 0], Y_recons_post[:, 0]), axis=0),
                         label='Reconstruction')

                Cross_Correlation = cross_correlation(
                    np.concatenate((Y_true_pre[:, 0], Y_true_post[:, 0]), axis=0),
                    np.concatenate((Y_recons_pre[:, 0], Y_recons_post[:, 0]), axis=0))
                plt.rcParams.update({'font.size': 20})
                # plt.title('Cross Correlation: ' + str("{:.3}".format(Cross_Correlation)), fontsize=20)
                plt.title('A Forecasting Task Before LLM-ABBA', fontsize=20)
                plt.xlabel('Inout Length', fontsize=20)
                plt.ylabel('Feature', fontsize=20)

                plt.legend()
                plt.grid(True, axis='y')
                plt.tight_layout()
                # plt.savefig('Compression_Reproduction_img_S' + str(i_reconst) + '_Feature' + str(i_plot) + '.jpg')
                plt.show()
                # plt.close()
                ##################################################################

            train_data_symbolic = []
            for i_data in range(len(symbols_train_data)):
                train_data_symbolic.append(listToString(list(symbols_train_data[i_data])))

            train_target_symbolic = []
            for i_data in range(len(symbols_train_target)):
                train_target_symbolic.append(listToString(list(symbols_train_target[i_data])))

            arranged_seq = np.random.randint(len(train_data_symbolic), size=int(len(train_data_symbolic) * 0.2))

            val_data_symbolic = [train_data_symbolic[index] for index in arranged_seq]
            val_target_symbolic = [train_target_symbolic[index] for index in arranged_seq]

        else:
            raise NotImplementedError("Method is not implemented, please contact the maintenance team.")

        data_TS = DatasetDict({
            'train': Dataset.from_dict({'text_outputs': train_target_symbolic, 'text_inputs': train_data_symbolic}),
            'val': Dataset.from_dict({'text_outputs': val_target_symbolic, 'text_inputs': val_data_symbolic})
        })

        # Check whether the specified path exists or not
        path = "../save/" + self.project_name + "/"
        isExist = os.path.exists(path)
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(path)
            print("The new project directory is created!")

        #### Saving Scaler and ABBA
        output_scaler = open(str("../save/" + self.project_name + "/" + self.task  + "_Scaler" + "_save.pkl"), 'wb')

        str1 = pickle.dumps(self.scaler)
        output_scaler.write(str1)
        output_scaler.close()

        # curr_loc = os.path.dirname(os.path.realpath(__file__))
        save_abba(self.xabba, str("../save/" + self.project_name + "/" + self.task  + "_ABBA" + "_save.pkl"))

        tokenized_train_dataset = data_TS['train'].map(self.build)
        tokenized_val_dataset = data_TS['val'].map(self.build)

        tokenized_train_dataset.set_format("torch")
        tokenized_val_dataset.set_format("torch")

        return tokenized_train_dataset, tokenized_val_dataset



    def model(self, peft_file = None, model_name = None, max_len = 2048) -> None:

        self.model_name = model_name
        self.max_len = max_len

        if peft_file is None:
            fsdp_plugin = FullyShardedDataParallelPlugin(
                state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
                optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False),
            )
            accelerator = Accelerator(fsdp_plugin=fsdp_plugin)

            """Model selection with parameters"""
            if self.quant_process is True:
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

                model_tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    model_max_len=max_len,
                    padding_side="right",
                    truncation=True,
                    add_eos_token=True,
                )

                model_tokenizer.padding_side = 'right'
                model_tokenizer.pad_token = model_tokenizer.eos_token
                model_tokenizer.pad_token_id = model_tokenizer.eos_token_id

                model_input = AutoModelForCausalLM.from_pretrained(
                    pretrained_model_name_or_path=model_name,
                    quantization_config=quantization_config,
                    # device_map="auto",
                    trust_remote_code=True,
                    torch_dtype=torch.bfloat16,
                )

                model_input.config.pad_token_id = model_input.config.eos_token_id
                model_input = prepare_model_for_kbit_training(model_input)

            else:

                model_tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    model_max_len=max_len,
                    padding_side="right",
                    truncation=True,
                    add_eos_token=True,
                )

                model_tokenizer.padding_side = 'right'
                model_tokenizer.pad_token = model_tokenizer.eos_token
                model_tokenizer.pad_token_id = model_tokenizer.eos_token_id
                model_input = AutoModelForCausalLM.from_pretrained(
                    pretrained_model_name_or_path=model_name,
                    # device_map="auto",
                    trust_remote_code=True,
                )
                # Data collator for padding a batch of examples to the maximum length seen in the batch
                self.model_data_collator = DataCollatorWithPadding(tokenizer=model_tokenizer)

                model_input.config.pad_token_id = model_input.config.eos_token_id
                model_input = prepare_model_for_kbit_training(model_input)

            #####   Loading LoRA config and PEFT model
            lora_config = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                r=self.lora_r,  # the dimension of the low-rank matrices
                lora_alpha=self.lora_alpha,  # scaling factor for LoRA activations vs pre-trained weight activations
                target_modules=self.target_modules,
                lora_dropout=self.lora_dropout,  # dropout probability of the LoRA layers
                bias='none',  # wether to train bias weights, set to 'none' for attention layers
                modules_to_save=self.modules_to_save,
            )

            model_input = get_peft_model(model_input, lora_config)
            # Apply the accelerator. You can comment this out to remove the accelerator.
            model_input = accelerator.prepare_model(model_input)

            model_input.print_trainable_parameters()
            model_input = model_input.cuda()

        else:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )

            model = AutoModelForCausalLM.from_pretrained(
                model_name,  # Mistral, same as before
                quantization_config=bnb_config,  # Same quantization config as before
                # device_map="auto",
                trust_remote_code=True,
            )

            model_tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                model_max_length=self.max_len,
                padding_side="right",
                truncation=True,
                add_eos_token=True,
            )

            model_tokenizer.padding_side = 'right'
            model_tokenizer.pad_token = model_tokenizer.eos_token
            model_tokenizer.pad_token_id = model_tokenizer.eos_token_id
            print(len(model_tokenizer))

            model_input = PeftModel.from_pretrained(model, peft_file)

        self.model_tokenizer = model_tokenizer
        return model_input, model_tokenizer



    def train(self, model_input, num_epochs, output_dir, train_dataset, val_dataset):
        """Train with validation"""
        if torch.cuda.device_count() > 1:  # If more than 1 GPU
            model_input.is_parallelizable = True
            model_input.model_parallel = True

        run_name = self.task + "-" + self.model_name

        trainer = Trainer(
            model=model_input,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            args=TrainingArguments(
                output_dir=output_dir,
                warmup_steps=5,
                per_device_train_batch_size=4,
                gradient_checkpointing=True,
                gradient_accumulation_steps=4,
                learning_rate=2e-4,  # Want about 10x smaller than the Mistral learning rate
                bf16=False,
                optim="paged_adamw_8bit",
                num_train_epochs=num_epochs,
                weight_decay=0.00005,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                logging_strategy="epoch",
                logging_dir="./logs",
                load_best_model_at_end=True,
                report_to="wandb",
                save_total_limit=1,
                do_eval=True,  # Perform evaluation at the end of training
                # report_to="wandb",           # Comment this out if you don't want to use weights & baises
                run_name=f"{run_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"  # Name of the W&B run (optional)
            ),
            data_collator=DataCollatorForLanguageModeling(self.model_tokenizer, mlm=False),
        )

        model_input.config.use_cache = False  # silence the warnings. Please re-enable for inference!
        trainer.train()


    def inference(self, project_name, data, task, prompt, ft_model, model_tokenizer, scalar="z-score",
                  seq_len_pre = 24, seq_len_post = 24,
                  llm_max_length=1024, llm_repetition_penalty=1.7,
                  llm_temperature=0.0, llm_max_new_tokens=256, draw_ts = False
                )-> None:
        """Inference """

        # self.task = task
        # self.model_tokenizer = model_tokenizer


        # Initialize the StandardScaler
        if scalar == "min-max":
            self.scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
        elif scalar == "z-score":
            self.scaler =  preprocessing.StandardScaler()

        # Initialize the ABBA block
        self.xabba = XABBA(tol=self.abba_tol, init=self.abba_init,
                           alpha=self.abba_alpha, scl=self.abba_scl, verbose=0)

        with open("../save/" + project_name + "/" + task  + "_ABBA" + "_save.pkl", 'rb') as file:
            self.xabba = pickle.loads(file.read())

        with open(str("../save/" + project_name + "/" + task  + "_Scaler" + "_save.pkl"), 'rb') as file:
            self.scaler = pickle.loads(file.read())

        if (task == "classification") or (task == "regression"):
            data_scaled = self.scaler.transform(data)
            symbols_test_data, params_test_data = self.xabba.transform(data_scaled)

        elif task == "forecasting":
            data_scaled = self.scaler.transform(data)
            symbols_test_data, params_test_data = self.xabba.transform(data_scaled, llm_split='Post')
            reconstruction_test_data = self.xabba_load.inverse_transform(symbols_test_data, params_test_data)
            test_data_same_shape = self.xabba.recast_shape(reconstruction_test_data,
                                                           recap_shape=data_scaled.shape,
                                                           llm_split='Post')  # recast into original shape

        else:
            print("The task is UNKNOWN!")
            pass

        test_data_symbolic = ListToString(list(symbols_test_data[0]))

        if prompt is not None:
            test_prompt = f"""{prompt}
            ### Symbolic Series: {test_data_symbolic}
            ### Results:
            """
        else:
            print("Please provide your Prompt!")

        model_input_text = model_tokenizer(test_prompt, return_tensors="pt").to('cuda')

        llm_out = model_tokenizer.decode(
            ft_model.generate(
                **model_input_text,
                max_new_tokens=llm_max_new_tokens,
                max_length=llm_max_length,
                repetition_penalty=llm_repetition_penalty,
                temperature=llm_temperature,
            )[0],
            skip_special_tokens=True
        )

        if (task == "classification") or (task == "regression"):
            return llm_out

        elif task == "forecasting":
            split_content = llm_out.split('### Results:')
            listed_split_content = StringToList(split_content[1])

            ################################################################## Rescue
            tokens_output_mistral = listed_split_content[1:-1]
            model_output_list2_processed = tokens_output_mistral
            for symbols_i in range(len(tokens_output_mistral)):
                if tokens_output_mistral[symbols_i] not in self.xabba.parameters.alphabets:
                    if symbols_i == 0:
                        model_output_list2_processed[symbols_i] = tokens_output_mistral[-1]
                    else:
                        model_output_list2_processed[symbols_i] = tokens_output_mistral[symbols_i - 1]


            reconst_test_output = self.xabba.inverse_transform(model_output_list2_processed, params_test_data)  # convert into array
            reconst_same_shape_output = self.xabba.recast_shape(reconst_test_output, recap_shape=[seq_len_post, data_scaled.size()[2]], llm_split='Post')
            forecasting_out = self.scaler.inverse_transform(reconst_same_shape_output)


            # if draw_ts is True:
                # ##################################################################
                # plt.plot(np.concatenate((Y_true_pre[:, i_plot], Y_true_post[:, i_plot]), axis=0), label='Ground Truth')
                # plt.plot(np.concatenate((Y_recons_pre[:, i_plot], Y_recons_post[:, i_plot]), axis=0),
                #          label='Reconstruction')
                #
                # Cross_Correlation = cross_correlation(
                #     np.concatenate((Y_true_pre[:, i_plot], Y_true_post[:, i_plot]), axis=0),
                #     np.concatenate((Y_recons_pre[:, i_plot], Y_recons_post[:, i_plot]), axis=0))
                # plt.rcParams.update({'font.size': 20})
                # plt.title('Cross Correlation: ' + str("{:.3}".format(Cross_Correlation)), fontsize=20)
                # plt.xlabel('Inout Length', fontsize=20)
                # plt.ylabel('Feature ' + str(i_plot + 1), fontsize=20)
                #
                # plt.legend()
                # plt.grid(True, axis='y')
                # plt.tight_layout()
                # # plt.savefig('Compression_Reproduction_img_S' + str(i_reconst) + '_Feature' + str(i_plot) + '.jpg')
                # plt.show()
                # # plt.close()
                # ##################################################################

            return forecasting_out