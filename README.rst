LLM-ABBA
========

.. image:: https://img.shields.io/pypi/v/llmabba?color=lightsalmon
   :alt: PyPI Version
   :target: https://pypi.org/project/llmabba/

.. image:: https://img.shields.io/pypi/dm/llmabba.svg?label=PyPI%20downloads
   :alt: PyPI Downloads
   :target: https://pypi.org/project/llmabba/

.. image:: https://img.shields.io/badge/Cython_Support-Accelerated-blue?style=flat&logoColor=cyan&labelColor=cyan&color=black
   :alt: Cython Support
   :target: https://github.com/inEXASCALE/llm-abba

.. image:: https://readthedocs.org/projects/llm-abba/badge/?version=latest
   :alt: Documentation Status
   :target: https://llm-abba.readthedocs.io/en/latest/

.. image:: https://img.shields.io/github/license/inEXASCALE/llm-abba
   :alt: License
   :target: https://github.com/inEXASCALE/llm-abba/blob/main/LICENSE

`llmabba` is a software framework for time series analysis using Large Language Models (LLMs) based on symbolic representation, as introduced in the paper `LLM-ABBA: Symbolic Time Series Approximation using Large Language Models <https://arxiv.org/abs/2411.18506>`_.

Time series analysis involves identifying patterns, trends, and structures within data sequences. Traditional methods like discrete wavelet transforms or symbolic aggregate approximation (SAX) convert continuous time series into symbolic representations for better analysis and compression. However, these methods often struggle with complex patterns.

`llmabba` enhances these techniques by leveraging LLMs, which excel in pattern recognition and sequence prediction. By applying LLMs to symbolic time series, `llmabba` discovers rich, meaningful representations, offering:

- **Higher accuracy and compression**: Better symbolic representations via LLMs, improving data compression and pattern accuracy.
- **Adaptability**: Robust performance across domains like finance, healthcare, and environmental science.
- **Scalability**: Efficient handling of large-scale time series datasets.
- **Automatic feature discovery**: Uncovers novel patterns that traditional methods may miss.

Key Features
------------
- Symbolic Time Series Approximation: Converts time series into symbolic representations.
- LLM-Powered Encoding: Enhances compression and pattern discovery.
- Efficient and Scalable: Suitable for large-scale datasets.
- Flexible Integration: Compatible with machine learning and statistical workflows.

Installation
------------
To set up a virtual environment, use one of these methods:

**Using venv**:

.. code-block:: bash

   mkdir ~/.myenv
   python -m venv ~/.myenv
   source ~/.myenv/bin/activate

**Using conda**:

.. code-block:: bash

   conda create -n myenv
   conda activate myenv

Install `llmabba` via pip:

.. code-block:: bash

   pip install llmabba

Usage
-----
Refer to the documentation and the `examples` folder for detailed usage.

To use quantized ABBA with fixed-point adaptive piecewise linear continuous approximation (FAPCA):

.. code-block:: python

   from llmabba import ABBA

   ts = [[1.2, 1.4, 1.3, 1.8, 2.2, 2.4, 2.1], [1.2, 1.3, 1.2, 2.2, 1.4, 2.4, 2.1]]
   abba = ABBA(tol=0.1, alpha=0.1)
   symbolic_representation = abba.encode(ts)
   print("Symbolic Representation:", symbolic_representation)
   reconstruction = abba.decode(symbolic_representation)
   print("Reconstruction:", reconstruction)

For time series classification:

.. code-block:: python

   import pandas as pd
   import numpy as np
   from llmabba.llmabba import LLMABBA
   from sklearn.model_selection import train_test_split

   project_name = "PTBDB"
   task_type = "classification"
   model_name = 'mistralai/Mistral-7B-Instruct-v0.1'
   prompt_input = """This is a classification task. Identify the 'ECG Abnormality' according to the given 'Symbolic Series'."""

   # Load and process data
   abnormal_df = pd.read_csv('../test_data/ptbdb_abnormal.csv', header=None)
   normal_df = pd.read_csv('../test_data/ptbdb_normal.csv', header=None)

   Y_data = np.concatenate((np.zeros([abnormal_df.shape[0]], dtype=int), np.ones([normal_df.shape[0]], dtype=int)))
   X_data = pd.concat([abnormal_df, normal_df]).to_numpy()

   arranged_seq = np.random.permutation(len(Y_data))
   train_data, test_data, train_target, test_target = train_test_split(X_data[arranged_seq], Y_data[arranged_seq], test_size=0.2)

   train_data_split = {'X_data': train_data[:500], 'Y_data': train_target[:500]}

   # Train with LLM-ABBA
   LLMABBA_classification = LLMABBA()
   model_input, model_tokenizer = LLMABBA_classification.model(model_name=model_name, max_len=2048)

   tokenized_train_dataset, tokenized_val_dataset = LLMABBA_classification.process(
       project_name=project_name,
       data=train_data_split,
       task=task_type,
       prompt=prompt_input,
       alphabet_set=-1,
       model_tokenizer=model_tokenizer,
       scalar="z-score"
   )

   LLMABBA_classification.train(
       model_input=model_input,
       num_epochs=1,
       output_dir='../save/',
       train_dataset=tokenized_train_dataset,
       val_dataset=tokenized_val_dataset
   )

   # Inference
   test_data = np.expand_dims(test_data[1], axis=0)
   peft_model_input, model_tokenizer = LLMABBA_classification.model(
       peft_file='../llm-abba-master/save/checkpoint-25/',
       model_name=model_name,
       max_len=2048
   )

   out_text = LLMABBA_classification.inference(
       project_name=project_name,
       data=test_data,
       task=task_type,
       prompt=prompt_input,
       ft_model=peft_model_input,
       model_tokenizer=model_tokenizer,
       scalar="z-score",
       llm_max_length=256,
       llm_repetition_penalty=1.9,
       llm_temperature=0.0,
       llm_max_new_tokens=2
   )

   print(out_text)

Visualization
------------
Currently under development.

Common Questions
---------------
**1. Why are some LLM-generated tokens not recognized by ABBA?**

LLM hallucination may generate tokens outside ABBA’s alphabet. Solutions include:

- **Method A**: Skip or replace unrecognized tokens, using `repetition_penalty=1.3` and `temperature=0.0`.
- **Method B**: Use as many LLM tokens as possible (e.g., 32,000 for Mistral) to avoid this issue.
- **Method C**: Use ASCII symbols instead of LLM tokens, specifying them in the prompt (under testing).
- **Method D**: Fine-tune the LLM with `add_tokens` (not recommended).

**2. Is inhibition level necessary?**

Adjusting the inhibition level helps LLMs adapt to downstream tasks, especially with special symbols (Method C).

Contributing
------------
To contribute:

1. Fork the repository.
2. Create a feature or bugfix branch.
3. Submit a pull request.

License
-------
Released under the MIT License.

Contact
-------
For questions, use GitHub issues or contact the paper’s authors.

References
----------
- Carson, E., Chen, X., and Kang, C., "Understanding time series via symbolic approximation," arXiv:2411.18506, 2024. `doi:10.48550/arXiv.2411.18506 <https://arxiv.org/abs/2411.18506>`_.