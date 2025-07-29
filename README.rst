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
