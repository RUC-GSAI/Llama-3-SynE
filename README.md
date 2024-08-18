<!-- ---
language:
- en
- zh
datasets:
- survivi/Llama-3-SynE-Dataset
library_name: transformers
pipeline_tag: text-generation
--- -->

<!-- ---
language:
- en
- zh
task_categories:
- text-generation
--- -->

<p align="center">
  <img src="https://github.com/RUC-GSAI/Llama-3-SynE/blob/main/assets/llama-3-syne-logo.png" width="400"/>
</p>

<p align="center">
 📄 <a href="https://arxiv.org/abs/2407.18743"> Report </a>&nbsp | &nbsp 🤗 <a href="https://huggingface.co/survivi/Llama-3-SynE">Model on Hugging Face</a>&nbsp | &nbsp 📊 <a href="https://huggingface.co/datasets/survivi/Llama-3-SynE-Dataset">CPT Dataset</a>
</p>

<p align="center">
 🔍 <a href="https://github.com/RUC-GSAI/Llama-3-SynE/blob/main/README.md">English</a>&nbsp | &nbsp<a href="https://github.com/RUC-GSAI/Llama-3-SynE/blob/main/README_zh.md">简体中文</a>
</p>

<!-- <p align="center">
  <img src="https://cdn-uploads.huggingface.co/production/uploads/651a29d566e78720a78317ec/I2rqZ19OY2qvW1V6nOakg.png" width="400"/>
</p>

<p align="center">
 📄 <a href="https://arxiv.org/abs/2407.18743"> Report </a>&nbsp | &nbsp 💻 <a href="https://github.com/RUC-GSAI/Llama-3-SynE">GitHub Repo</a>
</p>

<p align="center">
 🔍 <a href="https://huggingface.co/survivi/Llama-3-SynE/blob/main/README.md">English</a>&nbsp | &nbsp<a href="https://huggingface.co/survivi/Llama-3-SynE/blob/main/README_zh.md">简体中文</a>
</p>

> Here is the Llama-3-SynE model. The continual pre-training dataset is also available [here](https://huggingface.co/datasets/survivi/Llama-3-SynE-Dataset). -->

<!-- <p align="center">
  <img src="https://cdn-uploads.huggingface.co/production/uploads/651a29d566e78720a78317ec/I2rqZ19OY2qvW1V6nOakg.png" width="400"/>
</p>

<p align="center">
 📄 <a href="https://arxiv.org/abs/2407.18743"> Report </a>&nbsp | &nbsp 💻 <a href="https://github.com/RUC-GSAI/Llama-3-SynE">GitHub Repo</a>
</p>

<p align="center">
 🔍 <a href="https://huggingface.co/datasets/survivi/Llama-3-SynE-Dataset/blob/main/README.md">English</a>&nbsp | &nbsp<a href="https://huggingface.co/datasets/survivi/Llama-3-SynE-Dataset/blob/main/README_zh.md">简体中文</a>
</p>

> Here is the continual pre-training dataset. The Llama-3-SynE model is available [here](https://huggingface.co/survivi/Llama-3-SynE). -->

---

## News

- 🌟 **Training and evaluation code coming soon:)**
- ✨✨ `2024/08/12`: We released the [continual pre-training dataset](https://huggingface.co/datasets/survivi/Llama-3-SynE-Dataset).
- ✨✨ `2024/08/10`: We released the [Llama-3-SynE model](https://huggingface.co/survivi/Llama-3-SynE).
- ✨ `2024/07/26`: We released the [technical report](https://arxiv.org/abs/2407.18743), welcome to check it out!

## Model Introduction

**Llama-3-SynE** (<ins>Syn</ins>thetic data <ins>E</ins>nhanced Llama-3) is a significantly enhanced version of [Llama-3 (8B)](https://github.com/meta-llama/llama3), achieved through continual pre-training (CPT) to improve its **Chinese language ability and scientific reasoning capability**. By employing a meticulously designed data mixture and curriculum strategy, Llama-3-SynE successfully enhances new abilities while maintaining the original model’s performance. This enhancement process involves utilizing existing datasets and synthesizing high-quality datasets specifically designed for targeted tasks.

Key features of Llama-3-SynE include:

- **Enhanced Chinese Language Capabilities**: Achieved through topic-based data mixture and perplexity-based data curriculum.
- **Improved Scientific Reasoning**: Utilizing synthetic datasets to enhance multi-disciplinary scientific knowledge.
- **Efficient CPT**: Only consuming around 100 billion tokens, making it a cost-effective solution.

## Model List

| Model        | Type | Seq Length | Download                                                      |
| :----------- | :--- | :--------- | :------------------------------------------------------------ |
| Llama-3-SynE | Base | 8K         | [🤗 Huggingface](https://huggingface.co/survivi/Llama-3-SynE) |

## BenchMark

We divide all evaluation benchmarks into two groups. The first group is _major benchmarks_, which aim to evaluate the comprehensive capacities of LLMs. Note that we include commonly used math and code benchmarks in this group because it is standard practice to use these benchmarks for evaluating various general-purpose LLMs.

The second group is _scientific benchmarks_, which have a broader coverage of multidisciplinary scientific knowledge.

We report the eight-shot performance on GSM8K, ASDiv, and MAWPS, five-shot for C-Eval, CMMLU, MMLU, MATH, GaoKao, SciQ, SciEval, SAT-Math, and AQUA-RAT, three-shot for MBPP.
For HumanEval and ARC, we report the zero-shot evaluation performance. The best and second best are in **bold** and <ins>underlined</ins>, respectively.

### Major Benchmarks

| **Models**              | **MMLU**         | **C-Eval**       | **CMMLU**        | **MATH**         | **GSM8K**        | **ASDiv**        | **MAWPS**        | **SAT-Math**     | **HumanEval**    | **MBPP**         |
| :---------------------- | :--------------- | :--------------- | :--------------- | :--------------- | :--------------- | :--------------- | :--------------- | :--------------- | :--------------- | :--------------- |
| Llama-3-8B              | **66.60**        | 49.43            | 51.03            | 16.20            | 54.40            | 72.10            | 89.30            | 38.64            | <ins>36.59</ins> | **47.00**        |
| DCLM-7B                 | 64.01            | 41.24            | 40.89            | 14.10            | 39.20            | 67.10            | 83.40            | <ins>41.36</ins> | 21.95            | 32.60            |
| Mistral-7B-v0.3         | 63.54            | 42.74            | 43.72            | 12.30            | 40.50            | 67.50            | 87.50            | 40.45            | 25.61            | 36.00            |
| Llama-3-Chinese-8B      | 64.10            | <ins>50.14</ins> | <ins>51.20</ins> | 3.60             | 0.80             | 1.90             | 0.60             | 36.82            | 9.76             | 14.80            |
| MAmmoTH2-8B             | 64.89            | 46.56            | 45.90            | **34.10**        | **61.70**        | **82.80**        | <ins>91.50</ins> | <ins>41.36</ins> | 17.68            | 38.80            |
| Galactica-6.7B          | 37.13            | 26.72            | 25.53            | 5.30             | 9.60             | 40.90            | 51.70            | 23.18            | 7.31             | 2.00             |
| **Llama-3-SynE (ours)** | <ins>65.19</ins> | **58.24**        | **57.34**        | <ins>28.20</ins> | <ins>60.80</ins> | <ins>81.00</ins> | **94.10**        | **43.64**        | **42.07**        | <ins>45.60</ins> |

> On **Chinese evaluation benchmarks** (such as C-Eval and CMMLU), Llama-3-SynE significantly outperforms the base model Llama-3 (8B), indicating that our method is very effective in improving Chinese language capabilities.

> On **English evaluation benchmarks** (such as MMLU, MATH, and code evaluation benchmarks), Llama-3-SynE demonstrates comparable or better performance than the base model, indicating that our method effectively addresses the issue of catastrophic forgetting during the CPT process.

### Scientific Benchmarks

"PHY", "CHE", and "BIO" denote the physics, chemistry, and biology sub-tasks of the corresponding benchmarks.

| **Models**              | **SciEval PHY**  | **SciEval CHE**  | **SciEval BIO**  | **SciEval Avg.** | **SciQ**         | **GaoKao MathQA** | **GaoKao CHE**   | **GaoKao BIO**   | **ARC Easy**     | **ARC Challenge** | **ARC Avg.**     | **AQUA-RAT**     |
| :---------------------- | :--------------- | :--------------- | :--------------- | :--------------- | :--------------- | :---------------- | :--------------- | :--------------- | :--------------- | :---------------- | :--------------- | :--------------- |
| Llama-3-8B              | 46.95            | 63.45            | 74.53            | 65.47            | 90.90            | 27.92             | 32.85            | 43.81            | 91.37            | 77.73             | 84.51            | <ins>27.95</ins> |
| DCLM-7B                 | **56.71**        | 64.39            | 72.03            | 66.25            | **92.50**        | 29.06             | 31.40            | 37.14            | 89.52            | 76.37             | 82.94            | 20.08            |
| Mistral-7B-v0.3         | 48.17            | 59.41            | 68.89            | 61.51            | 89.40            | 30.48             | 30.92            | 41.43            | 87.33            | 74.74             | 81.04            | 23.23            |
| Llama-3-Chinese-8B      | 48.17            | 67.34            | 73.90            | <ins>67.34</ins> | 89.20            | 27.64             | 30.43            | 38.57            | 88.22            | 70.48             | 79.35            | 27.56            |
| MAmmoTH2-8B             | 49.39            | **69.36**        | <ins>76.83</ins> | **69.60**        | 90.20            | **32.19**         | <ins>36.23</ins> | <ins>49.05</ins> | **92.85**        | **84.30**         | **88.57**        | 27.17            |
| Galactica-6.7B          | 34.76            | 43.39            | 54.07            | 46.27            | 71.50            | 23.65             | 27.05            | 24.76            | 65.91            | 46.76             | 56.33            | 20.87            |
| **Llama-3-SynE (ours)** | <ins>53.66</ins> | <ins>67.81</ins> | **77.45**        | **69.60**        | <ins>91.20</ins> | <ins>31.05</ins>  | **51.21**        | **69.52**        | <ins>91.58</ins> | <ins>80.97</ins>  | <ins>86.28</ins> | **28.74**        |

> On **scientific evaluation benchmarks** (such as SciEval, GaoKao, and ARC), Llama-3-SynE significantly outperforms the base model, particularly showing remarkable improvement in Chinese scientific benchmarks (for example, a 25.71% improvement in the GaoKao biology subtest).

## Quick Start

Use the transformers backend for inference:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_path = "survivi/Llama-3-SynE"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
)
model.to("cuda:0")
model.eval()
prompt = "Hello world!"
inputs = tokenizer(prompt, return_tensors="pt")
inputs = inputs.to("cuda")
pred = model.generate(
    **inputs,
    max_new_tokens=2048,
    repetition_penalty=1.05,
    temperature=0.5,
    top_k=5,
    top_p=0.85,
    do_sample=True
)
pred = pred[0][len(inputs.input_ids[0]) :]
output = tokenizer.decode(pred, skip_special_tokens=True)
print(output)
```

Use the vLLM backend for inference:

```python
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

model_path = "survivi/Llama-3-SynE"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
sampling_params = SamplingParams(
    max_tokens=2048,
    repetition_penalty=1.05,
    temperature=0.5,
    top_k=5,
    top_p=0.85,
)
llm = LLM(
    model=model_path,
    tensor_parallel_size=1,
    trust_remote_code=True,
)
prompt = "Hello world!"
output = llm.generate(prompt, sampling_params)
output = output[0].outputs[0].text
print(output)
```

## License

This project is built upon Meta's Llama-3 model. The use of Llama-3-SynE model weights must follow the Llama-3 [license agreement](https://github.com/meta-llama/llama3/blob/main/LICENSE). The code in this open-source repository follows the [Apache 2.0](LICENSE) license.

## Citation

If you find our work helpful, please consider citing the following paper:

```
@article{jie2024llama3syne,
  title={Towards Effective and Efficient Continual Pre-training of Large Language Models},
  author={Chen, Jie and Chen, Zhipeng and Wang, Jiapeng and Zhou, Kun and Zhu, Yutao and Jiang, Jinhao and Min, Yingqian and Zhao, Wayne Xin and Dou, Zhicheng and Mao, Jiaxin and others},
  journal={arXiv preprint arXiv:2407.18743},
  year={2024}
}
```
