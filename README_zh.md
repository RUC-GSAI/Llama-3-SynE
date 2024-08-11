# Llama-3-SynE

<p align="center">
 📄<a href="https://arxiv.org/abs/2407.18743" target="_blank"> 报告 </a> • 🤗 <a href="https://huggingface.co/survivi/Llama-3-SynE" target="_blank">Hugging Face 仓库</a>
</p>

<p align="center">
 🔍<a href="README.md" target="_blank">English</a>
</p>

## 更新
- ✨✨ ``2024/08/10``: 我们发布了 [Llama-3-SynE 模型](https://huggingface.co/survivi/Llama-3-SynE)。
- ✨ ``2024/07/26``: 我们发布了 Llama-3-SynE 的 [技术报告](https://arxiv.org/abs/2407.18743)，欢迎查阅！

## 模型介绍

**Llama-3-SynE**（**Syn**thetic data **E**nhanced Llama-3）是 [Llama-3（8B）](https://github.com/meta-llama/llama3)的增强版，通过继续预训练（continual pre-training，CPT）来提升其**中文语言能力和科学推理能力**。通过精心设计的数据混合和课程策略，Llama-3-SynE 成功地在保持原始模型性能的同时增强了新能力。这个增强过程包括利用现有数据集并合成专门为目标任务设计的高质量数据集。

Llama-3-SynE 的主要特点包括：
- **增强的中文语言能力**：通过基于主题的数据混合和基于困惑度的数据课程实现。
- **改进的科学推理能力**：利用合成数据集来增强多学科的科学知识。
- **高效的继续预训练**：只消耗约 1000 亿个 token，成本效益高。

## 模型列表

| 模型            | 类型  | 序列长度 | 下载                                                                                                                      |
|-----------------|-------|----------|---------------------------------------------------------------------------------------------------------------------------|
| Llama-3-SynE    | Base  | 8K       | [🤗 Huggingface](https://huggingface.co/survivi/Llama-3-SynE) |

## 基准测试

我们将所有评估基准分为两组。第一组是 _主要基准_，旨在评估大语言模型的综合能力。值得注意的是我们在这一组基准中包括了常用的数学和代码基准，因为使用这些基准评估各种通用大语言模型是标准做法。

第二组是 _科学基准_，涵盖了多学科的科学知识。

我们报告了在 GSM8K、ASDiv 和 MAWPS 上的 8-shot 性能，C-Eval、CMMLU、MMLU、MATH、GaoKao、SciQ、SciEval、SAT-Math 和 AQUA-RAT 上的 5-shot 推理性能，MBPP 上的 3-shot 性能。
对于 HumanEval 和 ARC，我们报告了 0-shot 性能。最佳和次佳结果分别以 **粗体** 和 _斜体_ 标出。

### 主要基准

|       **模型**         | **MMLU**      | **C-Eval** | **CMMLU** | **MATH**      | **GSM8K** | **ASDiv** | **MAWPS** | **SAT-Math** | **HumanEval** | **MBPP** |
|---------------------------|---------------|----------|---------|---------------|---------|---------|---------|-----------|----------------|--------|
| Llama-3-8B            | **66.60**     | 49.43    | 51.03   | 16.20         | 54.40   | 72.10   | 89.30   | 38.64     | _36.59_        | **47.00** |
| DCLM-7B               | 64.01         | 41.24    | 40.89   | 14.10         | 39.20   | 67.10   | 83.40   | _41.36_   | 21.95          | 32.60  |
| Mistral-7B-v0.3       | 63.54         | 42.74    | 43.72   | 12.30         | 40.50   | 67.50   | 87.50   | 40.45     | 25.61          | 36.00  |
| Llama-3-Chinese-8B    | 64.10         | _50.14_  | _51.20_ | 3.60          | 0.80    | 1.90    | 0.60    | 36.82     | 9.76           | 14.80  |
| MAmmoTH2-8B           | 64.89         | 46.56    | 45.90   | **34.10**     | **61.70**| **82.80**| _91.50_ | _41.36_   | 17.68          | 38.80  |
| Galactica-6.7B        | 37.13         | 26.72    | 25.53   | 5.30          | 9.60    | 40.90   | 51.70   | 23.18     | 7.31           | 2.00   |
| **Llama-3-SynE (ours)**   | _65.19_       | **58.24**| **57.34**| _28.20_      | _60.80_ | _81.00_ | **94.10**| **43.64**| **42.07**      | _45.60_|

> 在 **中文评估基准**（如 C-Eval 和 CMMLU）上，Llama-3-SynE 显著优于基础模型 Llama-3（8B），表明我们的方法在提升中文语言能力方面非常有效。

> 在 **英文评估基准**（如 MMLU、MATH 和代码评估基准）上，Llama-3-SynE 展现出与基础模型相当或更好的性能，表明我们的方法在继续预训练过程中有效解决了灾难性遗忘问题。

### 科学基准

“PHY”、“CHE” 和 “BIO” 分别表示对应基准的物理、化学和生物子任务。

| **模型**         | **SciEval PHY** | **SciEval CHE** | **SciEval BIO** | **SciEval Avg.** | **SciQ** | **GaoKao MathQA** | **GaoKao CHE** | **GaoKao BIO** | **ARC Easy** | **ARC Challenge** | **ARC Avg.** | **AQUA-RAT** |
|--------------------|-----------------|-----------------|-----------------|------------------|---------------|-------------------|----------------|----------------|---------------|-------------------|--------------|-------------------|
| Llama-3-8B         | 46.95           | 63.45           | 74.53           | 65.47            | 90.90         | 27.92             | 32.85          | 43.81          | 91.37         | 77.73             | 84.51        | _27.95_           |
| DCLM-7B            | **56.71**       | 64.39           | 72.03           | 66.25            | **92.50**     | 29.06             | 31.40          | 37.14          | 89.52         | 76.37             | 82.94        | 20.08             |
| Mistral-7B-v0.3    | 48.17           | 59.41           | 68.89           | 61.51            | 89.40         | 30.48             | 30.92          | 41.43          | 87.33         | 74.74             | 81.04        | 23.23             |
| Llama-3-Chinese-8B | 48.17           | 67.34           | 73.90           | _67.34_          | 89.20         | 27.64             | 30.43          | 38.57          | 88.22         | 70.48             | 79.35        | 27.56             |
| MAmmoTH2-8B        | 49.39           | **69.36**       | _76.83_         | **69.60**        | 90.20         | **32.19**         | _36.23_        | _49.05_        | **92.85**     | **84.30**         | **88.57**    | 27.17             |
| Galactica-6.7B     | 34.76           | 43.39           | 54.07           | 46.27            | 71.50         | 23.65             | 27.05          | 24.76          | 65.91         | 46.76             | 56.33        | 20.87             |
| **Llama-3-SynE (ours)** | _53.66_   | _67.81_         | **77.45**       | **69.60**        | _91.20_       | _31.05_           | **51.21**      | **69.52**      | _91.58_       | _80.97_           | _86.28_      | **28.74**         |

> 在 **科学评估基准**（如 SciEval、GaoKao 和 ARC）上，Llama-3-SynE 显著优于基础模型，特别是在中文科学基准上表现出显著提升（例如，高考生物子测试中提升了 25.71%）。

## 快速开始

基于 transformers 进行推理：

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

基于 vLLM 进行推理：

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

## 许可证

本项目基于 Meta 的 Llama-3 模型开发，Llama-3-SynE 模型权重的使用必须遵循 Llama-3 [许可协议](https://github.com/meta-llama/llama3/blob/main/LICENSE)。此开源代码库中的代码遵循 [Apache 2.0](LICENSE) 许可证。

## 引用

如果您觉得我们的工作对您有帮助，请考虑引用以下论文：

```
@article{jie2024llama3syne,
  title={Towards Effective and Efficient Continual Pre-training of Large Language Models},
  author={Chen, Jie and Chen, Zhipeng and Wang, Jiapeng and Zhou, Kun and Zhu, Yutao and Jiang, Jinhao and Min, Yingqian and Zhao, Wayne Xin and Dou, Zhicheng and Mao, Jiaxin and others},
  journal={arXiv preprint arXiv:2407.18743},
  year={2024}
}
```
