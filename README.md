# CValueAlignment Project

本文提出一个面向中国社会主义核心价值观（Core Socialist Values，CSV）的大语言模型价值观对齐评估框架CValue-Alignment，并基于Qwen-2.5-0.5B和LLaMA-3.2-1B训练的CValue-Evaluator对DeepSeek-R1、GLM-4、以及Claude-3-Opus等国内外8个主流大语言模型进行评估，发现不同LLM在价值观对齐过程中所体现出的不同侧重与倾向性。

---

## 项目结构说明

### `training/`
包含用于模型训练的核心代码（共 8 个 `.py` 文件），如数据预处理、模型定义、训练循环等。

###  `LLM_value_assessment/`
包含用于对大模型输出进行价值观评价的评估实验代码（共 5 个 `.py` 文件）。

###  `LLM_Label/`
包含用于人工标注模型输出的辅助工具（共 11 个 `.py` 文件），支持多轮标注与可视化。

---

## 数据文件说明（位于 `data/` 文件夹中）

| 文件名                        | 内容说明                       |
|-----------------------------|------------------------------|
| `500_question_answer.xlsx` | 原始问题及各模型回答汇总，共500条 |
| `Single_DeepSeek_2000.csv`          | DeepSeek模型标注的2000条数据     |
| `Single_GPT_2000.csv`             | GPT模型标注的2000条数据      |
| `Triple_LLM_2000.csv`        | 多模型联合标注的2000条数据       |
| `Annotation_correction_2000 .csv`           | 两轮人工修正后的2000条最终版本    |
|`Annotation_correction1000_human300.xlsx` | 两轮标注修正的 1000条数据+300条人工修正数据

---

##  注意事项

- 所有代码中的 `api_key` 已替换为 `"YOUR_API_KEY"`，请根据使用需求自行添加。
- 标注数据仅供研究用途，若公开发布请注意数据合规。