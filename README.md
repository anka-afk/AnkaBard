# AnkaBard

一个从零开始训练的诗句续写 LLM, 提供 Mamba 架构实现与 Tramsformer 架构实现两个版本。

## Colab

为没有相应运行环境的人提供。

https://colab.research.google.com/drive/1TlLS74zXAvtMH__OEVE2_ixLDG0fXS7Q?usp=sharing

## 本地运行

```bash
git clone https://github.com/anka-afk/AnkaBard.git
cd AnkaBard
pip install -r requirements.txt
cd TransformerVersion # 或者 cd MambaVersion
python run.py
```

## 全流程

#### 阶段 1：准备数据

对训练数据进行预处理，生成符合模型输入格式的数据集。

#### 阶段 2：配置参数

```
--------------------------------------------------
STAGE 2: 训练配置
我们的模型有 1.318372 M 参数
```

#### 阶段 3：预训练

通过让模型阅读大量文本，培养模型的基础语言能力。该过程称为预训练（pretrain）。

在本案例中，我们让模型阅读大量的唐诗，从而培养其写诗的语感。

#### 阶段 4：微调

有了基础语言能力后，使用指令微调（instruction fine-tune）技术可以让模型根据指令生成更加符合要求的文本。

在之前的预训练阶段，并不存在一首诗的概念。因为训练数据是所有诗文的拼接，模型只学会不停地生成符合诗句格式的文本，而不会在生成完一首诗后停下来。

在微调阶段，通过设置训练数据的结构，可以让模型学会按照我们的指令写诗。在这个例子中，我们要求模型根据用户提供的题目生成一首完整的诗。

#### 阶段 5：对齐

对齐（alignment）是大模型之所以强大的重要原因之一，也是 ChatGPT 能够成功的关键因素。在实际应用场景中，有必要让大模型的输出符合人类价值观，避免危险、冒犯、歧视等话题产生。

在本案例中，我们尝试让模型与五言诗这一偏好对齐。

# 饿饿

![AnkaBard](./assets/hungry.png)
