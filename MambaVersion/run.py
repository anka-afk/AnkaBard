import torch

from dataset import Dataset
from evaluator import Evaluator
from model import TutorialLLM
from trainer import Trainer


print(f'{"-"*50}\nSTAGE 1: 准备数据')
# 要处理的并行项目数量，称为批次大小
batch_size = 16
# 要处理的文本最大长度
max_length = 256
# 如果可用，在 GPU(cuda) 上运行模型
device = "cuda" if torch.cuda.is_available() else "cpu"
# 设置随机种子以实现可重现性
torch.manual_seed(2024)
dataset = Dataset("data.json", batch_size, max_length, device)
print("检查一批预训练数据：")
print(dataset.get_batch_pretrain("train"))
print("检查一批微调数据：")
print(next(dataset.get_batch_generator_finetune("train")))
print("检查一批对齐数据：")
print(next(dataset.get_batch_generator_alignment("train")))

print(f'{"-"*50}\nSTAGE 2: 训练配置')
# Transformer 中嵌入向量的维度
dim_embedding = 64
# Mamba 中的层数
num_layer = 4
# Mamba SSM 的状态维度
d_state = 16
# Mamba SSM 卷积核的大小
d_conv = 4
# Mamba SSM 内部维度扩展因子
expand = 2
# 创建一个 TutorialLLM 实例
model = TutorialLLM(
    dataset.vocabulary_size,
    dim_embedding,
    max_length,
    num_layer,
    d_state,
    d_conv,
    expand,
    device,
)
# 将模型切换到训练模式并将数据移动到指定设备
model.train()
model.to(device)
# 显示模型大小
print(
    f"我们的模型有 {sum(parameter.numel() for parameter in model.parameters())/1e6} M 参数"
)
# 评估预训练过程的迭代次数（每次迭代处理一个批次）
iterations_to_evaluate_pretrain = 100
# 评估预训练过程的迭代间隔
interval_to_evaluate_pretrain = 100
# 评估微调过程的迭代间隔
interval_to_evaluate_finetune = 50
# 评估对齐过程的迭代间隔
interval_to_evaluate_alignment = 50
# 创建一个 Evaluator 实例以评估训练期间的性能
evaluator = Evaluator(
    dataset,
    device,
    iterations_to_evaluate_pretrain,
    interval_to_evaluate_pretrain,
    interval_to_evaluate_finetune,
    interval_to_evaluate_alignment,
)
# 创建一个 Trainer 实例用于后续训练
trainer = Trainer(model, dataset, evaluator, device)

print(f'{"-"*50}\nSTAGE 3: 预训练')
print("在此阶段，模型将学习如何写诗的基本知识。\n")
# 预训练的迭代次数（每次迭代处理一个批次）
iterations_for_pretrain = 20000
# 预训练模型
trainer.pretrain(iterations_for_pretrain)

print(f'{"-"*50}\nSTAGE 4: 微调')
print("在此阶段，模型将学习根据指令生成诗歌。")
print("在我们的例子中，我们要求模型生成一首给定标题的诗。\n")
# 微调模型的 epoch 数
epochs_for_finetune = 5
# 微调模型
trainer.finetune(epochs_for_finetune)

print(f'{"-"*50}\nSTAGE 5: 对齐偏好')
print("在此阶段，模型将学习生成我们偏好的诗歌。")
print("在我们的例子中，我们更喜欢五言诗而不是其他诗。\n")
# 将模型与我们的偏好对齐的 epoch 数
epochs_for_alignment = 3
# 将模型与我们的偏好对齐
trainer.align(epochs_for_alignment)
