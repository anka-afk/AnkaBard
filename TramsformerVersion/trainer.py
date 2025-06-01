import torch

from dataset import Dataset
from evaluator import Evaluator
from model import DpoWrapper, TutorialLLM


class Trainer:
    """
    模型训练器。

    此模块提供预训练、微调和对齐模型的方法。
    """

    def __init__(
        self, model: TutorialLLM, dataset: Dataset, evaluator: Evaluator, device: str
    ) -> None:
        """
        使用模型、数据集、评估器和设备初始化训练器。

        参数：
            model：要训练的模型。
            dataset：提供训练数据的数据集。
            evaluator：评估模型性能的评估器。
            device：运行模型的设备（'cpu' 或 'cuda'）。
        """
        self.model = model
        self.dataset = dataset
        self.evaluator = evaluator
        self.device = device

    def pretrain(self, iterations: int) -> None:
        """
        预训练模型一定数量的迭代。

        对于每次迭代，使用一批预训练数据来训练模型。

        参数：
            iterations：预训练模型的迭代次数。
        """
        # 重置评估器以清除损失历史记录
        self.evaluator.reset()
        # 使用学习率 1e-3 初始化优化器
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)

        for i in range(iterations):
            # 获取一批预训练数据
            inputs, labels = self.dataset.get_batch_pretrain("train")
            # 前向传播并计算损失
            _, loss = self.model(inputs, labels)

            # 评估模型性能
            self.evaluator.evaluate_pretrain(self.model, i, loss.item())

            # 反向传播并更新模型
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        print("保存预训练模型...")
        torch.save(self.model, "model_pretrain.pth")

    def finetune(self, epochs) -> None:
        """
        微调模型一定数量的 epoch。

        对于每个 epoch，使用一批微调数据来训练模型。

        参数：
            epochs：微调模型的 epoch 数。
        """
        # 使用学习率 1e-3 初始化优化器
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)

        for epoch in range(epochs):
            # 重置评估器以清除每个 epoch 的损失历史记录
            self.evaluator.reset()
            for i, (inputs, labels) in enumerate(
                self.dataset.get_batch_generator_finetune("train")
            ):
                # 前向传播并计算损失
                _, loss = self.model(inputs, labels)

                # 评估模型性能
                self.evaluator.evaluate_finetune(self.model, epoch, i, loss.item())

                # 反向传播并更新模型
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        print("保存微调模型...")
        torch.save(self.model, "model_finetune.pth")

    def align(self, epochs) -> None:
        """
        将模型与我们的偏好对齐一定数量的 epoch。

        对于每个 epoch，使用一批对齐数据来训练模型。

        参数：
            epochs：将模型与我们的偏好对齐的 epoch 数。
        """
        # 对齐需要 DPO 的参考模型，我们使用 DpoWrapper 来管理这两个模型
        dpo_wrapper = DpoWrapper(self.model)
        # 使用学习率 1e-5 初始化优化器
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-5)

        for epoch in range(epochs):
            # 重置评估器以清除每个 epoch 的损失历史记录
            self.evaluator.reset()
            for i, (
                positive_inputs,
                positive_labels,
                negative_inputs,
                negative_labels,
            ) in enumerate(self.dataset.get_batch_generator_alignment("train")):
                loss, reward_margin = dpo_wrapper.forward(
                    positive_inputs, positive_labels, negative_inputs, negative_labels
                )

                # 每隔 evaluation_interval 迭代评估模型
                self.evaluator.evaluate_alignment(
                    dpo_wrapper, epoch, i, loss.item(), reward_margin.item()
                )

                # 反向传播并更新模型
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        print("保存对齐模型...")
        torch.save(self.model, "model_aligned.pth")
