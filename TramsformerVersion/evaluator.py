import torch

from dataset import Dataset
from model import DpoWrapper, TutorialLLM


class Evaluator:
    """
    模型评估器。

    此模块提供在训练期间评估模型性能的方法。
    """

    def __init__(
        self,
        dataset: Dataset,
        device: str,
        iterations_to_evaluate_pretrain: int,
        interval_to_evaluate_pretrain: int,
        interval_to_evaluate_finetune: int,
        interval_to_evaluate_alignment: int,
    ) -> None:
        """
        使用数据集、设备和评估间隔初始化评估器。

        参数：
            dataset：提供评估数据的数据集。
            device：运行模型的设备（'cpu' 或 'cuda'）。
            iterations_to_evaluate_pretrain：评估预训练过程的迭代次数。
            interval_to_evaluate_pretrain：评估预训练过程的迭代间隔。
            interval_to_evaluate_finetune：评估微调过程的迭代间隔。
            interval_to_evaluate_alignment：评估对齐过程的迭代间隔。
        """
        self.dataset = dataset
        self.device = device
        self.iterations_to_evaluate_pretrain = iterations_to_evaluate_pretrain
        self.interval_to_evaluate_pretrain = interval_to_evaluate_pretrain
        self.interval_to_evaluate_finetune = interval_to_evaluate_finetune
        self.interval_to_evaluate_alignment = interval_to_evaluate_alignment

        self.test_input = "<INS>請用以下題目寫一首詩<INP>春夜喜雨<RES>"

        self.reset()

    def reset(self) -> None:
        """
        重置损失和奖励边际累加器。
        """
        self.train_loss_sum = 0
        self.train_reward_margin_sum = 0

    @torch.inference_mode()
    def evaluate_pretrain(
        self, model: TutorialLLM, iteration: int, train_loss: float
    ) -> None:
        """
        在预训练过程中评估模型性能。

        此方法应在训练期间的每次迭代中调用。
        训练损失和评估损失将每 `interval_to_evaluate_pretrain` 次迭代打印一次。
        将生成一首以“春夜喜雨”为标题的诗，以查看模型的表现。

        参数：
            model：要评估的模型。
            iteration：当前迭代次数。
            train_loss：当前迭代的训练损失。
        """
        if iteration % self.interval_to_evaluate_pretrain == 0:
            # 获取平均训练损失和评估损失
            mean_loss_train = self.train_loss_sum / self.interval_to_evaluate_pretrain
            self.reset()
            evaluate_loss = self.evaluate_pretrain_loss(
                model, self.iterations_to_evaluate_pretrain
            )
            print(
                f"Step {iteration}, train loss {mean_loss_train:.4f}, evaluate loss {evaluate_loss:.4f}"
            )

            # 让我们生成一首以“春夜喜雨”为标题的诗，看看模型的表现
            test_tokens = torch.tensor(
                self.dataset.encode("春夜喜雨"), dtype=torch.long, device=self.device
            ).unsqueeze(0)
            print("Generate first 100 characters of poems starting with 春夜喜雨:")
            print(
                self.dataset.decode(
                    model.generate(test_tokens, max_new_tokens=100)[0].tolist()
                )
            )

        # 累加训练损失
        self.train_loss_sum += train_loss

    @torch.inference_mode()
    def evaluate_pretrain_loss(self, model: TutorialLLM, iterations: int) -> float:
        """
        在预训练过程中评估模型损失。

        参数：
            model：要评估的模型。
            iterations：评估模型的迭代次数。

        返回：
            评估中模型的平均损失。
        """
        losses = torch.zeros(iterations)
        # 评估模型 `iterations` 次
        for k in range(iterations):
            # 获取一批预训练数据并计算损失
            inputs, labels = self.dataset.get_batch_pretrain("evaluate")
            _, loss = model(inputs, labels)
            losses[k] = loss.item()
        loss = losses.mean()
        return loss

    @torch.inference_mode()
    def evaluate_finetune(
        self, model: TutorialLLM, epoch: int, iteration: int, train_loss: float
    ) -> None:
        """
        在微调过程中评估模型性能。

        此方法应在训练期间的每次迭代中调用。
        训练损失和评估损失将每 `interval_to_evaluate_finetune` 次迭代打印一次。
        将生成一首以“春夜喜雨”为标题的诗，以查看模型的表现。

        参数：
            model：要评估的模型。
            epoch：当前 epoch 数。
            iteration：当前迭代次数。
            train_loss：当前迭代的训练损失。
        """
        if iteration % self.interval_to_evaluate_finetune == 0:
            # 获取平均训练损失和评估损失
            mean_loss_train = self.train_loss_sum / self.interval_to_evaluate_finetune
            self.reset()
            evaluate_loss = self.evaluate_finetune_loss(model)
            print(
                f"Epoch {epoch}, step {iteration}, train loss {mean_loss_train:.4f}, evaluate loss {evaluate_loss:.4f}"
            )

            # 让我们生成一首给定标题的诗，看看模型的表现
            test_tokens = torch.tensor(
                self.dataset.encode(self.test_input),
                dtype=torch.long,
                device=self.device,
            ).unsqueeze(0)
            output = self.dataset.decode(
                model.generate(test_tokens, max_new_tokens=100)[0].tolist()
            )
            # 将输出截断到文本结束字符 '\0'
            output = output[: output.find("\0")]
            print("Generate a complete poem for title 春夜喜雨:")
            print(output[len(self.test_input) :])

        # 累加训练损失
        self.train_loss_sum += train_loss

    @torch.inference_mode()
    def evaluate_finetune_loss(self, model: TutorialLLM) -> float:
        """
        在微调过程中评估模型损失。

        参数：
            model：要评估的模型。

        返回：
            评估中模型的平均损失。
        """
        loss_sum = 0
        # 获取微调数据的批生成器
        batch_generator = self.dataset.get_batch_generator_finetune("evaluate")
        # 通过处理生成器生成的所有批次来评估模型
        for k, batch in enumerate(batch_generator):
            inputs, labels = batch
            _, loss = model(inputs, labels)
            loss_sum += loss.item()
        loss = loss_sum / (k + 1)
        return loss

    @torch.inference_mode()
    def evaluate_alignment(
        self,
        dpo_wrapper: DpoWrapper,
        epoch: int,
        iteration: int,
        train_loss: float,
        train_reward_margin: float,
    ) -> None:
        """
        在对齐过程中评估模型性能。

        此方法应在训练期间的每次迭代中调用。
        训练损失和评估损失将每 `interval_to_evaluate_alignment` 次迭代打印一次。
        将打印对齐模型和参考模型生成的诗歌，以比较两个模型。

        参数：
            dpo_wrapper：要评估的 DPO 封装器。
            epoch：当前 epoch 数。
            iteration：当前迭代次数。
            train_loss：当前迭代的训练损失。
            train_reward_margin：当前迭代的训练奖励边际。
        """
        if iteration % self.interval_to_evaluate_alignment == 0:
            # 计算此间隔的平均损失
            mean_loss_train = self.train_loss_sum / self.interval_to_evaluate_alignment
            mean_reward_margin_train = (
                self.train_reward_margin_sum / self.interval_to_evaluate_alignment
            )
            self.reset()
            evaluate_loss, evaluate_reward_margin = self.evaluate_alignment_loss(
                dpo_wrapper
            )
            print(
                f"Epoch {epoch}, step {iteration}, train loss {mean_loss_train:.4f}, evaluate loss {evaluate_loss:.4f}, train reward margin {mean_reward_margin_train:.4f}, evaluate reward margin {evaluate_reward_margin:.4f}"
            )

            # 让我们分别要求两个模型生成一首诗
            test_tokens = torch.tensor(
                self.dataset.encode(self.test_input),
                dtype=torch.long,
                device=self.device,
            ).unsqueeze(0)
            aligned_output = self.dataset.decode(
                dpo_wrapper.aligned_model.generate(test_tokens, max_new_tokens=100)[
                    0
                ].tolist()
            )
            reference_output = self.dataset.decode(
                dpo_wrapper.reference_model.generate(test_tokens, max_new_tokens=100)[
                    0
                ].tolist()
            )
            # 将输出截断到文本结束字符 '\0'
            aligned_output = aligned_output[: aligned_output.find("\0")]
            reference_output = reference_output[: reference_output.find("\0")]
            print("Generate a complete poem for title 春夜喜雨:")
            print("Aligned model:")
            print(aligned_output[len(self.test_input) :])
            print("Reference model:")
            print(reference_output[len(self.test_input) :])

        # 累加训练损失和奖励边际
        self.train_loss_sum += train_loss
        self.train_reward_margin_sum += train_reward_margin

    @torch.inference_mode()
    def evaluate_alignment_loss(self, dpo_wrapper: DpoWrapper) -> tuple[float, float]:
        """
        在对齐过程中评估模型损失。

        参数：
            dpo_wrapper：要评估的 DPO 封装器。

        返回：
            评估中模型的平均损失和奖励边际。
        """
        loss_sum = 0
        reward_margin_sum = 0
        # 获取对齐数据的批生成器
        batch_generator = self.dataset.get_batch_generator_alignment("evaluate")
        # 通过处理生成器生成的所有批次来评估模型
        for k, (
            positive_inputs,
            positive_labels,
            negative_inputs,
            negative_labels,
        ) in enumerate(batch_generator):
            loss, reward_margin = dpo_wrapper.forward(
                positive_inputs, positive_labels, negative_inputs, negative_labels
            )
            loss_sum += loss.item()
            reward_margin_sum += reward_margin.item()
        loss = loss_sum / (k + 1)
        reward_margin = reward_margin_sum / (k + 1)
        return loss, reward_margin
