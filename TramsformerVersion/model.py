import copy
from typing import Optional
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import Tensor


class AttentionHead(nn.Module):
    """
    单头自注意力。

    此模块为一批序列计算自注意力，
    其中每个序列的最大长度为 `max_length` 个标记，
    每个标记是维度为 `dim_embed` 的向量。
    注意力机制参考了著名的 Transformer 论文《Attention Is All You Need》。
    """

    def __init__(self, dim_embed: int, head_size: int, max_length: int) -> None:
        """
        使用 3 个线性层和一个掩码缓冲区初始化模块。

        参数：
            dim_embed：输入序列中每个标记向量的维度。
            head_size：输出向量的维度。
            max_length：任何标记序列的最大长度。也称为最大上下文长度。
        """
        super().__init__()
        # 创建线性层将输入张量投影到键张量、查询张量和值张量。
        # 这 3 个层执行相同的转换，但不共享权重。
        # 训练后，这些层将学习输入向量的不同方面。
        self.project_to_key = nn.Linear(dim_embed, head_size, bias=False)
        self.project_to_query = nn.Linear(dim_embed, head_size, bias=False)
        self.project_to_value = nn.Linear(dim_embed, head_size, bias=False)
        # 创建一个矩阵缓冲区，将方阵掩码为下三角矩阵。
        # 这用于为自注意力机制添加因果约束，
        # 这意味着每个标记只能看到之前的标记，而不能看到未来的标记。
        self.register_buffer("tril", torch.tril(torch.ones(max_length, max_length)))

    def forward(self, input: Tensor) -> Tensor:
        """
        计算输入张量的自注意力。

        参数：
            input：形状为 (B, T, `dim_embed`) 的张量，其中 B 是批大小，
                T 是标记序列长度，`dim_embed` 是每个标记向量的维度。

        返回：
            形状为 (B, T, `head_size`) 的张量。输入张量中的每个向量都被转换为维度为 `head_size` 的新向量，
            该向量捕获自注意力。
        """
        B, T, dim_embed = input.shape
        # 将输入张量投影到键张量、查询张量和值张量
        key = self.project_to_key(input)  # (B, T, dim_embed) -> (B, T, head_size)
        query = self.project_to_query(input)  # (B, T, dim_embed) -> (B, T, head_size)
        value = self.project_to_value(input)  # (B, T, dim_embed) -> (B, T, head_size)
        # 计算自注意力权重
        weights = query @ key.transpose(
            -2, -1
        )  # (B, T, head_size) @ (B, head_size, T) -> (B, T, T)
        # 缩放注意力权重
        weights *= dim_embed**-0.5
        # 掩码注意力权重以遵守因果约束
        # 裁剪 tril 矩阵以适应当前输入的大小
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        # 将注意力权重转换为概率
        weights = F.softmax(weights, dim=-1)
        # 将注意力应用于值
        output = weights @ value  # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)
        return output


class MultiHeadAttention(nn.Module):
    """
    多头自注意力。

    多头自注意力通过聚合多个 `AttentionHead` 模块的输出来组成。
    """

    def __init__(
        self, dim_embed: int, num_heads: int, head_size: int, max_length: int
    ) -> None:
        """
        使用连接的 `AttentionHead` 和一个投影层初始化模块。

        参数：
            dim_embed：输入张量中每个标记向量的维度。
            num_heads：多头注意力中包含的头数。
            head_size：每个头的输出向量的维度。
            max_length：任何标记序列的最大长度。也称为最大上下文长度。
        """
        super().__init__()
        # 创建 `num_heads` 个注意力头的列表
        self.heads = nn.ModuleList(
            [AttentionHead(dim_embed, head_size, max_length) for _ in range(num_heads)]
        )
        # 创建一个线性层，将所有头的连接输出投影到原始维度。
        # 在我们的例子中，连接输出恰好与原始维度相同，因此我们可以跳过此投影层。
        # 但通常，头的输出可能与输入具有不同的维度。
        self.project = nn.Linear(head_size * num_heads, dim_embed)

    def forward(self, input: Tensor) -> Tensor:
        """
        计算输入张量的多头自注意力。

        参数：
            input：形状为 (B, T, `dim_embed`) 的张量，其中 B 是批大小，
                T 是标记序列长度，`dim_embed` 是每个标记向量的维度。

        返回：
            形状为 (B, T, `dim_embed`) 的张量。输入张量中的每个向量都被转换为相同维度的新向量，
            该向量捕获多头自注意力。
        """
        # 将输入张量发送到每个注意力头并连接输出
        output = torch.cat(
            [head(input) for head in self.heads], dim=-1
        )  # (B, T, dim_embed) -> [(B, T, head_size)] * num_heads -> (B, T, head_size * num_heads)
        # 将连接输出投影到原始维度
        output = self.project(
            output
        )  # (B, T, head_size * num_heads) -> (B, T, dim_embed)
        return output


class FeedForward(nn.Module):
    """
    前馈神经网络。

    此模块是一个简单的前馈神经网络，包含 2 个线性层和一个 ReLU 激活函数。
    它将输入维度放大 4 倍，然后缩小回原始维度，以学习输入张量中更复杂的模式。
    """

    def __init__(self, dim_embed: int) -> None:
        """
        使用 2 个线性层和一个 ReLU 激活函数初始化模块。

        参数：
            dim_embed：输入张量中每个标记向量的维度。
        """
        super().__init__()
        # 创建一个包含 2 个线性层和一个 ReLU 激活函数的顺序模块。
        # 第一层将输入维度放大 4 倍。然后将 ReLU 激活函数应用于输出。
        # 最后，第二层将维度缩小回原始大小。
        self.feed_forward = nn.Sequential(
            nn.Linear(dim_embed, 4 * dim_embed),
            nn.ReLU(),
            nn.Linear(4 * dim_embed, dim_embed),
        )

    def forward(self, input: Tensor) -> Tensor:
        """
        计算输入张量的前馈神经网络输出。

        参数：
            input：形状为 (B, T, `dim_embed`) 的张量，其中 B 是批大小，
                T 是标记序列长度，`dim_embed` 是每个标记向量的维度。
        """
        return self.feed_forward(
            input
        )  # (B, T, dim_embed) -> (B, T, 4 * dim_embed) -> (B, T, dim_embed)


class TranformerBlock(nn.Module):
    """
    Transformer 块。

    此模块是一个单独的 Transformer 块，由多头自注意力和前馈神经网络组成。
    每个子模块之前都应用层归一化以稳定训练过程。
    """

    def __init__(self, dim_embed: int, num_heads: int, max_length: int) -> None:
        """
        使用多头自注意力、前馈神经网络和 2 个层归一化层初始化模块。

        参数：
            dim_embed：输入张量中每个标记向量的维度。
            num_heads：多头注意力中包含的头数。
            max_length：任何标记序列的最大长度。也称为最大上下文长度。
        """
        super().__init__()
        # 为简单起见，我们将 `head_size` 选择为 `dim_embed` 的一个因子。
        head_size = dim_embed // num_heads
        # 创建多头自注意力模块
        self.multi_head_attention = MultiHeadAttention(
            dim_embed, num_heads, head_size, max_length
        )
        # 创建前馈神经网络模块
        self.feed_forward = FeedForward(dim_embed)
        # 创建 2 个层归一化层
        self.layer_norm1 = nn.LayerNorm(dim_embed)
        self.layer_norm2 = nn.LayerNorm(dim_embed)

    def forward(self, input: Tensor) -> Tensor:
        """
        计算输入张量的 Transformer 块输出。

        我们将注意力头和前馈神经网络视为残差流。

        参数：
            input：形状为 (B, T, `dim_embed`) 的张量，其中 B 是批大小，
                T 是标记序列长度，`dim_embed` 是每个标记向量的维度。

        返回：
            形状为 (B, T, `dim_embed`) 的张量。输入张量中的每个向量都被转换为相同维度的新向量，
            该向量捕获 Transformer 机制。
        """
        # 应用多头自注意力并作为残差流添加到输入张量
        output = input + self.multi_head_attention(
            self.layer_norm1(input)
        )  # (B, T, dim_embed) + (B, T, dim_embed) -> (B, T, dim_embed)
        # 应用前馈神经网络并作为残差流添加到输出张量
        output = output + self.feed_forward(
            self.layer_norm2(output)
        )  # (B, T, dim_embed) + (B, T, dim_embed) -> (B, T, dim_embed)
        return output


class TutorialLLM(nn.Module):
    """
    教程大型语言模型。

    这是一个基于 Transformer 架构构建的非常简单的语言模型。
    它类似于 GPT-2 模型，但仅用于教育目的。
    """

    def __init__(
        self,
        vocabulary_size: int,
        dim_embed: int,
        max_length: int,
        num_head: int,
        num_layer: int,
        device: str,
    ) -> None:
        """
        使用标记嵌入表、位置嵌入表、多个 Transformer 块、一个最终层归一化层和一个线性层初始化模型。

        参数：
            vocabulary_size：词汇表中唯一标记的数量。
            dim_embed：模型中嵌入向量的维度。
            max_length：要处理的文本的最大长度。也称为最大上下文长度。
            num_head：多头注意力中的头数。
            num_layer：模型中的 Transformer 块数。
            device：运行模型的设备，可以是 'cpu' 或 'cuda'。
        """
        super().__init__()
        self.max_length = max_length
        self.device = device
        # 创建标记嵌入表将标记 ID 转换为向量
        self.token_embedding_table = nn.Embedding(vocabulary_size, dim_embed)
        # 创建位置嵌入表为标记向量添加位置信息
        self.position_embedding_table = nn.Embedding(max_length, dim_embed)
        # 创建一系列 Transformer 块
        self.transformer_blocks = nn.Sequential(
            *[
                TranformerBlock(dim_embed, num_head, max_length)
                for _ in range(num_layer)
            ]
        )
        # 为最终输出创建层归一化层
        self.layer_norm_final = nn.LayerNorm(dim_embed)
        # 创建一个线性层将输出从嵌入空间投影到词汇空间
        self.project = nn.Linear(dim_embed, vocabulary_size)

    def forward(
        self, token_ids: Tensor, labels: Tensor = None, reduce_loss: bool = True
    ) -> tuple[Tensor, Optional[Tensor]]:
        """
        计算模型的正向传播。

        参数：
            token_ids：形状为 (B, T) 的张量，其中 B 是批大小，T 是标记序列长度。
                该张量包含输入序列的标记 ID。
            labels：形状为 (B, T) 的张量，其中 B 是批大小，T 是标记序列长度。
                该张量包含目标序列的真实标记 ID。如果为 None，模型将不计算损失。

        返回：
            模型的 logits 和损失（如果提供了标签）。
        """
        B, T = token_ids.shape
        # 获取标记嵌入和位置嵌入
        token_embedding = self.token_embedding_table(
            token_ids
        )  # (B, T) -> (B, T, dim_embed)
        position_embedding = self.position_embedding_table(
            torch.arange(T, device=self.device)
        )  # (T) -> (T, dim_embed)
        # 在最后一个维度添加标记嵌入和位置嵌入
        embedding = (
            token_embedding + position_embedding
        )  # (B, T, dim_embed) + (T, dim_embed) -> (B, T, dim_embed)
        # 将嵌入通过 Transformer 块
        embedding = self.transformer_blocks(
            embedding
        )  # (B, T, dim_embed) -> (B, T, dim_embed)
        # 对最终输出应用层归一化
        embedding = self.layer_norm_final(
            embedding
        )  # (B, T, dim_embed) -> (B, T, dim_embed)
        # 将输出投影到词汇空间
        logits = self.project(embedding)  # (B, T, dim_embed) -> (B, T, vocabulary_size)

        if labels is None:
            loss = None
        else:
            B, T, vocabulary_size = logits.shape
            # 将 logits 展平为词汇空间中的向量列表
            logits = logits.view(B * T, vocabulary_size)
            # 将标签展平为标记 ID 列表
            labels = labels.view(B * T)
            # 计算 logits 和标签之间的交叉熵损失
            loss = F.cross_entropy(logits, labels, reduce=reduce_loss)

        return logits, loss

    def generate(self, token_ids: Tensor, max_new_tokens: int) -> Tensor:
        """
        根据输入标记生成后续标记。

        参数：
            token_ids：形状为 (B, T) 的张量，其中 B 是批大小，T 是标记序列长度。
                该张量包含输入序列的标记 ID。
            max_new_tokens：要生成的新标记的最大数量。

        返回：
            生成序列的标记 ID 张量。
        """
        for _ in range(max_new_tokens):
            # 如果输入序列超过最大长度，则裁剪
            token_ids_available = token_ids[
                :, -self.max_length :
            ]  # (B, T) -> (B, T'), where T' = min(T, max_length)
            # 运行模型以获取 logits
            logits, _ = self(token_ids_available)  # (B, T') -> (B, T', vocabulary_size)
            # 选择最后一个标记的 logits，其中应预测下一个标记
            logits = logits[
                :, -1, :
            ]  # (B, T', vocabulary_size) -> (B, vocabulary_size)
            # 应用 softmax 获取概率
            probs = F.softmax(
                logits, dim=-1
            )  # (B, vocabulary_size) -> (B, vocabulary_size)
            # 从概率分布中采样下一个标记
            idx_next = torch.multinomial(
                probs, num_samples=1
            )  # (B, vocabulary_size) -> (B, 1)
            # 将下一个标记附加到输入序列以进行下一次迭代
            token_ids = torch.cat(
                (token_ids, idx_next), dim=1
            )  # (B, T) + (B, 1) -> (B, T+1)
            # 如果下一个标记是 ID 为 0 的序列结束标记，则停止
            if idx_next.item() == 0:
                break
        return token_ids


class DpoWrapper:
    """
    直接偏好优化封装器。

    此模块封装了对齐模型和参考模型以计算 DPO 损失。
    请注意，此类别不是 `nn.Module` 的子类，因此您不能直接调用它。
    相反，您应该手动调用 `forward` 方法来计算 DPO 损失。
    """

    def __init__(
        self, model: TutorialLLM, beta: float = 0.1, positive_weight: float = 0.8
    ) -> None:
        """
        使用对齐模型和超参数初始化封装器。

        参数：
            model：要优化的微调模型。
            beta：控制对齐损失强度的超参数。
            positive_weight：DPO 损失中正奖励的权重。它应该在 [0, 1] 之间。
        """
        self.aligned_model = model
        self.beta = beta
        self.positive_weight = positive_weight
        self.negative_weight = 1 - positive_weight
        # 克隆模型以创建 DPO 的参考模型
        self.reference_model = copy.deepcopy(model)

    def forward(
        self,
        positive_token_ids: Tensor,
        positive_labels: Tensor,
        negative_token_ids: Tensor,
        negative_labels: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        两个模型的正向传播以计算 DPO 损失。

        参数：
            positive_token_ids：形状为 (B, T) 的张量，其中 B 是批大小，T 是标记序列长度。
                该张量包含正输入序列的标记 ID。
            positive_labels：形状为 (B, T) 的张量，其中 B 是批大小，T 是标记序列长度。
                该张量包含正目标序列的真实标记 ID。
            negative_token_ids：形状为 (B, T) 的张量，其中 B 是批大小，T 是标记序列长度。
                该张量包含负输入序列的标记 ID。
            negative_labels：形状为 (B, T) 的张量，其中 B 是批大小，T 是标记序列长度。
                该张量包含负目标序列的真实标记 ID。
            beta：控制对齐损失强度的超参数。

        返回：
            DPO 损失和奖励边际。
        """
        # 在对齐模型和参考模型上正向传播正样本和负样本
        _, positive_loss = self.aligned_model(
            positive_token_ids, positive_labels, False
        )
        _, negative_loss = self.aligned_model(
            negative_token_ids, negative_labels, False
        )
        with torch.inference_mode():
            _, reference_positive_loss = self.reference_model(
                positive_token_ids, positive_labels, False
            )
            _, reference_negative_loss = self.reference_model(
                negative_token_ids, negative_labels, False
            )

        # 实现 DPO（直接偏好优化）损失
        positive_reward = reference_positive_loss - positive_loss
        negative_reward = negative_loss - reference_negative_loss
        # 我们为正奖励和负奖励选择不同的权重。在我们的例子中，我们为正奖励设置了更高的权重，
        # 以避免模型在正样本上的性能下降。性能下降问题在 DPO 中很常见，
        # 模型倾向于优化负奖励多于正奖励，因为在负样本上表现更差更容易。
        reward_margin = (
            self.positive_weight * positive_reward
            + self.negative_weight * negative_reward
        )
        loss = -F.logsigmoid(self.beta * reward_margin).mean()
        return loss, reward_margin.mean()
