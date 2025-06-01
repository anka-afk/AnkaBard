import copy
from typing import Optional
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import Tensor


class MambaSSM(nn.Module):
    def __init__(
        self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = self.expand * self.d_model

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )

        self.x_proj = nn.Linear(
            self.d_inner, self.d_state * 2, bias=False
        )  # For B and C
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)
        nn.init.constant_(self.dt_proj.bias, -5.0)  # 初始化偏置以稳定 delta

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        self.A = nn.Parameter(torch.arange(1, d_state + 1).float().reshape(1, -1))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.delta_softplus = nn.Softplus()

    def forward(self, x: Tensor):
        # x: (B, L, D)
        B, L, D = x.shape

        x_and_res = self.in_proj(x)  # (B, L, 2 * D_inner)
        x_conv, x_gate = x_and_res.split(self.d_inner, dim=-1)

        x_conv = x_conv.transpose(1, 2)  # (B, D_inner, L)
        x_conv = self.conv1d(x_conv)[:, :, :L]  # (B, D_inner, L)
        x_conv = F.silu(x_conv)
        x_conv = x_conv.transpose(1, 2)  # (B, L, D_inner)

        # SSM
        delta = self.delta_softplus(self.dt_proj(x_conv))  # (B, L, D_inner)
        B_proj, C_proj = self.x_proj(x_conv).split(
            self.d_state, dim=-1
        )  # (B, L, d_state), (B, L, d_state)

        # Discretize A, B
        dA = torch.exp(delta.unsqueeze(-1) * self.A)  # (B, L, D_inner, d_state)
        dB = delta.unsqueeze(-1) * B_proj.unsqueeze(2)  # (B, L, D_inner, d_state)

        h = torch.zeros(
            B, self.d_inner, self.d_state, device=x.device
        )  # (B, D_inner, d_state)
        output_ssm = []
        for i in range(L):
            h = dA[:, i] * h + dB[:, i] * x_conv[:, i].unsqueeze(
                -1
            )  # (B, D_inner, d_state)
            output_ssm.append(
                torch.einsum("bs,bds->bd", C_proj[:, i], h)
            )  # (B, D_inner)
        output_ssm = torch.stack(output_ssm, dim=1)  # (B, L, D_inner)

        output = output_ssm + x_conv * self.D  # (B, L, D_inner)
        output = output * F.silu(x_gate)  # (B, L, D_inner)

        return self.out_proj(output)  # (B, L, D)


class MambaBlock(nn.Module):
    def __init__(
        self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2
    ):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba_ssm = MambaSSM(d_model, d_state, d_conv, expand)

    def forward(self, x: Tensor):
        # x: (B, L, D)
        residual = x
        x = self.norm(x)
        x = self.mamba_ssm(x)
        return x + residual


class TutorialLLM(nn.Module):
    """
    教程大型语言模型。

    这是一个基于 Mamba 架构构建的非常简单的语言模型。
    """

    def __init__(
        self,
        vocabulary_size: int,
        dim_embed: int,
        max_length: int,
        num_layer: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        device: str = "cpu",
    ) -> None:
        """
        使用标记嵌入表、位置嵌入表、多个 Mamba 块、一个最终层归一化层和一个线性层初始化模型。

        参数：
            vocabulary_size：词汇表中唯一标记的数量。
            dim_embed：模型中嵌入向量的维度。
            max_length：要处理的文本的最大长度。也称为最大上下文长度。
            num_layer：模型中的 Mamba 块数。
            d_state：Mamba SSM 的状态维度。
            d_conv：Mamba SSM 卷积核的大小。
            expand：Mamba SSM 内部维度扩展因子。
            device：运行模型的设备，可以是 'cpu' 或 'cuda'。
        """
        super().__init__()
        self.max_length = max_length
        self.device = device
        # 创建标记嵌入表将标记 ID 转换为向量
        self.token_embedding_table = nn.Embedding(vocabulary_size, dim_embed)
        # 创建位置嵌入表为标记向量添加位置信息
        self.position_embedding_table = nn.Embedding(max_length, dim_embed)
        # 创建一系列 Mamba 块
        self.mamba_blocks = nn.Sequential(
            *[MambaBlock(dim_embed, d_state, d_conv, expand) for _ in range(num_layer)]
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
        # 将嵌入通过 Mamba 块
        embedding = self.mamba_blocks(
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
