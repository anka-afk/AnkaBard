from collections.abc import Generator
import json
import random
import torch
from torch import Tensor


class Dataset:
    """
    数据集类，用于加载和处理诗歌数据。它提供了生成预训练、指令微调和对齐数据批次的方法。

    数据表示为标记 ID 列表，其中每个标记是词汇表中的一个字符。
    标记 ID 和字符映射存储在 `encode` 和 `decode` 方法中。
    """

    def __init__(
        self,
        input_path: str = "data.json",
        batch_size: int = 16,
        max_length: int = 256,
        device: str = "cpu",
    ) -> None:
        """
        使用 JSON 文件或诗歌初始化数据集。

        输入 JSON 文件包含一个诗歌列表，每首诗都有一个标题和一段段落列表。
        所有数据将以 5:3:2 的比例分为预训练、指令微调和对齐数据。
        + 对于预训练数据，诗歌组合在一起形成长文本。
        + 对于指令微调数据，每首诗都格式化为指令-响应对。
            指令是固定字符串“请用以下题目写一首诗”和标题，而响应是诗歌的段落。
        + 对于对齐数据，每个项目包含一对正负诗歌。正对是原始诗歌，
            而负对至少有一个段落被其他诗歌中的随机段落替换。

        每个类别中的数据将进一步分为训练集和评估集。
        所有数据都将标记化为标记 ID 序列，其中每个标记是词汇表中的一个字符。
        这对于模型处理数据是必要的。

        参数：
            input_path：包含诗歌数据的 JSON 文件的路径。
            batch_size：批次中的项目数。
            max_length：要处理的文本的最大长度。
            device：运行模型的设备（'cpu' 或 'cuda'）。
        """
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device

        # 加载诗歌 JSON 文件
        poems = json.load(open(input_path, "r", encoding="utf-8"))
        # 打乱诗歌使其随机
        random.seed(2024)
        random.shuffle(poems)
        # 将数据按 5:3:2 的比例分为预训练、指令微调和对齐
        pretrain_poems = poems[: int(len(poems) * 0.5)]
        finetune_poems = poems[int(len(poems) * 0.5) : int(len(poems) * 0.8)]
        alignment_poems = poems[int(len(poems) * 0.8) :]

        # 重新格式化预训练数据。所有诗歌直接连接起来形成长文本。
        # 我们不关心预训练阶段的格式。这些数据仅用于让
        # 模型学习诗歌文本的样式。
        pretrain_texts = []
        for poem in pretrain_poems:
            paragraphs_text = "\n".join(poem["paragraphs"])
            pretrain_texts.append(f'{poem["title"]}\n{paragraphs_text}')
        pretrain_text = "\n\n".join(pretrain_texts)
        print("整个预训练数据是一个长文本，所有诗歌都连接在一起。以下是前 100 个字符：")
        print(pretrain_text[:100])

        # 重新格式化指令微调数据。
        # 每首诗都格式化为指令-响应对。
        finetune_texts = []
        instruction = "請用以下題目寫一首詩"
        instruction_label = "<INS>"
        input_label = "<INP>"
        response_label = "<RES>"
        for poem in finetune_poems:
            paragraphs_text = "\n".join(poem["paragraphs"])
            content = f"{instruction_label}{instruction}{input_label}{poem['title']}{response_label}{paragraphs_text}"
            finetune_texts.append(content)
        print("指令微调数据是格式化文本的列表。以下是第一个项目：")
        print(finetune_texts[0])

        # 重新格式化对齐数据。
        # 对齐数据包括一对正负诗歌。
        # 正诗是五言诗，而负诗是随机的非五言诗。
        five_words_poems = []
        other_poems = []
        for poem in alignment_poems:
            if all(len(paragraph) == 12 for paragraph in poem["paragraphs"]):
                five_words_poems.append(poem)
            else:
                other_poems.append(poem)
        alignment_texts = []
        for positive_poem in five_words_poems:
            negative_poem = random.choice(other_poems)
            positive_paragraphs_text = "\n".join(positive_poem["paragraphs"])
            negative_paragraphs_text = "\n".join(negative_poem["paragraphs"])
            positive_text = f"{instruction_label}{instruction}{input_label}{positive_poem['title']}{response_label}{positive_paragraphs_text}"
            negative_text = f"{instruction_label}{instruction}{input_label}{negative_poem['title']}{response_label}{negative_paragraphs_text}"
            alignment_texts.append((positive_text, negative_text))
        print("对齐数据是一对正负对。以下是第一个对：")
        print(alignment_texts[0])

        # 从诗歌和指令中出现的所有字符创建词汇表。
        # 注意，我们在末尾添加了一个特殊字符 '\0'，用作文本结束标记。
        # 文本结束标记有助于模型知道何时停止生成文本。
        all_text = f'{pretrain_text}{"".join(finetune_texts)}{"".join([pair[0] + pair[1] for pair in alignment_texts])}\0'
        # 获取唯一字符的排序列表
        characters = sorted(list(set(all_text)))
        self.vocabulary_size = len(characters)
        print(f"数据集长度: {len(all_text)}, 词汇表大小: {self.vocabulary_size}")
        # 创建字符到索引的映射，反之亦然
        character_to_index = {
            character: index for index, character in enumerate(characters)
        }
        index_to_character = {
            index: character for index, character in enumerate(characters)
        }
        # 编码和解码方法，用于在字符和索引之间转换
        self.encode = lambda text: [character_to_index[character] for character in text]
        self.decode = lambda index_list: "".join(
            [index_to_character[index] for index in index_list]
        )

        # 将预训练数据分为 90% 训练集和 10% 评估集
        pretrain_data = torch.tensor(self.encode(pretrain_text), dtype=torch.long)
        self.pretrain_train_data = pretrain_data[: int(0.9 * len(pretrain_data))]
        self.pretrain_evaluate_data = pretrain_data[int(0.9 * len(pretrain_data)) :]

        # 将指令微调数据分为 90% 训练集和 10% 评估集
        finetune_data = [
            torch.tensor(self.encode(finetune_text), dtype=torch.long)
            for finetune_text in finetune_texts
        ]
        self.finetune_train_data = finetune_data[: int(0.9 * len(finetune_data))]
        self.finetune_evaluate_data = finetune_data[int(0.9 * len(finetune_data)) :]

        # 将对齐数据分为 90% 训练集和 10% 评估集
        alignment_data = [
            (
                torch.tensor(self.encode(pair[0]), dtype=torch.long),
                torch.tensor(self.encode(pair[1]), dtype=torch.long),
            )
            for pair in alignment_texts
        ]
        self.alignment_train_data = alignment_data[: int(0.9 * len(alignment_data))]
        self.alignment_evaluate_data = alignment_data[int(0.9 * len(alignment_data)) :]

    def get_batch_pretrain(self, split: str) -> tuple[Tensor, Tensor]:
        """
        生成一批预训练数据。

        每个批次都是一个长度为 `max_length` 的随机文本块。
        因此，预训练数据中没有 epoch 边界。批次始终是唯一的。

        参数：
            split：指示是生成用于训练还是评估的批次（'train' 或 'evaluate'）。

        返回：
            两个形状为 (`batch_size`, `max_length`) 的张量，其中第一个张量是输入标记，第二个张量是标签标记。
            第二个维度是文本的长度。我们通过将输入向右移动一个字符来形成每个标签。
        """
        # 选择训练集或评估集
        data = (
            self.pretrain_train_data
            if split == "train"
            else self.pretrain_evaluate_data
        )
        # 随机选择批次中每个项目的起始索引
        start_indices = torch.randint(len(data) - self.max_length, (self.batch_size,))
        # 输入文本是批次中每个项目在区间 [start_index, start_index + max_length) 中的所有字符
        inputs = torch.stack(
            [data[index : index + self.max_length] for index in start_indices]
        )
        # 标签文本是批次中每个项目在区间 [start_index + 1, start_index + max_length + 1) 中的所有字符。
        # 因此，标签文本与输入文本相同，但向右移动了 1 个字符。
        # 这为单个输入-标签对形成了 `max_length` 个训练示例。
        # 对于从 `start_index` 到 `start_index + i` 的每个子序列，其中 i = 1, 2, ..., `max_length`，标签是 `start_index + i + 1`，表示下一个字符。
        labels = torch.stack(
            [data[index + 1 : index + self.max_length + 1] for index in start_indices]
        )
        # 将张量移动到设备并返回
        return inputs.to(self.device), labels.to(self.device)

    def get_batch_generator_finetune(
        self, split: str
    ) -> Generator[tuple[Tensor, Tensor], None, None]:
        """
        获取一个生成器，用于生成指令微调数据批次。

        数据以流式方式消耗，因此生成器将继续生成批次以形成一个 epoch。
        这对于训练模型多个 epoch 而无需将所有数据加载到内存中很有用。

        参数：
            split：指示是生成用于训练还是评估的批次（'train' 或 'evaluate'）。

        生成：
            两个形状为 (batch_size, T) 的张量，其中第一个张量是输入标记，第二个张量是标签标记，T <= `max_length`。
            第二个维度是文本的长度。我们通过将输入向右移动一个字符来形成每个标签。
        """
        # 选择训练集或评估集
        data = (
            self.finetune_train_data
            if split == "train"
            else self.finetune_evaluate_data
        )

        # 初始化一个空列表来存储批次
        batch = []
        for item in data:
            batch.append(item)
            # 如果批次已满，则处理并生成
            if len(batch) >= self.batch_size:
                inputs, labels = self.process_batch(batch)
                # 为下一次迭代重置批次
                batch = []
                # 将输入和标签批次返回给调用者
                yield inputs.to(self.device), labels.to(self.device)
        # 如果仍有剩余项目，则处理并生成
        if len(batch) > 0:
            inputs, labels = self.process_batch(batch)
            yield inputs.to(self.device), labels.to(self.device)

    def get_batch_generator_alignment(
        self, split: str
    ) -> Generator[tuple[Tensor, Tensor, Tensor, Tensor], None, None]:
        """
        获取一个生成器，用于生成对齐数据批次。

        数据以流式方式消耗，因此生成器将继续生成批次以形成一个 epoch。

        参数：
            split：指示是生成用于训练还是评估的批次（'train' 或 'evaluate'）。

        返回：
            两组形状为 (batch_size, T) 的张量，分别用于正批次和负批次。
            每组包含输入标记和标签标记。
        """
        # 所有输入和标签都初始化为最大长度的零
        positive_inputs = torch.zeros(
            self.batch_size, self.max_length, dtype=torch.long
        )
        positive_labels = torch.zeros(
            self.batch_size, self.max_length, dtype=torch.long
        )
        negative_inputs = torch.zeros(
            self.batch_size, self.max_length, dtype=torch.long
        )
        negative_labels = torch.zeros(
            self.batch_size, self.max_length, dtype=torch.long
        )

        # 选择训练集或评估集
        data = (
            self.alignment_train_data
            if split == "train"
            else self.alignment_evaluate_data
        )

        # 初始化一个空列表来存储批次
        batch = []
        for positive_item, negative_item in data:
            batch.append((positive_item, negative_item))
            # 如果批次已满，则处理并生成
            if len(batch) >= self.batch_size:
                positive_inputs, positive_labels = self.process_batch(
                    [item[0] for item in batch]
                )
                negative_inputs, negative_labels = self.process_batch(
                    [item[1] for item in batch]
                )
                # 为下一次迭代重置批次
                batch = []
                # 将输入和标签批次返回给调用者
                yield positive_inputs.to(self.device), positive_labels.to(
                    self.device
                ), negative_inputs.to(self.device), negative_labels.to(self.device)
        # 如果仍有剩余项目，则处理并生成
        if len(batch) > 0:
            positive_inputs, positive_labels = self.process_batch(
                [item[0] for item in batch]
            )
            negative_inputs, negative_labels = self.process_batch(
                [item[1] for item in batch]
            )
            yield positive_inputs.to(self.device), positive_labels.to(
                self.device
            ), negative_inputs.to(self.device), negative_labels.to(self.device)

    def process_batch(self, batch: list) -> tuple[Tensor, Tensor]:
        """
        处理一批标记 ID 列表。

        参数：
            batch：标记 ID 列表的列表，其中每个列表都是由标记 ID 表示的诗歌。

        返回：
            输入标记 ID 列表和标签标记 ID 的批次。标签指代每个输入序列的下一个字符。
        """
        # 所有输入和标签都初始化为最大长度的零
        inputs = torch.zeros(len(batch), self.max_length, dtype=torch.long)
        labels = torch.zeros(len(batch), self.max_length, dtype=torch.long)
        for i, item in enumerate(batch):
            # 将实际值分配给零初始化的张量
            available_length = (
                len(item) if len(item) < self.max_length else self.max_length
            )
            inputs[i, :available_length] = item[:available_length]
            # 与预训练数据格式相同，标签是输入的下一个字符
            labels[i, : available_length - 1] = item[1:available_length]

            # 通过将剩余的零设置为 -100 来掩码它们（损失函数将忽略这些标记）
            mask = labels[i] == 0
            indices = torch.nonzero(mask).squeeze()
            # 检查标签中是否有多个零
            if indices.numel() > 1:
                # 排除第一个零，因为它标记着文本的结束
                labels[i, indices[1:]] = -100
        return inputs, labels
