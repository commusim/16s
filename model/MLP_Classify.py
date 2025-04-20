import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import argparse
# 定义模型
class HybridModel(nn.Module):
    def __init__(self, feature_dims, hidden_dim=128, output_dim=1, dropout_rate=0.3):
        """
        :param feature_dims: 各特征组的维度，字典形式 {'feature_name': dim}
        :param hidden_dim: 隐藏层维度
        :param output_dim: 输出维度（1 表示回归，>1 表示分类）
        :param dropout_rate: Dropout 率
        """
        super(HybridModel, self).__init__()

        # 定义每个特征分组的独立全连接层
        self.feature_layers = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(dim, hidden_dim),  # 将特征映射到隐藏层维度
                nn.ReLU(),                   # 激活函数
                nn.Dropout(dropout_rate)     # Dropout 防止过拟合
            )
            for name, dim in feature_dims.items()
        })

        # 共享全连接层
        self.shared_fc = nn.Sequential(
            nn.Linear(len(feature_dims) * hidden_dim, hidden_dim),  # 拼接后的特征进入全连接层
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),                 # 可选：进一步压缩特征
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)                  # 输出层
        )

    def forward(self, feature_dict):
        """
        :param feature_dict: 字典形式的输入特征，{'feature_name': tensor}
        :return: 模型输出
        """
        # 对每个特征分组分别通过对应的全连接层
        embeddings = [self.feature_layers[name](feature_dict[name]) for name in feature_dict]

        # 拼接所有特征分组的输出
        combined = torch.cat(embeddings, dim=1)

        # 通过共享全连接层生成最终输出
        logits = self.shared_fc(combined)
        return torch.softmax(logits, dim=1)  # 返回 Softmax 输出

class ResidualBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_rate):
        super(ResidualBlock, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        return x + self.fc(x)

class HybridResNetModel(nn.Module):
    def __init__(self, feature_dims, hidden_dim=128, output_dim=1, dropout_rate=0.3):
        super(HybridResNetModel, self).__init__()

        # 定义每个特征分组的独立全连接层
        self.feature_layers = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            )
            for name, dim in feature_dims.items()
        })

        # 共享全连接层（含残差块）
        self.shared_fc = nn.Sequential(
            ResidualBlock(len(feature_dims) * hidden_dim, hidden_dim, dropout_rate),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, feature_dict):
        embeddings = [self.feature_layers[name](feature_dict[name]) for name in feature_dict]
        combined = torch.cat(embeddings, dim=1)

        # 通过共享全连接层生成最终输出
        logits = self.shared_fc(combined)
        return torch.softmax(logits, dim=1)

from torch_geometric.nn import GCNConv

class HybridGNNModel(nn.Module):
    def __init__(self, feature_dims, hidden_dim=128, output_dim=1, dropout_rate=0.3):
        super(HybridGNNModel, self).__init__()

        # 定义每个特征分组的独立全连接层
        self.feature_layers = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            )
            for name, dim in feature_dims.items()
        })

        # 图卷积层
        self.gcn = GCNConv(hidden_dim, hidden_dim)

        # 共享全连接层
        self.shared_fc = nn.Sequential(
            nn.Linear(hidden_dim * len(feature_dims), hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, feature_dict, edge_index):
        embeddings = [self.feature_layers[name](feature_dict[name]) for name in feature_dict]
        combined = torch.cat(embeddings, dim=1)

        # 图卷积
        gcn_output = self.gcn(combined, edge_index)
        pooled_output = gcn_output.mean(dim=1)  # 平均池化

        # 通过共享全连接层生成最终输出
        logits = self.shared_fc(pooled_output)
        return torch.softmax(logits, dim=1)


class HybridTransformerModel(nn.Module):
    def __init__(self, feature_dims, hidden_dim=128, output_dim=1, dropout_rate=0.3, num_heads=4):
        super(HybridTransformerModel, self).__init__()

        # 定义每个特征分组的独立全连接层
        self.feature_layers = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            )
            for name, dim in feature_dims.items()
        })

        # 多头注意力层
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout_rate)

        # 共享全连接层
        self.shared_fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim*2, output_dim)
        )

    def forward(self, feature_dict):
        embeddings = [self.feature_layers[name](feature_dict[name]) for name in feature_dict]
        combined = torch.stack(embeddings, dim=0)  # (num_features, batch_size, hidden_dim)

        # 使用多头注意力
        attn_output, _ = self.attention(combined, combined, combined)
        pooled_output = attn_output.mean(dim=0)  # 平均池化
        # import pdb; pdb.set_trace()

        # 通过共享全连接层生成最终输出
        logits = self.shared_fc(pooled_output)
        return torch.softmax(logits, dim=1)

# 定义离散化函数
def discretize_labels(labels, num_classes=20):
    """
    将连续标签离散化为 0 到 num_classes-1 的类别。
    :param labels: 连续标签 (numpy 数组)
    :param num_classes: 类别数量
    :return: 离散化的类别标签
    """
    bins = np.linspace(0, 100, num_classes + 1)  # 创建等宽区间
    discrete_labels = np.digitize(labels, bins) - 1  # 将标签映射到区间索引
    return discrete_labels


# 示例：定义模型
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cell Annotation Script")
    parser.add_argument('--i', type=list, default=[0,1,2,3,4,5,6,7],
                        help='Path to the input CSV file containing cell cluster markers.')
    parser.add_argument('--preprocess', type=bool, default=True)
    parser.add_argument('--mode', type=str, default="regression")
    parser.add_argument('--method', type=str, default="RF")
    args = parser.parse_args()




    from module import preprocess_features
    import numpy as np

    # 加载数据
    sample = np.load("out/dataset.npy")

    # 定义特征分组的列索引范围
    label_col = 0  # 第一列为标签
    id_col = 1     # 第二列为ID
    diversity_cols = slice(2, 6)  # 第2-5列为多样性特征
    metabolism_gene_cols = slice(6, 7243)  # 第6-7242列为代谢和基因特征
    direct_microbe_cols = slice(7243, 82732)  # 第7243-82731列为直接检测到的菌特征
    species_abundance_cols = slice(82732, 85881)  # 第82732-85880列为物种层级丰度特征
    synthetic_features_cols = slice(85881, 85974)  # 第85881-85973列为合成特征
    genus_species_cols = slice(85974, 88351)  # 第85974-88350列为属和种的物种层级丰度特征
    pathway_cols = slice(88351, 98819)  # 第88351-98818列为基因和代谢通路特征
    pathogen_cols = slice(98819, 98824)  # 后续补充的病原菌检测丰度特征

    # 提取各特征分组并存入字典
    features_dict = {
        "diversity": sample[:, diversity_cols],  # 多样性特征
        "metabolism_gene": sample[:, metabolism_gene_cols],  # 代谢和基因特征
        "direct_microbe": sample[:, direct_microbe_cols],  # 直接检测到的菌特征
        "species_abundance": sample[:, species_abundance_cols],  # 物种层级丰度特征
        "synthetic": sample[:, synthetic_features_cols],  # 合成特征
        "genus_species": sample[:, genus_species_cols],  # 属和种的物种层级丰度特征
        "pathway": sample[:, pathway_cols],  # 基因和代谢通路特征
        "pathogen": sample[:, pathogen_cols]  # 病原菌检测丰度特征
    }

    # 打印字典中的键和形状，验证数据是否正确提取
    for key, value in features_dict.items():
        features_dict[key] = torch.tensor(preprocess_features(value), dtype=torch.float32)
        print(f"{key}: {features_dict[key].shape}")

    # 假设各特征分组的维度如下
    feature_dims = {
        "diversity": 4,               # 多样性特征
        "metabolism_gene": 500,      # 代谢基因特征
        "direct_microbe": 500,      # 直接检测到的菌特征
        "species_abundance": 500,    # 物种层级丰度特征
        "synthetic": 93,              # 合成特征
        "genus_species": 500,        # 属和种的物种层级丰度特征
        "pathway": 500,             # 基因和代谢通路特征
        "pathogen": 5                # 病原菌检测丰度特征
    }

    # 离散化标签
    num_classes = 20  # 设置类别数量
    labels = discretize_labels(sample[:, label_col], num_classes=num_classes)
    labels_tensor = torch.tensor(labels, dtype=torch.long)  # 分类任务的标签应为整数类型
    # 初始化模型
    
    # model = HybridModel(feature_dims, hidden_dim=128, output_dim=num_classes)  # 20 个类别
    # model = HybridTransformerModel(feature_dims, hidden_dim=128, output_dim=num_classes) 
    model = HybridResNetModel(feature_dims, hidden_dim=128, output_dim=num_classes) 
    # model = HybridGNNModel(feature_dims, hidden_dim=128, output_dim=num_classes) 
    
    # 转换为 TensorDataset 和 DataLoader
    dataset = TensorDataset(*list(features_dict.values()), labels_tensor)
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()  
    optimizer = optim.Adam(model.parameters(), lr=0.001,weight_decay=1e-4)

    # 训练、验证和测试模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            inputs = {name: batch[i].to(device) for i, name in enumerate(features_dict.keys())}
            targets = batch[-1].to(device).long()
            import pdb; pdb.set_trace()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {running_loss / len(train_loader):.4f}")

        # 验证
        model.eval()
        with torch.no_grad():
            val_loss = correct = total = 0
            for batch in val_loader:
                inputs = {name: batch[i].to(device) for i, name in enumerate(features_dict.keys())}
                targets = batch[-1].to(device).long()

                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
            print(f"Validation Loss: {val_loss / len(val_loader):.4f}")
            print(f"Validation Accuracy: {100 * correct / total:.2f}%")

    # 测试
    model.eval()
    with torch.no_grad():
        test_loss = correct = total = 0
        for batch in test_loader:
            inputs = {name: batch[i].to(device) for i, name in enumerate(features_dict.keys())}
            targets = batch[-1].to(device).long()

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
        print(f"Test Loss: {test_loss / len(test_loader):.4f}")
        print(f"Test Accuracy: {100 * correct / total:.2f}%")

# Test Loss: 2.9538
# Test Accuracy: 11.86%

