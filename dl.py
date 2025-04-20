from module import preprocess_features, discretize_labels, features_dict, X_all
import numpy as np
from model.MLP import HybridModel, HybridResNetModel, HybridGNNModel, HybridTransformerModel, MLPModel, MLPResNetModel, MLPGNNModel, MLPTransformerModel
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau

import argparse
import wandb


wandb.login(key="2bb94cfa588f1afb85377da3cd5c484592a4fbf6")


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


def main(args):

    # 离散化标签
    if args.mode == "regression":
        num_classes = 1
        labels_tensor = torch.tensor(X_all[-1], dtype=torch.float32)  # 标签转为 (N, 1)
    else:
        num_classes = 20  # 设置类别数量
        labels = discretize_labels(X_all[-1], num_classes=num_classes)
        labels_tensor = torch.tensor(labels, dtype=torch.long)  # 分类任务的标签应为整数类型


    # 初始化模型
    if args.method == "Hybrid":
        model = HybridModel(feature_dims, hidden_dim=128, output_dim=num_classes, mode=args.mode)  # 20 个类别
    elif args.method == "Transformer":
        model = HybridTransformerModel(feature_dims, hidden_dim=128, output_dim=num_classes, mode=args.mode) 
    elif args.method == "ResNet":
        model = HybridResNetModel(feature_dims, hidden_dim=128, output_dim=num_classes, mode=args.mode) 
    elif args.method == "GNN":
        model = HybridGNNModel(feature_dims, hidden_dim=128, output_dim=num_classes, mode=args.mode) 
    elif args.method == "MLP":
        model = MLPModel(feature_dims, hidden_dim=128, output_dim=num_classes, mode=args.mode) 
    elif args.method == "MLPResNet":
        model = MLPResNetModel(feature_dims, hidden_dim=128, output_dim=num_classes, mode=args.mode) 
    elif args.method == "MLPGNN":
        model = MLPGNNModel(feature_dims, hidden_dim=128, output_dim=num_classes, mode=args.mode)  
    elif args.method == "MLPTransformer":
        model = MLPTransformerModel(feature_dims, hidden_dim=128, output_dim=num_classes, mode=args.mode) 

    # 转换为 TensorDataset 和 DataLoader
    dataset = TensorDataset(*list(features_dict.values()), labels_tensor)
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 训练、验证和测试模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 定义损失函数和优化器
    if args.mode == "regression":
        criterion = nn.MSELoss()  
    else:
        # unique_classes, class_counts = torch.unique(labels_tensor, return_counts=True)
        # class_weights = torch.tensor([1.0 / count for count in class_counts])
        criterion = nn.CrossEntropyLoss()    
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=args.weight_decay)  # 添加 L2 正则化
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            inputs = {name: batch[i].to(device) for i, name in enumerate(features_dict.keys())}
            
            targets = batch[-1].to(device)
            if args.mode == "classification":
                targets = targets.long()
            else:
                targets = targets.unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(inputs)
            # import pdb; pdb.set_trace()
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
                targets = batch[-1].to(device)
                if args.mode == "classification":
                    targets = targets.long()
                else:
                    targets = targets.unsqueeze(1)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                outputs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
            print(f"Validation Loss: {val_loss / len(val_loader):.4f}")

            # 使用 wandb 记录指标
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": running_loss / len(train_loader),
                "val_loss": val_loss / len(val_loader),
            })
            if args.mode == "classification":
                print(f"Validation Accuracy: {100 * correct / total:.2f}%")
                wandb.log({
                    "val_accuracy": 100 * correct / total
                })


    import os
    os.makedirs(f"./outputs/",exist_ok=True)
    torch.save(model.state_dict(),f"./outputs/{args.weight_decay}-{args.mode}-{args.method}.pth")

    # 测试
    model.eval()
    with torch.no_grad():
        test_loss = correct = total = 0
        for batch in test_loader:
            inputs = {name: batch[i].to(device) for i, name in enumerate(features_dict.keys())}
            targets = batch[-1].to(device).long()
            if args.mode == "classification":
                    targets = targets.long()
            else:
                targets = targets.unsqueeze(1)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            outputs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            wandb.log({
                "test_loss": test_loss / len(test_loader),
            })
        if args.mode == "classification":
            print(f"Test Accuracy: {100 * correct / total:.2f}%")
            wandb.log({
                "Test Accuracy": 100 * correct / total,
            })
    print(f"Test Loss: {test_loss / len(test_loader):.4f}")

            


# 示例：定义模型
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cell Annotation Script")
    parser.add_argument('--i', type=list, default=[0,1,2,3,4,5,6,7],
                        help='Path to the input CSV file containing cell cluster markers.')
    parser.add_argument('--preprocess', type=bool, default=True)
    parser.add_argument('--mode', type=str, default="classification",choices=['regression','classification'])
    parser.add_argument('--method', type=str, default="MLPTransformer",choices=['Transformer','ResNet','GNN','Hybrid','MLPTransformer','MLPResNet','MLPGNN','MLP'])
    parser.add_argument('--weight_decay', type=float, default=1e-4,choices=[1e-4,1e-3,1e-2,1e-1])
    args = parser.parse_args()

    # 初始化 wandb
    wandb.init(project="16s",name=f"{args.mode}-{args.method}")
    main(args)
    