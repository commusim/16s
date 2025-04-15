import numpy as np
import torch
from torch.utils.data import Dataset

class CustomCsvDataset(Dataset):
    def __init__(self, csv_file):
        """
        初始化Dataset类。
        :param csv_file: CSV文件路径
        """
        # 读取CSV文件
        with open(csv_file, 'r') as f:
            lines = f.readlines()
        import pdb; pdb.set_trace()
        # 解析CSV文件内容
        self.labels = np.array(lines[0].strip().split(','), dtype=np.float32)  # 第一行是标签
        self.ids = lines[1].strip().split(',')  # 第二行是ID
        self.features = np.array([line.strip().split(',') for line in lines[2:]], dtype=np.float32)  # 剩下的是特征
        # 转换为Tensor
        self.labels_tensor = torch.tensor(self.labels, dtype=torch.float32)
        self.features_tensor = torch.tensor(self.features, dtype=torch.float32)

    def __len__(self):
        """
        返回数据集的长度。
        """
        return len(self.features)

    def __getitem__(self, idx):
        """
        获取指定索引的数据。
        :param idx: 数据索引
        :return: 标签、ID和特征
        """
        label = self.labels_tensor[idx]
        feature = self.features_tensor[idx]
        id_ = self.ids[idx]
        return label, id_, feature

    def to_numpy(self):
        """
        将数据集转换为NumPy格式。
        :return: NumPy数组形式的标签和特征
        """
        return self.labels, self.features


class CustomNpyDataset(Dataset):
    def __init__(self, npy_file):
        """
        初始化Dataset类。
        :param npy_file: .npy文件路径
        """
        # 加载.npy文件
        self.data = np.load(npy_file)  # 假设.npy文件的形状为 (N, 98824)
        
        # 定义特征分组的列索引范围
        self.label_col = 0  # 第一列为标签
        self.id_col = 1     # 第二列为ID
        self.diversity_cols = slice(2, 6)  # 第2-5列为多样性特征
        self.metabolism_gene_cols = slice(6, 7243)  # 第6-7242列为代谢和基因特征
        self.direct_microbe_cols = slice(7243, 82732)  # 第7243-82731列为直接检测到的菌特征
        self.species_abundance_cols = slice(82732, 85881)  # 第82732-85880列为物种层级丰度特征
        self.synthetic_features_cols = slice(85881, 85974)  # 第85881-85973列为合成特征
        self.genus_species_cols = slice(85974, 88351)  # 第85974-88350列为属和种的物种层级丰度特征
        self.pathway_cols = slice(88351, 98819)  # 第88351-98818列为基因和代谢通路特征
        self.pathogen_cols = slice(98819, 98824)  # 后续补充的病原菌检测丰度特征

    def __len__(self):
        """
        返回数据集的长度。
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        获取指定索引的数据。
        :param idx: 数据索引
        :return: 标签、ID和各个特征分组
        """
        sample = self.data[idx]
        
        # 提取标签和ID
        label = sample[self.label_col]
        id_ = sample[self.id_col]
        
        # 提取各特征分组
        diversity_features = sample[self.diversity_cols]
        metabolism_gene_features = sample[self.metabolism_gene_cols]
        direct_microbe_features = sample[self.direct_microbe_cols]
        species_abundance_features = sample[self.species_abundance_cols]
        synthetic_features = sample[self.synthetic_features_cols]
        genus_species_features = sample[self.genus_species_cols]
        pathway_features = sample[self.pathway_cols]
        pathogen_features = sample[self.pathogen_cols]
        
        return {
            "label": label,
            "id": id_,
            "diversity_features": diversity_features,
            "metabolism_gene_features": metabolism_gene_features,
            "direct_microbe_features": direct_microbe_features,
            "species_abundance_features": species_abundance_features,
            "synthetic_features": synthetic_features,
            "genus_species_features": genus_species_features,
            "pathway_features": pathway_features,
            "pathogen_features": pathogen_features
        }


# 示例使用
if __name__ == "__main__":
    # 创建Dataset实例
    dataset = CustomNpyDataset("./out/dataset.npy")
    # 打印Dataset的长度
    print(f"Dataset length: {len(dataset)}")
    
    # 获取第一个样本
    dict = dataset[0]
    import pdb; pdb.set_trace()
    print(f"First sample - Label: {label}, ID: {id_}, Feature: {feature}")
    
    # 将数据集转换为NumPy格式
    labels_np, features_np = dataset.to_numpy()
    print(f"Labels shape (NumPy): {labels_np.shape}")
    print(f"Features shape (NumPy): {features_np.shape}")