
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np

def select_highly_variable_features(features, top_k=500):
    """
    选择高变特征。
    :param features: 输入特征矩阵 (numpy 数组)
    :param top_k: 保留的高变特征数量
    :return: 筛选后的高变特征矩阵
    """
    # 计算每个特征的方差
    variances = np.var(features, axis=0)
    # 找到方差最大的 top_k 个特征的索引
    top_k_indices = np.argsort(variances)[-top_k:]
    # 提取高变特征
    selected_features = features[:, top_k_indices]
    return selected_features

def log_transform(features):
    """
    对非零值进行对数变换。
    :param features: 输入特征矩阵 (numpy 数组)
    :return: 转换后的特征矩阵
    """
    features = features.copy()
    features[features > 0] = np.log1p(features[features > 0])  # 对非零值取 log1p
    return features

def min_max_normalize(features):
    """
    使用 Min-Max 归一化。
    :param features: 输入特征矩阵 (numpy 数组)
    :return: 归一化后的特征矩阵
    """
    scaler = MinMaxScaler()
    normalized_features = scaler.fit_transform(features)
    return normalized_features

def z_score_standardize(features):
    """
    使用 Z-Score 标准化。
    :param features: 输入特征矩阵 (numpy 数组)
    :return: 标准化后的特征矩阵
    """
    scaler = StandardScaler()
    standardized_features = scaler.fit_transform(features)
    return standardized_features

def preprocess_features(features, top_k=500, normalize_method="minmax"):
    """
    综合预处理流程：高变特征选择 -> 对数变换 -> 正则化。
    :param features: 输入特征矩阵 (numpy 数组)
    :param top_k: 保留的高变特征数量
    :param normalize_method: 正则化方法 ("minmax" 或 "zscore")
    :return: 预处理后的特征矩阵
    """
    # 1. 高变特征选择
    hvf_features = select_highly_variable_features(features, top_k=top_k)
    
    # 2. 对数变换
    log_hvf_features = log_transform(hvf_features)
    
    # 3. 正则化
    if normalize_method == "minmax":
        preprocessed_features = min_max_normalize(log_hvf_features)
    elif normalize_method == "zscore":
        preprocessed_features = z_score_standardize(log_hvf_features)
    else:
        raise ValueError("normalize_method must be 'minmax' or 'zscore'")
    
    return preprocessed_features

