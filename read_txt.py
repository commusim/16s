import csv

# 文件路径
input_file_path = '/zangzelin/data/guhe-microbiomedata/stage1sampledata-WL-clean.txt' # 输入的 txt 文件
output_file_path = 'output.csv'     # 输出的 csv 文件

# 打开输入文件并逐行读取
with open(input_file_path, 'r', encoding='utf-8') as infile, \
     open(output_file_path, 'w', newline='', encoding='utf-8') as outfile:
    
    # 创建 CSV 写入器
    writer = csv.writer(outfile)
    
    # 逐行处理数据
    for line in infile:
        # 按制表符分割每行数据
        parts = line.strip().split('\t')
        # 确保每行有正确的列数（第一列为 ID，后续为特征）
        if len(parts) > 98823:
            # 如果列数不足，补齐缺失的列（用空字符串或0填充）    
            # 写入表头（可选）
            header = parts[:98823]
            header[98822] = header[98822][:5]
            parts_data = parts[-98823:]
            parts_data[0] = parts_data[0][5:]
            writer.writerow(header)
            writer.writerow(parts_data)
            writer.writerow("\n")

        # 确保每行有正确的列数（第一列为 ID，后续为特征）
        if len(parts) < 98823:
            # 如果列数不足，补齐缺失的列（用空字符串或0填充）
            parts += ['0'] * (98823 - len(parts))
        if(len(parts) == 98823):
            # 写入 CSV 文件
            writer.writerow(parts)
            writer.writerow("\n")

print(f"数据已成功保存到 {output_file_path}")