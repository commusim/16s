import csv

# 输入文件路径
large_csv_path = '/root/commusim/16s/data/id.csv'  # 包含 id 的 CSV 文件
output_csv_path = '/root/commusim/16s/data/output.csv'  # 输出的 CSV 文件

label_csv_path = '/root/commusim/16s/output.csv'   # 包含 label 的 CSV 文件

# 加载 label.csv 中的所有 id 和对应的 label 到一个字典中
label_dict = {}
with open(label_csv_path, mode='r', encoding='utf-8') as label_file:
    reader = csv.reader(label_file)
    next(reader)  # 跳过表头（如果有的话）
    for row in reader:
        if row:  # 确保行不为空
            label_dict[row[0]] = row[1]  # 假设第一列为 id，第二列为 label

# 统计筛选的行数和总行数
filtered_count = 0
total_count = 0
import pdb; pdb.set_trace()
# 打开大文本数据集进行筛选
with open(large_csv_path, mode='r', encoding='utf-8') as large_file:
    reader = csv.reader(large_file)
    
    # 打开输出文件进行写入
    with open(output_csv_path, mode='w', encoding='utf-8', newline='') as output_file:
        writer = csv.writer(output_file)
        
        # # 写入表头（假设原文件有表头）
        # header = next(reader)
        # writer.writerow([header[0], 'Label'])  # 添加新表头
        
        # 遍历大文本数据集的每一行
        for row in reader:
            total_count += 1  # 统计总行数
            current_id = row[0]  # 当前行的 id
            
            # 检查当前 id 是否存在于 label_dict 中
            if current_id in label_dict:
                filtered_count += 1  # 统计筛选的行数
                new_row = [current_id, label_dict[current_id]]  # 添加对应的 label
            else:
                new_row = [current_id, 'False']  # 如果不存在，添加 False
                
            # 写入新的行到输出文件
            writer.writerow(new_row)

# 输出统计信息
print(f"总行数: {total_count}")
print(f"筛选行数: {filtered_count}")
if total_count > 0:
    print(f"筛选比例: {filtered_count / total_count * 100:.2f}%")
else:
    print("总行数为 0，无法计算比例。")