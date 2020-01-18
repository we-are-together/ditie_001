import pandas as pd
import glob
import os
all_files = glob.glob(os.path.join("*.csv"))    # 遍历当前目录下的所有以.csv结尾的文件
all_data_frame = []
print(all_files)
row_count = 0
for file in all_files:
    data_frame = pd.read_csv(file)
    all_data_frame.append(data_frame)
    # axis=0纵向合并 axis=1横向合并
data_frame_concat = pd.concat(all_data_frame, axis=0, ignore_index=True, sort=False)
data_frame_concat.to_csv("02total.csv", index=False, encoding="utf-8")     # 将重组后的数据集重新写入一个文件