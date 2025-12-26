import os
from raw2hdf5 import process_one_hand, write_hdf5

ROOT = "/gemini-2/user/private/data/data_umi_fruit"
session_path = os.path.join(ROOT, "session_063")
out_path = os.path.join(ROOT, "hdf5", "62.hdf5")

# 找左右手
hands = os.listdir(session_path)
left_hand = [h for h in hands if h.startswith("left_hand")][0]
right_hand = [h for h in hands if h.startswith("right_hand")][0]
left_path = os.path.join(session_path, left_hand)
right_path = os.path.join(session_path, right_hand)

# 处理单手
left_data = process_one_hand(left_path)
right_data = process_one_hand(right_path)

# 写 HDF5，强制写入
write_hdf5(out_path, left_data, right_data, max_start_diff_ms=100)
