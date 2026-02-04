# UMI / LeRobot 数据处理脚本说明

本目录包含 UMI 原始数据从 **采集 → 清洗 → 校验 → 转换 → 发布** 的完整数据处理流水线脚本。
整体流程按照 `0 → 1 → 2 → 3 → 4` 五个阶段顺序执行。

---

## 0️⃣ Session 预处理与整理

### `0_rename_session.py`

* 对原始数据中的 session 目录进行统一重命名
* 规范 session 命名格式，便于后续批量处理与索引

### `merge_sessions.py`

* 合并多个 session 为一个连续数据集
* 适用于多次采集但需要统一训练或发布的场景
* merge多个数据集合，同时会裁剪成224，加速转换

---

## 1️⃣ 空数据与结构检查

### `1_check_empty.py`

* 检查 session 中是否存在：

  * 空目录
  * 空文件
  * 缺失关键数据的 session
* 用于在正式处理前快速剔除无效数据

---

## 2️⃣ 数据合法性与质量筛选

### `2_data_sanity_filter.py`

* 对原始数据进行左右手的延时检查

### `3_bad_video.py`

* 检测并标记异常视频数据，例如：

  * 视频损坏

### `check_l_r_diff.py`

* 检查左右臂（或左右视角）数据差异
* 用于发现不同步、异常偏移等问题

### `get_fps.py`

* 统计视频真实 FPS
* 用于验证采集端 FPS 与配置是否一致

---

## 3️⃣ Raw Data → HDF5（中间格式）

### `raw2hdf5.py`

* 将原始采集数据转换为标准 HDF5 格式
* 作为中间数据格式，便于调试和后续处理

### `raw2hdf5_neareast_multcores.py`

* Raw → HDF5 的多进程版本
* 使用最近邻策略进行多模态数据对齐
* 适用于大规模数据加速处理

### `raw2hdf5_slerp.py`

* Raw → HDF5 转换（使用 SLERP 插值）
* 主要用于四元数 / 位姿数据的平滑插值

### `hdf5_test.py`

* 对生成的 HDF5 文件进行完整性与可读性测试

### `hdf5_tree.py`

* 以树状结构展示 HDF5 文件内部层级
* 用于调试字段结构与数据内容

---

## 4️⃣ Raw / HDF5 → LeRobot 数据集

### `hdf52lerobot.py`

* 将 HDF5 数据转换为 LeRobot 标准数据集格式

### `4_raw2lerobot_abs_joint.py`

* Raw data 直接转换为 LeRobot
* 使用 **绝对 joint** 表示

### `4_raw2lerobot_abs_Quaternion.py`

* Raw data → LeRobot
* 使用 **绝对位姿 + Quaternion** 表示

### `4_raw2lerobot_abs_Q_multitasks.py`

* Raw data → LeRobot（多任务版本）
* 支持多任务数据结构

### `4_raw2lerobot_abs_Q_multitasks224.py`

* 多任务版本
* 图像分辨率为 **224×224**
* 主要用于视觉模型训练

### `4_raw2lerobot_abs_Q_x_episode.py`

* 按 episode 维度切分并转换为 LeRobot
* 适用于 episode 独立训练 / 评估场景

### `lerobot_check.py`

* 校验生成的 LeRobot 数据集格式与字段完整性
* 防止训练阶段因字段缺失或格式错误报错

---

## 📦 其他工具脚本

### `lerobot2hf.py`

* 将 LeRobot 数据集转换 / 上传为 HuggingFace Dataset 格式

### `hf_update_data.py`

* 对已有 HuggingFace 数据集进行增量更新

### `single_process.py`

* 单进程处理版本
* 主要用于调试或小规模数据验证

### `start.bash`

* 数据处理流水线启动脚本
* 用于串联多个步骤的一键执行（未完成）

---

