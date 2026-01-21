import pandas as pd
from PIL import Image
from io import BytesIO
import os

file_path = "/gemini-1/space/tong/data/water_trash/lerobot/Loki0929/teleai_umi/data/chunk-000/episode_000000.parquet"
save_dir = "./images_left_wrist"
os.makedirs(save_dir, exist_ok=True)

df = pd.read_parquet(file_path)
print("Columns:", df.columns)

img_column = 'observation.images.left_wrist'

for i, img_data in enumerate(df[img_column]):
    pil_img = None

    try:
        if isinstance(img_data, dict):
            # dict -> bytes
            if 'bytes' in img_data:
                pil_img = Image.open(BytesIO(img_data['bytes']))
            else:
                print(f"第 {i} 张 dict 没有 'bytes' 字段，跳过")
                continue

        elif isinstance(img_data, bytes):
            # bytes -> PIL
            pil_img = Image.open(BytesIO(img_data))

        elif isinstance(img_data, str):
            # base64 -> bytes -> PIL
            import base64
            pil_img = Image.open(BytesIO(base64.b64decode(img_data)))

        elif isinstance(img_data, (list, tuple)):
            # 可能是 list -> 转 numpy -> PIL
            import numpy as np
            pil_img = Image.fromarray(np.array(img_data))

        else:
            print(f"无法处理第 {i} 张图片，类型: {type(img_data)}")
            continue

        # 保存图片
        pil_img.save(os.path.join(save_dir, f"left_wrist_{i:06d}.png"))

    except Exception as e:
        print(f"第 {i} 张图片处理失败: {e}")

print(f"保存完成，共 {len(df)} 张 left_wrist 图片，保存在 {save_dir}")
