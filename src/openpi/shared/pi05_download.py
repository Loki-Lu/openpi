import os
from pathlib import Path

# 1. 指定缓存根目录（非常关键）
os.environ["OPENPI_DATA_HOME"] = "/gemini-1/space/tong/models/jaxpi"

# 2. 如果你在 GCS 上需要关 TensorStore 文件锁（强烈建议）
os.environ["TENSORSTORE_CONTEXT"] = '{"file_io_locking": false}'

# 3. 引入你刚才那段代码里的函数
from openpi.shared.download import maybe_download  # 路径按你项目实际改

def main():
    url = "gs://openpi-assets/checkpoints/pi05_base/params"

    local_path = maybe_download(
        url,
        # 如果是公开 bucket，一般不需要额外参数
        # anon=True  # 视环境而定
    )

    print("✅ Download finished")
    print("Local path:", local_path)

    # 简单 sanity check
    assert Path(local_path).exists()

if __name__ == "__main__":
    main()
