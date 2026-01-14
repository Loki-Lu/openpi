#远端服务器的用法
#autossh -M 0 -T -L 8000:127.0.0.1:8000 fcy@root@ssh-411.default@218.84.152.122 -p 30022
import sys
sys.path.insert(0, '/home/nvidia/r1pro_TeleAI')

import dataclasses
import enum
import logging
import pathlib
import time

import numpy as np
from openpi_client import websocket_client_policy as _websocket_client_policy
import polars as pl
import rich
import tqdm
import tyro
from vla_planner import VLAPlanner
import cv2
import numpy as np
import math

W, H = 1920, 1080

def init_camera(dev_path):
    cap = cv2.VideoCapture(dev_path, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开 {dev_path}")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YU12'))
    cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    cap.set(cv2.CAP_PROP_FPS, 60)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    return cap

def grab_bgr(cap):
    ok, raw = cap.read()
    if not ok:
        return np.zeros((H, W, 3), dtype=np.uint8)  # 读取失败返回全黑
    yuv = np.ascontiguousarray(raw).reshape(H * 3 // 2, W)
    bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
    return bgr

logger = logging.getLogger(__name__)


class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"
    UMI = "umi"


@dataclasses.dataclass
class Args:
    """Command line arguments."""

    # Host and port to connect to the server.
    host: str = "0.0.0.0"
    # Port to connect to the server. If None, the server will use the default port.
    port: int | None = 8000
    # API key to use for the server.
    api_key: str | None = None
    # Number of steps to run the policy for.
    num_steps: int = 200
    # Path to save the timings to a parquet file. (e.g., timing.parquet)
    timing_file: pathlib.Path | None = None
    # Environment to run the policy in.
    # env: EnvMode = EnvMode.ALOHA_SIM


class TimingRecorder:
    """Records timing measurements for different keys."""

    def __init__(self) -> None:
        self._timings: dict[str, list[float]] = {}

    def record(self, key: str, time_ms: float) -> None:
        """Record a timing measurement for the given key."""
        if key not in self._timings:
            self._timings[key] = []
        self._timings[key].append(time_ms)

    def get_stats(self, key: str) -> dict[str, float]:
        """Get statistics for the given key."""
        times = self._timings[key]
        return {
            "mean": float(np.mean(times)),
            "std": float(np.std(times)),
            "p25": float(np.quantile(times, 0.25)),
            "p50": float(np.quantile(times, 0.50)),
            "p75": float(np.quantile(times, 0.75)),
            "p90": float(np.quantile(times, 0.90)),
            "p95": float(np.quantile(times, 0.95)),
            "p99": float(np.quantile(times, 0.99)),
        }

    def print_all_stats(self) -> None:
        """Print statistics for all keys in a concise format."""

        table = rich.table.Table(
            title="[bold blue]Timing Statistics[/bold blue]",
            show_header=True,
            header_style="bold white",
            border_style="blue",
            title_justify="center",
        )

        # Add metric column with custom styling
        table.add_column("Metric", style="cyan", justify="left", no_wrap=True)

        # Add statistical columns with consistent styling
        stat_columns = [
            ("Mean", "yellow", "mean"),
            ("Std", "yellow", "std"),
            ("P25", "magenta", "p25"),
            ("P50", "magenta", "p50"),
            ("P75", "magenta", "p75"),
            ("P90", "magenta", "p90"),
            ("P95", "magenta", "p95"),
            ("P99", "magenta", "p99"),
        ]

        for name, style, _ in stat_columns:
            table.add_column(name, justify="right", style=style, no_wrap=True)

        # Add rows for each metric with formatted values
        for key in sorted(self._timings.keys()):
            stats = self.get_stats(key)
            values = [f"{stats[key]:.1f}" for _, _, key in stat_columns]
            table.add_row(key, *values)

        # Print with custom console settings
        console = rich.console.Console(width=None, highlight=True)
        console.print(table)

    def write_parquet(self, path: pathlib.Path) -> None:
        """Save the timings to a parquet file."""
        logger.info(f"Writing timings to {path}")
        frame = pl.DataFrame(self._timings)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(path)


def main(args: Args) -> None:
    # Initialize cameras
    left_cam = init_camera("/dev/video10")
    right_cam = init_camera("/dev/video8")
    # Initialize robot
    robot_xht = VLAPlanner()
    robot_xht.init()
    # state_list=robot_xht.get_act()
    # print(state_list)
    # for i in range(1000): #假设1000帧轨迹
    #     #这里执行policy，得到action_list=policy(state_list)
    #     state_list=robot_xht.get_act()
    #     action_list=[0.0+0.1*math.sin(0.5*math.pi*i/90),0.0,0.0,0.0,0.0,0.0,1.0,80.0,
    #               0.0,0.0,0.0+0.1*math.sin(0.5*math.pi*i/90),0.0,0.0,0.0,1.0,80.0] #假设的轨迹数据
    #     robot_xht.go_vlatgt(action_list)
    #     robot_xht.set_tgt()
    #     robot_xht.sleep()
    # print(state_list)

    # obs_fn = {
    #     EnvMode.ALOHA: _random_observation_aloha,
    #     EnvMode.ALOHA_SIM: _random_observation_aloha,
    #     EnvMode.DROID: _random_observation_droid,
    #     EnvMode.LIBERO: _random_observation_libero,
    #     EnvMode.UMI: _random_observation_umi,
    # }[args.env]



    policy = _websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
        api_key=args.api_key,
    )
    logger.info(f"Server metadata: {policy.get_server_metadata()}")

    # Send a few observations to make sure the model is loaded.
    for _ in range(2):
        obs = _observation_umi(right_cam=right_cam, left_cam=left_cam, robot=robot_xht)
        policy.infer(obs)

    timing_recorder = TimingRecorder()

    for _ in tqdm.trange(args.num_steps, desc="Running policy"):
        inference_start = time.time()
        # Generate a observation.
        obs = _observation_umi(right_cam=right_cam, left_cam=left_cam, robot=robot_xht)
        # Get action from the policy.
        action = policy.infer(obs)
        print("action_list:", action["actions"][0], action["actions"].shape)
        for i, val in enumerate(action["actions"]):
            if i >=2 and i<=40:
                print(f"action[{i}]: {val}")
                # robot_xht.go_vlatgt(val)
                # robot_xht.set_tgt()
                # robot_xht.sleep()
                print(robot_xht.get_act())

        timing_recorder.record("client_infer_ms", 1000 * (time.time() - inference_start))
        for key, value in action.get("server_timing", {}).items():
            timing_recorder.record(f"server_{key}", value)
        for key, value in action.get("policy_timing", {}).items():
            timing_recorder.record(f"policy_{key}", value)

    timing_recorder.print_all_stats()

    if args.timing_file is not None:
        timing_recorder.write_parquet(args.timing_file)


# def _random_observation_aloha() -> dict:
#     return {
#         "state": np.ones((14,)),
#         "images": {
#             "cam_high": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
#             "cam_low": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
#             "cam_left_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
#             "cam_right_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
#         },
#         "prompt": "do something",
#     }


# def _random_observation_droid() -> dict:
#     return {
#         "observation/exterior_image_1_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
#         "observation/wrist_image_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
#         "observation/joint_position": np.random.rand(7),
#         "observation/gripper_position": np.random.rand(1),
#         "prompt": "do something",
#     }


# def _random_observation_libero() -> dict:
#     return {
#         "observation/state": np.random.rand(8),
#         "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
#         "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
#         "prompt": "do something",
#     }

def _observation_umi(left_cam, right_cam, robot) -> dict:
    # time.sleep(0.1)  # 模拟采集延时
    state = robot.get_act()
    print("state_list:", state)

    # 采集摄像头图像
    left_bgr = grab_bgr(left_cam)
    right_bgr = grab_bgr(right_cam)
    front_bgr = np.zeros_like(left_bgr)  # 1080p全零图像

    # resize到 224x224
    left_img  = cv2.resize(left_bgr, (224, 224))
    right_img = cv2.resize(right_bgr, (224, 224))
    # front_img = cv2.resize(front_bgr, (224, 224))
    # 显示图像
    # cv2.imshow("Left Wrist", left_img)
    # cv2.imshow("Right Wrist", right_img)
    # # cv2.imshow("Front (zero)", front_img)
    # cv2.waitKey(1)  # 非阻塞刷新
    return {
        "observation/state": state,
        # "observation/images/front": front_img,
        "observation/images/left_wrist": left_img,
        "observation/images/right_wrist": right_img,
        "prompt": "Put the eggplant and bananas into the basket.",
    }



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(tyro.cli(Args))
