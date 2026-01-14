#远端服务器的用法
#autossh -M 0 -T -L 8000:127.0.0.1:8000 fcy@root@ssh-411.default@218.84.152.122 -p 30022


# ┌───────────────────────────────┐
# │          推理线程 (Inference)   │
# │ - 不停采图                     │
# │ - 调用 server 得到 50-step plan │
# │ - 用最新 plan 覆盖共享变量      │
# └───────────────┬───────────────┘
#                 │
#                 ▼
# ┌───────────────────────────────┐
# │       共享数据: latest_plan    │
# │  ActionPlan(actions, idx, ts) │
# └───────────────┬───────────────┘
#                 │
#                 ▼
# ┌───────────────────────────────┐
# │       控制线程 (Control)       │
# │ - 固定频率读取 latest_plan    │
# │ - 执行 plan.actions[idx]      │
# │ - idx += 1                     │
# │ - 新 plan 到来 idx 会重置      │
# └───────────────────────────────┘


# ============================================================
# Remote server usage:
# autossh -M 0 -T -L 8000:127.0.0.1:8000 fcy@root@ssh-411.default@218.84.152.122 -p 30022
# autossh -M 0 -T -L 8000:127.0.0.1:8000 dx@192.168.151.72 -p 22
# ============================================================


import sys
sys.path.insert(0, '/home/nvidia/r1pro_TeleAI')

import dataclasses
import logging
import threading
import time

import numpy as np
import cv2

from openpi_client import websocket_client_policy as _websocket_client_policy
from vla_planner import VLAPlanner
import tyro

# ============================================================
# Global config
# ============================================================

W, H = 1920, 1080
logger = logging.getLogger(__name__)

# ============================================================
# Camera utils
# ============================================================

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
        return np.zeros((H, W, 3), dtype=np.uint8)
    yuv = np.ascontiguousarray(raw).reshape(H * 3 // 2, W)
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)


# ============================================================
# CLI args
# ============================================================

@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int | None = 8000
    api_key: str | None = None
    control_hz: float = 30.0        # robot control frequency
    max_plan_age: float = 1.0       # seconds, plan timeout


# ============================================================
# Observation
# ============================================================

def _observation_umi(left_cam, right_cam, robot) -> dict:
    state = robot.get_act()

    left_img  = cv2.resize(grab_bgr(left_cam),  (224, 224))
    right_img = cv2.resize(grab_bgr(right_cam), (224, 224))
        # 显示图像
    cv2.imshow("Left Wrist", left_img)
    cv2.imshow("Right Wrist", right_img)
    # cv2.imshow("Front (zero)", front_img)
    cv2.waitKey(1)  # 非阻塞刷新
    return {
        "observation/state": state,
        "observation/images/left_wrist": left_img,
        "observation/images/right_wrist": right_img,
        "prompt": "Put the eggplant and bananas into the basket.",
    }


# ============================================================
# Action plan (50-step trajectory)
# ============================================================

@dataclasses.dataclass
class ActionPlan:
    actions: np.ndarray        # (T=50, D)
    idx: int = 0
    timestamp: float = 0.0


latest_plan: ActionPlan | None = None
plan_lock = threading.Lock()


# ============================================================
# Inference thread
# ============================================================

def inference_loop(policy, left_cam, right_cam, robot, stop_event):
    global latest_plan
    logger.info("Inference thread started")

    # Warmup
    for _ in range(2):
        obs = _observation_umi(left_cam, right_cam, robot)
        policy.infer(obs)

    while not stop_event.is_set():
        obs = _observation_umi(left_cam, right_cam, robot)
        result = policy.infer(obs)
        time.sleep(0.3)  # 避免过快循环
        plan = ActionPlan(
            actions=result["actions"],   # shape (50, D)
            idx=0,
            timestamp=time.time(),
        )

        # 覆盖旧 plan，idx 自动重置为 0
        with plan_lock:
            latest_plan = plan


# ============================================================
# Control thread
# ============================================================

def control_loop(robot, stop_event, control_hz, max_plan_age):
    global latest_plan
    dt = 1.0 / control_hz

    last_plan_timestamp = 0.0  # 记录上一次 plan 的 timestamp

    while not stop_event.is_set():
        plan = None

        # 获取最新 plan 并检查是否是新 plan
        with plan_lock:
            if latest_plan is not None:
                plan = latest_plan
                if plan.timestamp != last_plan_timestamp:
                    plan.idx = 0
                    last_plan_timestamp = plan.timestamp

        if plan is not None:
            age = time.time() - plan.timestamp
            if age <= max_plan_age:
                i = min(plan.idx, len(plan.actions) - 1)
                if i >=5:
                    
                    act = plan.actions[i]
                    print("action -->", i)
                    robot.go_vlatgt(act)
                    robot.set_tgt()

                # idx 自增
                with plan_lock:
                    if latest_plan is plan:
                        latest_plan.idx += 1  # 每次执行 5 steps
            else:
                # plan 过期 → hold last step 或安全停止
                pass

        time.sleep(dt)


# ============================================================
# Main
# ============================================================

def main(args: Args):
    logging.basicConfig(level=logging.INFO)

    # Cameras
    left_cam  = init_camera("/dev/video10")
    right_cam = init_camera("/dev/video8")

    # Robot
    robot = VLAPlanner()
    robot.init()

    # Policy
    policy = _websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
        api_key=args.api_key,
    )
    logger.info(f"Server metadata: {policy.get_server_metadata()}")

    stop_event = threading.Event()

    t_infer = threading.Thread(
        target=inference_loop,
        args=(policy, left_cam, right_cam, robot, stop_event),
        daemon=True,
    )

    t_ctrl = threading.Thread(
        target=control_loop,
        args=(robot, stop_event, args.control_hz, args.max_plan_age),
        daemon=True,
    )

    t_infer.start()
    t_ctrl.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Stopping...")
        stop_event.set()
        t_infer.join()
        t_ctrl.join()


if __name__ == "__main__":
    main(tyro.cli(Args))
