#远端服务器的用法
#autossh -M 0 -T -L 8000:127.0.0.1:8000 fcy@root@ssh-411.default@218.84.152.122 -p 30022


# ============================================================
# Remote server usage:
# autossh -M 0 -T -L 8000:127.0.0.1:8000 tong@root@ssh-32.default@58.59.115.26 -p 30022
# autossh -M 0 -T -L 8000:127.0.0.1:8000 dx@192.168.156.158 -p 22
# ============================================================
import sys
sys.path.insert(0, '/home/nvidia/r1pro_TeleAI')
import threading
import time
import numpy as np
import collections
import logging
import cv2
from openpi_client import websocket_client_policy as _websocket_client_policy
from vla_planner import VLAPlanner

logger = logging.getLogger(__name__)
W, H = 1280, 1280

# ===============================
# Camera utils + multithread buffer
# ===============================
def init_camera(dev_path):
    cap = cv2.VideoCapture(dev_path, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开 {dev_path}")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YU12'))
    cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
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
    bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
    return bgr

class CameraBuffer:
    """线程安全摄像头缓冲，只保留最新一帧"""
    def __init__(self, dev_path):
        self.cap = init_camera(dev_path)
        self.lock = threading.Lock()
        self.frame = None
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.update_loop, daemon=True)
        self.thread.start()

    def update_loop(self):
        while not self.stop_event.is_set():
            frame = grab_bgr(self.cap)
            with self.lock:
                self.frame = frame
            time.sleep(0.001)

    def get_frame(self):
        with self.lock:
            if self.frame is None:
                return np.zeros((H, W, 3), dtype=np.uint8)
            return self.frame.copy()

    def stop(self):
        self.stop_event.set()
        self.thread.join()
        self.cap.release()

# ===============================
# Action Queue
# ===============================
class ActionQueue:
    def __init__(self, maxlen=1000):
        self.queue = collections.deque(maxlen=maxlen)
        self.lock = threading.Lock()

    def append_chunk(
        self,
        actions: np.ndarray,
        *,
        action_delay: int = 0,
        use_steps: int | None = None,
        clear_old: bool = False,
        obs_delay: int = 0,
    ):
        """
        actions: (T, D)

        action_delay:
            丢弃新 infer action 的前 N 步

        use_steps:
            delay 之后，真正使用多少步（None = 全用）

        clear_old:
            是否对旧队列做处理

        obs_delay:
            clear_old 时，保留旧队列最前面的 obs_delay 个 action
        """

        # ---------- 处理新 action ----------
        T = len(actions)
        start = min(action_delay, T)
        remaining = actions[start:]

        if use_steps is None:
            use_steps = len(remaining)
        use_steps = min(use_steps, len(remaining))

        new_actions = remaining[:use_steps]

        with self.lock:
            # ---------- 处理旧队列 ----------
            if clear_old:
                if obs_delay > 0 and len(self.queue) > 0:
                    keep = list(self.queue)[:obs_delay]
                    self.queue.clear()
                    for a in keep:
                        self.queue.append(a)
                else:
                    self.queue.clear()

            # ---------- append 新 action ----------
            for a in new_actions:
                self.queue.append(a)
    def pop_action(self):
        with self.lock:
            if self.queue:
                return self.queue.popleft()
            return None

    def size(self):
        with self.lock:
            return len(self.queue)
# ===============================
# Observation
# ===============================
def _observation_umi(left_cam_buf, right_cam_buf, robot):
    state = robot.get_act()

    left_img_full  = left_cam_buf.get_frame()   # (1080, 1920, 3)
    right_img_full = right_cam_buf.get_frame()

    # h, w, _ = left_img_full.shape
    # crop_size = 1080  # 正方形大小

    # # 裁剪横向中心 crop_size
    # start_x = (w - crop_size) // 2
    # end_x = start_x + crop_size

    # left_img_cropped  = left_img_full[:, start_x:end_x]
    # right_img_cropped = right_img_full[:, start_x:end_x]

    # print("Cropped shape:", left_img_cropped.shape)  # 应该是 (1080, 1080, 3)

    # 缩放到 224x224 给模型用
    left_img  = cv2.resize(left_img_full, (224, 224))
    right_img = cv2.resize(right_img_full, (224, 224))

    # 显示
    cv2.imshow("Left Wrist", left_img)
    cv2.imshow("Right Wrist", right_img)
    cv2.waitKey(1)

    return {
        "observation/state": state,
        "observation/images/left_wrist": left_img,
        "observation/images/right_wrist": right_img,
        "prompt": "Grasp the bottle into the trash bin.",
    }



# ===============================
# Inference thread
# ===============================

def inference_loop(policy, left_cam_buf, right_cam_buf, robot, action_queue, stop_event, use_steps, action_delay,chunk_size_threshold=0.0):
    logger.info("Inference thread started")

    # Warmup
    for _ in range(2):
        obs = _observation_umi(left_cam_buf, right_cam_buf, robot)
        policy.infer(obs)

    while not stop_event.is_set():
        if action_queue.size() / use_steps <= chunk_size_threshold:
            obs = _observation_umi(left_cam_buf, right_cam_buf, robot)
            result = policy.infer(obs)  # shape (50, D)
            action_queue.append_chunk(
                result["actions"],
                action_delay=action_delay,
                use_steps=use_steps,
                clear_old=True,   
                obs_delay=0,
            )
        else:
            time.sleep(0.01)

# ===============================
# Control loop
# ===============================
def control_loop(robot, action_queue, control_hz=20):
    dt = 1.0 / control_hz
    action = action_queue.pop_action()
    if action is not None:
        robot.go_vlatgt(action)
        robot.set_tgt()
    robot.sleep()
    time.sleep(dt)

# ===============================
# Main async pipeline
# ===============================
def main_async(policy, left_cam_buf, right_cam_buf, robot, num_steps=1000, control_hz=20, use_steps=30, action_delay=10):
    action_queue = ActionQueue(maxlen=500)
    stop_event = threading.Event()

    # Start inference thread
    infer_thread = threading.Thread(
        target=inference_loop,
        args=(policy, left_cam_buf, right_cam_buf, robot, action_queue, stop_event, use_steps, action_delay),
        daemon=True,
    )
    infer_thread.start()

    logger.info("Starting control loop...")
    try:
        for _ in range(num_steps):
            control_loop(robot, action_queue, control_hz)
    except KeyboardInterrupt:
        logger.info("Stopping...")
    finally:
        stop_event.set()
        infer_thread.join()
        left_cam_buf.stop()
        right_cam_buf.stop()

# ===============================
# Entry point
# ===============================
if __name__ == "__main__":
    import tyro
    import dataclasses
    import pathlib

    @dataclasses.dataclass
    class Args:
        host: str = "0.0.0.0"
        port: int | None = 8000
        api_key: str | None = None
        num_steps: int = 20000
        control_hz: int = 20
        use_steps: int = 50
        action_delay: int = 0

    args = tyro.cli(Args)

    # Initialize camera buffers
    left_cam_buf  = CameraBuffer("/dev/video10")
    right_cam_buf = CameraBuffer("/dev/video8")

    # Initialize robot
    robot_xht = VLAPlanner()
    robot_xht.init()

    # Initialize policy client
    policy = _websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
        api_key=args.api_key,
    )
    logger.info(f"Server metadata: {policy.get_server_metadata()}")

    main_async(policy, left_cam_buf, right_cam_buf, robot_xht, num_steps=args.num_steps, control_hz=args.control_hz, use_steps=args.use_steps, action_delay=args.action_delay)
