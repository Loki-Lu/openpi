import subprocess
import json

def get_fps(path):
    cmd = [
        'ffprobe', '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=r_frame_rate',
        '-of', 'json', path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    info = json.loads(result.stdout)
    num, den = info['streams'][0]['r_frame_rate'].split('/')
    return float(num) / float(den)

print(get_fps("/gemini/user/private/organize_small/session_001/left_hand_250801DR48FP25002269/RGB_Images/video.mp4"))
