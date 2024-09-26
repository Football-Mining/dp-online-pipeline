import cv2
import time
from functools import wraps
import multiprocessing as mp
from shared_image_manager import SharedImageQueue, SharedImage
from sub_tasks import *

from oss import OSSClient
import json
from set_config_from_opencv import set_config_from_opencv
from utils import get_pull_url


INPUT_STREAM_LEFT_TEST = "http://111.229.130.191/api/get-oss-video-stream/unokGdEwXiH/playlist.m3u8"
INPUT_STREAM_RIGHT_TEST = "http://111.229.130.191:80/api/get-oss-video-stream/MyJE4YRjSbu/playlist.m3u8"
OUTPUT_STREAM_TEST = "rtmp://jushoop-live-videos.oss-cn-shanghai.aliyuncs.com/live/MKYpKkqwnqE?playlistName=playlist.m3u8&OSSAccessKeyId=LTAI5tK6wdonDpPs4q3dTp5p&Expires=1725878612&Signature=xHwXrZxgti5zfEyVfgUVf8wQNDk%3D"

SHAPE = (1080, 1920, 3)

LOCAL_TEST_PATH_LEFT = "left.mp4"
LOCAL_TEST_PATH_RIGHT = "right.mp4"

def create_and_start_process(pool, target, args):
    p = mp.Process(target=target, args=args)
    p.start()
    pool.append(p)
    return p


def start_tasks(img_size, input_urls, output_url, audio_path, points_config, dp_live_config, debug=False):

    shared_img_for_stitch_left = SharedImage(shape=img_size)
    shared_img_for_stitch_right = SharedImage(shape=img_size)
    shared_img_for_warp_left = SharedImage(shape=img_size)
    shared_img_for_warp_right = SharedImage(shape=img_size)
    shared_img_for_det_left = SharedImage(shape=img_size)
    shared_img_for_det_right = SharedImage(shape=img_size)
    shared_img_for_push = SharedImage(shape=img_size, debug=True)

    queue_detect_result_left = mp.Queue(maxsize=50)
    queue_detect_result_right = mp.Queue(maxsize=50)

    queue_cameraman_result = mp.Queue(maxsize=50)
    queue_cameraman_result_left = mp.Queue(maxsize=50)
    queue_cameraman_result_right = mp.Queue(maxsize=50)

    stop_flag = mp.Value('b', False)

    # 创建并启动读取左右流的进程
    # read_process = mp.Process(target=read_stream, args=(INPUT_STREAM_LEFT_TEST, INPUT_STREAM_RIGHT_TEST, queue_origin_images_for_stitch, stop_flag))
    # read_process.start()

    process_pool = []

    read_process_left = create_and_start_process(pool=process_pool, target=read_stream, args=(img_size, input_urls[0], None, shared_img_for_warp_left, shared_img_for_det_left, stop_flag, "left", dp_live_config["init_frame_seconds"], dp_live_config["skip_frame_seconds"]))
    read_process_right = create_and_start_process(pool=process_pool, target=read_stream, args=(img_size, input_urls[1], audio_path, shared_img_for_warp_right, shared_img_for_det_right, stop_flag, "right", dp_live_config["init_frame_seconds"], dp_live_config["skip_frame_seconds"]))
    camera_man_process = create_and_start_process(pool=process_pool, target=cameraman, args=(points_config, queue_detect_result_left, queue_detect_result_right, queue_cameraman_result, queue_cameraman_result_left, queue_cameraman_result_right, stop_flag, debug))
    warp_process_left = create_and_start_process(pool=process_pool, target=warp_frame, args=(points_config, dp_live_config, shared_img_for_warp_left, queue_cameraman_result_left, "left", shared_img_for_stitch_left, stop_flag))
    warp_process_right = create_and_start_process(pool=process_pool, target=warp_frame, args=(points_config, dp_live_config, shared_img_for_warp_right, queue_cameraman_result_right, "right", shared_img_for_stitch_right, stop_flag))

    transform_process = create_and_start_process(pool=process_pool, target=transform_frame, args=(points_config, dp_live_config, shared_img_for_stitch_left, shared_img_for_stitch_right, queue_cameraman_result, shared_img_for_push, stop_flag, debug))
    detect_process_left = create_and_start_process(pool=process_pool, target=detect, args=(shared_img_for_det_left, queue_detect_result_left, stop_flag, 'left', points_config, dp_live_config))
    detect_process_right = create_and_start_process(pool=process_pool, target=detect, args=(shared_img_for_det_right, queue_detect_result_right, stop_flag, 'right', points_config, dp_live_config))
    # # cameraman_process = create_and_start_process(pool=process_pool, target=cameraman, args=(queue_stitched_images, queue_detect_result_left, queue_detect_result_right, queue_target_images, stop_flag))
    
    # time.sleep(10)
    push_process = create_and_start_process(pool=process_pool, target=push_stream, args=(img_size, audio_path, shared_img_for_push, output_url, stop_flag))


    # 主进程中处理队列中的帧
    try:
        while True:
            time.sleep(0.1)
    finally:
        print("clearing....")
        # 清理
        cv2.destroyAllWindows()
        stop_flag.value = True  # 设置共享变量以通知子进程退出
        time.sleep(3)
        for p in process_pool:
            p.terminate()
        print("cleared....")

def calc_mat():
    dp_live_config = json.load(open("dp_live_config.json", "r"))
    match_id = dp_live_config["match_id"]
    device_id = dp_live_config["device_id"]
    points_config = json.load(open(f"{device_id}/{match_id}/points_config.json", "r"))

    img_transformer = ImageTransformer(points_config, dp_live_config)
    img_transformer.precalculate()

def reset_pull():
    dp_live_config = json.load(open("dp_live_config.json", "r"))
    match_id = dp_live_config["match_id"]
    device_id = dp_live_config["device_id"]
    
    left_channel = f"{device_id}_{match_id}_left"
    right_channel = f"{device_id}_{match_id}_right"

    oss_client = OSSClient()

    oss_client.delete_folder(left_channel)
    oss_client.delete_folder(right_channel)

def main():
    dp_live_config = json.load(open("dp_live_config.json", "r"))
    match_id = dp_live_config["match_id"]
    device_id = dp_live_config["device_id"]
    
    left_channel = f"{device_id}_{match_id}_left"
    right_channel = f"{device_id}_{match_id}_right"

    points_config = json.load(open(f"{device_id}/{match_id}/points_config.json", "r"))

    oss_client = OSSClient()

    oss_client.delete_folder(match_id)

    left_pull_url = get_pull_url(left_channel)
    right_pull_url = get_pull_url(right_channel)
    # left_pull_url = "left_playlist.m3u8"
    # right_pull_url = "right_playlist.m3u8"

    audio_path = f"{device_id}/{match_id}/audio.aac"

    push_url = oss_client.create_live_channel(f"{match_id}")[0]

    start_tasks(SHAPE, (left_pull_url, right_pull_url), push_url, audio_path, points_config, dp_live_config, debug=dp_live_config["debug"])



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run Example')
    parser.add_argument('task', type=str, help='task')
    args = parser.parse_args()

    if args.task == "set":
        set_config_from_opencv()
    elif args.task == "calc":
        calc_mat()
    elif args.task == "run":
        main()
    elif args.task == "reset_pull":
        reset_pull()
    else:
        print("unknown task")