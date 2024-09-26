from func_wrapper import sub_task, iteration_timer
from reader import ffmpegReader
import subprocess
from utils import get_and_init_stitcher, get_regist_imgs
from detector_model import BallDetector, PlayerDetector
from equirectangular_model import EquirectangularModel
from func_wrapper import sub_task, iteration_timer
from cameraman_model import CameramanModel
import numpy as np
import cv2
import time
from tqdm import tqdm
from multiprocessing import Pool
import os
from functools import partial
from image_transformer import ImageTransformer
from queue import Queue


class Drawer:
    def __init__(self):
        self.position = (10, 50)  # 起始位置
        self.line_spacing = 50    # 行间距
        self.text_color = (0, 0, 255)  # 文本颜色
        self.font_scale = 1       # 字体大小
        self.thickness = 1       # 字体粗细

    def draw_texts(self, img, texts):
        for i, text in enumerate(texts):
                cv2.putText(img, text, (self.position[0], self.position[1] + i * self.line_spacing), cv2.FONT_HERSHEY_SIMPLEX, 
                            self.font_scale, self.text_color, self.thickness, cv2.LINE_AA)


@sub_task
def read_stream(size, url, audio_path, shared_img_for_stitch, shared_img_for_det, stop_flag, direction, init_seconds=0, skip_seconds=0):
    """
    从左右两个M3U8直播流中读取画面，返回left, right的img，左右流默认已对齐
    """
    def skip():
        img = reader.next()
    def init():
        img = reader.next()
        for i in range(init_frames):
            local_img_queue.put(img)
    @iteration_timer(f"read_stream_{direction}", "get")
    def get():
        global img
        img = reader.next()
        
        if local_img_queue.qsize() < init_frames:
            for i in range(init_frames):
                local_img_queue.put(img) # 为了减少cameraman中滤波引起的延迟，展示的frame queue中预先多放几十帧，这样形成错位就实现了相当于用未来的检测结果来跟随当前帧
        local_img_queue.put(img)

    @iteration_timer(f"read_stream_{direction}", "put")    
    def put():
        if img is not None:
            shared_img_for_stitch.put(local_img_queue.get())
            shared_img_for_det.put(img)
        else:
            time.sleep(2)
            print(f"{direction} stream is empty")

    reader = ffmpegReader(url, size)
    init_frames = int(init_seconds * 30)
    skip_frames = int(skip_seconds * 30)
    skipped = 0
    local_img_queue = Queue()
    if audio_path is not None:
        delay_seconds = skip_seconds-init_seconds
        reader.start_ffmpeg_audio(audio_path, delay_seconds)
    # process = None
    while skipped < skip_frames:
        skip()
        skipped += 1
        if skipped % 100 == 0:
            print(f"{skipped} {direction} frames skipped")
    while not stop_flag.value:
        get()
        put()
    reader.stop()
    


@sub_task
def detect(shared_img_for_det, queue_detect_result, stop_flag, direction, points_config, dp_live_config):
    @iteration_timer(f"detect_{direction}", "get")
    def get():
        global img
        img = shared_img_for_det.get()
    @iteration_timer(f"detect_{direction}", "task")
    def task():
        global player_points, ball_points
        player_points = player_detector.predict(img, draw=False, crop_box=points_config[f"{direction}_crop_size"], direction=direction)
        player_points = player_detector.map_points(player_points, stitcher, direction=direction, court_points=points_config["polygon"])

        ball_points = ball_detector.predict(img, draw=False, crop_box=points_config[f"{direction}_crop_size"], direction=direction)
        ball_points = ball_detector.map_points(ball_points, stitcher, direction=direction, court_points=points_config["polygon"])

    @iteration_timer(f"detect_{direction}", "put")
    def put():
        queue_detect_result.put((player_points, ball_points))
        # process.stdin.write(img.tobytes())
    
    ball_detector = BallDetector(direction)
    player_detector = PlayerDetector(direction)
    stitcher = get_and_init_stitcher(dp_live_config["device_id"])
    # cmd = [
    #     'ffmpeg',
    #     '-y',  # 覆盖输出文件
    #     '-f', 'rawvideo',
    #     '-vcodec', 'rawvideo',
    #     '-pix_fmt', 'bgr24',
    #     '-s', f'1920x1080',  # 图像尺寸
    #     '-r', '30',  # 帧率
    #     '-i', '-',  # 输入来自标准输入
    #     '-c:v', 'libx264',
    #     '-pix_fmt', 'yuv420p',
    #     '-preset', 'fast',
    #     '-f', 'flv',
    #     # '-loglevel', 'debug',
    #     f"rtmp://jushoop-live-videos.oss-cn-shanghai.aliyuncs.com/live/DanTAP7ePdK?playlistName=playlist.m3u8&OSSAccessKeyId=LTAI5tK6wdonDpPs4q3dTp5p&Expires=1726248430&Signature=gQQh9wYaipbZFqMJ4nlQsbu2%2B2k%3D"
    # ]

    # process = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    while not stop_flag.value:
        get()
        task()
        put()

@sub_task
def cameraman(points_config, queue_detect_result_left, queue_detect_result_right, queue_camera_man_result, queue_camera_man_result_left, queue_camera_man_result_right, stop_flag, debug=False):
    @iteration_timer("cameraman", "get")
    def get():
        global left_res, right_res
        left_res = queue_detect_result_left.get()
        right_res = queue_detect_result_right.get()


    @iteration_timer("cameraman", "task")
    def task():
        global result, target_x
        player_points_left, ball_points_left = left_res
        player_points_right, ball_points_right = right_res
        
        player_points = player_points_left + player_points_right
        ball_points = ball_points_left + ball_points_right
        result = cameraman_model.predict(player_points, ball_points, debug_mode=debug)
        target_x = result[1] if debug else result
        

    @iteration_timer("cameraman", "put")
    def put():
        queue_camera_man_result.put(result)
        queue_camera_man_result_left.put(target_x)
        queue_camera_man_result_right.put(target_x)

    cameraman_model = CameramanModel(points_config)
    while not stop_flag.value:
        get()
        task()
        put()


@sub_task
def warp_frame(points_config, dp_live_config, shared_img_for_warp, queue_camera_man_result, direction, shared_img_for_stitch, stop_flag):
    @iteration_timer(f"warp_frame_{direction}", "get")
    def get():
        global img, target_x
        img = shared_img_for_warp.get()
        # img = get_regist_imgs()[["left", "right"].index(direction)]
        target_x = queue_camera_man_result.get()

    @iteration_timer(f"warp_frame_{direction}", "task")
    def task():
        global result
        result = img_transformer.compute_img(img, target_x, direction)

    @iteration_timer(f"warp_frame_{direction}", "put")
    def put():
        shared_img_for_stitch.put(result)

    img_transformer = ImageTransformer(points_config, dp_live_config)
    while not stop_flag.value:
        get()
        task()
        put()

@sub_task
def transform_frame(points_config, dp_live_config, shared_img_for_stitch_left, shared_img_for_stitch_right, queue_camera_man_result, shared_img_for_push, stop_flag, debug=False):
    @iteration_timer("transform", "get")
    def get():
        global left_img, right_img, cameraman_result, target_x
        left_img = shared_img_for_stitch_left.get()
        right_img = shared_img_for_stitch_right.get()



        cameraman_result = queue_camera_man_result.get()
        target_x = cameraman_result[1] if debug else cameraman_result

    @iteration_timer("transform", "task")
    def task():
        global result, target_x

        result = img_transformer.transform(left_img, right_img, target_x)
        if debug:
            target_x, target_x_filtered, mean_player_pos, player_speed, player_focus, ball_focus, max_player_pos, min_player_pos, ball_detected, focus_slider, player_positions= cameraman_result
            texts = [f"target_x : {target_x}", f"target_x_filtered : {target_x_filtered}", f"mean_player_pos: {mean_player_pos}", f"player_speed : {player_speed}", f"player_focus : {player_focus}", f"ball_focus : {ball_focus}", f"max_player_pos : {max_player_pos}", f"min_player_pos : {min_player_pos}", f"ball_detected: {ball_detected}", f"focus_slider: {focus_slider}", f"player_positions: {player_positions}"]
            drawer.draw_texts(result, texts)

    
    @iteration_timer("transform", "put")
    def put():
        shared_img_for_push.put(result)
    
    drawer = Drawer()
    img_transformer = ImageTransformer(points_config, dp_live_config)

    while not stop_flag.value:
        get()
        task()
        put()
    

# @sub_task
# def transform_frame(points_config, shared_img_for_stitch_left, shared_img_for_stitch_right, queue_detect_result_left, queue_detect_result_right, shared_img_for_push, stop_flag, debug=False):
#     @iteration_timer("transform", "get")
#     def get():
#         global left_img, right_img, left_res, right_res
#         left_img = shared_img_for_stitch_left.get()
#         right_img = shared_img_for_stitch_right.get()

#         left_res = queue_detect_result_left.get()
#         right_res = queue_detect_result_right.get()
#     @iteration_timer("transform", "task")
#     def task():
#         global result
#         player_points_left, ball_points_left = left_res
#         player_points_right, ball_points_right = right_res
        
#         player_points = player_points_left + player_points_right
#         ball_points = ball_points_left + ball_points_right

#         if debug:
#             target_x, target_x_filtered, mean_player_pos, player_speed, player_focus, ball_focus, max_player_pos, min_player_pos = cameraman_model.predict(player_points, ball_points, debug_mode=True)
#         else:
#             target_x_filtered = cameraman_model.predict(player_points, ball_points)

#         result = img_transformer.transform(left_img, right_img, target_x_filtered)
#         if debug:
#             texts = [f"target_x : {target_x}", f"target_x_filtered : {target_x_filtered}", f"mean_player_pos: {mean_player_pos}", f"player_speed : {player_speed}", f"player_focus : {player_focus}", f"ball_focus : {ball_focus}", f"max_player_pos : {max_player_pos}", f"min_player_pos : {min_player_pos}"]
#             drawer.draw_texts(result, texts)
#     @iteration_timer("transform", "put")
#     def put():
#         shared_img_for_push.put(result)

#     img_transformer = ImageTransformer(points_config)
#     cameraman_model = CameramanModel(points_config)
#     drawer = Drawer()
#     while not stop_flag.value:
#         get()
#         task()
#         put()


@sub_task
def push_stream(size, audio_path, shared_img_for_push, target_url, stop_flag):
    @iteration_timer(f"push_stream", "get")
    def get():
        global img, audio
        img = shared_img_for_push.get()
    @iteration_timer(f"push_stream", "task")
    def task():
        process.stdin.write(img.tobytes())
        # process.stdin.write(audio.tobytes())

    # 启动 ffmpeg 进程
    cmd = [
        'ffmpeg',
        '-y',  # 覆盖输出文件
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-pix_fmt', 'bgr24',
        '-s', f'{size[1]}x{size[0]}',  # 图像尺寸
        '-r', '30',  # 帧率
        '-i', '-',  # 输入来自标准输入
        '-i', audio_path,  # 输入音频数据
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-preset', 'fast',
        '-c:a', 'aac',  # 音频编码器
        '-strict', 'experimental',  # 允许实验性编码器
        '-b:a', '128k',  # 音频比特率
        '-f', 'flv',
        # '-loglevel', 'debug',
        target_url
    ]

    process = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    # process = None

    try:
        while not stop_flag.value:
                get()
                task()
    except Exception as e:
        print(f"Error during stream pushing: {e}")
    finally:
        if process is not None:
            process.stdin.close()
            process.terminate()
            process.wait()
