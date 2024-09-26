import subprocess
import time
import numpy as np

class ffmpegReader(object):
    def __init__(self, url, size):
        self.url = url
        self.process = None
        # self.size = (1920, 1080)
        # self.size = (2560, 1440)
        self.size = size


        # assert input_fps in [30, 60]

        # self.skip_current = False

        self._start_ffmpeg_video()


    def _start_ffmpeg_video(self):
        cmd = [
            'ffmpeg',
            '-re',
            '-live_start_index', '0',
            '-protocol_whitelist', 'tcp,file,http,crypto,data',
            '-i', self.url,
            # '-playlist_start_number', '0', 
            '-r', "30",
            '-f', 'image2pipe',
            '-pix_fmt', 'bgr24',
            '-vcodec', 'rawvideo', '-'
        ]
        self.process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    
    def start_ffmpeg_audio(self, audio_path, delay_seconds=0):
        print("starting ffmpeg audio", audio_path, delay_seconds)
        if delay_seconds < 0 :
            delay = abs(delay_seconds) * 1000
            cmd = [
                'ffmpeg',
                '-protocol_whitelist', 'tcp,file,http,crypto,data',
                '-i', self.url,
                '-af', f'adelay={delay}|{delay}',
                '-acodec', 'aac',  # 输出 PCM 格式
                '-loglevel', 'quiet',
                '-y',
                audio_path
            ]
        else:
            cmd = [
                'ffmpeg',
                '-ss', f'{delay_seconds}',
                '-protocol_whitelist', 'tcp,file,http,crypto,data',
                '-i', self.url,
                '-acodec', 'aac',  # 输出 PCM 格式
                '-loglevel', 'quiet',
                '-y',
                audio_path
            ]
        print(f'ffmpeg cmd: {cmd}')
        self.audio_process = subprocess.Popen(cmd)


    def next(self):
        image = self._next()

        return image

    def _next(self):
        if self.process is None or self.process.poll() is not None:
            print("fuck")
            return None

        try:
            raw_image = self.process.stdout.read(self.size[0] * self.size[1] * 3)
            image = np.frombuffer(raw_image, dtype=np.uint8).reshape((self.size[1], self.size[0], 3))
            return image
        except Exception as e:
            print(f"Error reading frame: {e}")
            return None
        
    def stop(self):
        """停止FFmpeg进程"""
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            self.process.wait()
            self.process = None

def write_audio_video_to_stream(video_reader, audio_reader, output_path, video_size=(1920, 1080), fps=30):
    """
    将音视频数据逐帧写入新的文件中。

    :param video_reader: 视频读取器对象
    :param audio_reader: 音频读取器对象
    :param output_path: 输出文件路径
    :param video_size: 视频尺寸，默认为 (1920, 1080)
    :param fps: 帧率，默认为 30
    """

    # 使用 FFmpeg 合并音视频数据
    cmd = [
        'ffmpeg',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-s', f'1920x1080',  # 视频尺寸
        '-pix_fmt', 'bgr24',  # 指定像素格式
        '-r', 30,  # 帧率
        '-i', '-',  # 输入视频数据
        '-i', '-',  # 输入音频数据
        '-c:v', 'libx264',  # 视频编码器
        '-preset', 'ultrafast',  # 编码速度
        '-c:a', 'aac',  # 音频编码器
        '-strict', 'experimental',  # 允许实验性编码器
        '-b:a', '128k',  # 音频比特率
        output_path  # 输出文件路径
    ]

    process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    try:
        while True:
            image, audio = video_reader.next(), audio_reader.next()
            if image is None or audio is None:
                break

            # 写入视频帧
            process.stdin.write(image.tobytes())

            # 写入音频帧
            process.stdin.write(audio.tobytes())

    except Exception as e:
        print(f"Error writing frames: {e}")

    finally:
        # 关闭进程
        process.stdin.close()
        process.wait()

        # 停止 FFmpeg 进程
        video_reader.stop()
        audio_reader.stop()

if __name__ == "__main__":
    pass