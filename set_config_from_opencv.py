from utils import get_points_config, get_and_init_stitcher, get_pull_url
import json
from equirectangular_model import EquirectangularModel
import cv2
import subprocess


# 主函数
def set_perspective_param(panorama_path, name="test"):
    # 打开图片
    img = cv2.imread(panorama_path, cv2.IMREAD_COLOR)

    # 创建实例
    e = EquirectangularModel(size=img.shape[:2])

    # 设置初始位置
    
    x_value = 1250
    y_value = 500
    fov = 34
    if name == "left":
        x_value = 936
        y_value = 390
        fov = 32
    elif name == "middle":
        x_value = 1250
        y_value = 500
        fov = 36
    elif name == "right":
        x_value = 1621
        y_value = 386
        fov = 32

    # 鼠标事件处理
    dragging = False
    start_x, start_y = 0, 0

    # 显示初始图片
    persp = e.GetPerspectiveFromCoord(img, x_value, y_value, fov)
    cv2.imshow(f"{name}", persp)


    def mouse_event(event, x, y, flags, param):
        nonlocal x_value, y_value, fov, dragging, start_x, start_y

        if event == cv2.EVENT_LBUTTONDOWN:
            # 鼠标按下
            dragging = True
            start_x, start_y = x, y

        elif event == cv2.EVENT_MOUSEMOVE:
            # 鼠标移动
            if dragging:
                delta_x = x - start_x
                delta_y = y - start_y
                x_value -= delta_x
                y_value -= delta_y
                start_x, start_y = x, y
                fov = e.fov_func(x_value)
                persp = e.GetPerspectiveFromCoord(img, x_value, y_value, fov)
                cv2.imshow(f"{name}", persp)

        elif event == cv2.EVENT_LBUTTONUP:
            # 鼠标释放
            dragging = False

    cv2.setMouseCallback(f"{name}", mouse_event)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('='):
            print(1)
            fov -= 1
            fov = max(1, min(90, fov))
            persp = e.GetPerspectiveFromCoordStupid(img, x_value, y_value, fov)
            cv2.imshow(f"{name}", persp)
        elif key == ord('-'):
            fov += 1
            fov = max(1, min(90, fov))
            persp = e.GetPerspectiveFromCoordStupid(img, x_value, y_value, fov)
            cv2.imshow(f"{name}", persp)
        elif key == ord('w'):
            y_value -= 5
            persp = e.GetPerspectiveFromCoordStupid(img, x_value, y_value, fov)
            cv2.imshow(f"{name}", persp)
        elif key == ord('s'):
            y_value += 5
            persp = e.GetPerspectiveFromCoordStupid(img, x_value, y_value, fov)
            cv2.imshow(f"{name}", persp)
        elif key == ord('a'):
            x_value -= 5
            persp = e.GetPerspectiveFromCoordStupid(img, x_value, y_value, fov)
            cv2.imshow(f"{name}", persp)
        elif key == ord('d'):
            x_value += 5
            persp = e.GetPerspectiveFromCoordStupid(img, x_value, y_value, fov)
            cv2.imshow(f"{name}", persp)
            

    cv2.destroyAllWindows()

    return (int(x_value), int(y_value), int(fov))




def set_court_points(panorama_path):
    img = cv2.imread(panorama_path, cv2.IMREAD_COLOR)
     # 创建实例
    e = EquirectangularModel(size=img.shape[:2])

    # 设置初始位置
    x_value = 1250
    y_value = 500
    fov = 34

    # 显示初始图片
    cv2.imshow("Image", img)

    # 鼠标事件处理
    dragging = False
    start_x, start_y = 0, 0
    points = []

    def mouse_event(event, x, y, flags, param):
        nonlocal x_value, y_value, fov, dragging, start_x, start_y, points, img

        if event == cv2.EVENT_LBUTTONDOWN:
            # 鼠标按下
            if len(points) < 6:
                points.append((x, y))
                cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
                cv2.imshow("Image", img)
                # print(f"Point added at ({x}, {y})")

    cv2.setMouseCallback("Image", mouse_event)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == 27:  # ESC键
            if points:
                last_point = points.pop()
                cv2.circle(img, last_point, 5, (0, 0, 0), -1)  # 撤销上一个点
                cv2.imshow("Image", img)
                # print(f"Last point removed at {last_point}")

    cv2.destroyAllWindows()
    return points


def extract_frame_at_second(video_path, output_path, second=10):
    """
    使用ffmpeg从视频中提取指定秒数的帧并保存为PNG格式。
    
    :param video_path: 视频文件的路径
    :param output_path: 输出图片的路径
    :param second: 要提取的帧所在的秒数，默认为10秒
    """
    # 构造ffmpeg命令
    ffmpeg_command = [
        'ffmpeg',
        '-live_start_index', '0',
        '-protocol_whitelist', 'tcp,file,http,crypto,data',
        '-i', video_path,  # 输入文件
        '-ss', str(second),  # 开始时间
        '-vframes', '1',  # 只获取一帧
        '-y',
        output_path  # 输出文件
    ]
    
    # 执行命令
    try:
        subprocess.run(ffmpeg_command, check=True)
        print(f"成功从视频 {video_path} 的第 {second} 秒提取帧并保存为 {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"执行ffmpeg命令时发生错误: {e}")

def set_config_from_opencv():

    target_second = 10
    dp_live_config = json.load(open("dp_live_config.json", "r"))
    match_id = dp_live_config["match_id"]
    device_id = dp_live_config["device_id"]

    stitcher = get_and_init_stitcher(device_id=device_id)

    output_left = f'{device_id}/init_frame_left.png'
    output_right = f'{device_id}/init_frame_right.png'
    output_panorama = f'{device_id}/init_panorama.png'

    left_url = get_pull_url(f"{device_id}_{match_id}_left")
    right_url = get_pull_url(f"{device_id}_{match_id}_right")

    extract_frame_at_second(left_url, output_left, target_second)
    extract_frame_at_second(right_url, output_right, target_second)

    left = cv2.imread(output_left, cv2.IMREAD_COLOR)
    right = cv2.imread(output_right, cv2.IMREAD_COLOR)
    res = stitcher.stitch(left, right)
    cv2.imwrite(output_panorama, res)


    points = set_court_points(output_panorama)
    left_most_setting = set_perspective_param(output_panorama, "left")
    middle_point_setting = set_perspective_param(output_panorama, "middle")
    right_most_setting = set_perspective_param(output_panorama, "right")

    print(points, left_most_setting, middle_point_setting, right_most_setting)
    points_config = get_points_config(dp_live_config, points, left_most_setting, middle_point_setting, right_most_setting)
    json.dump(points_config, open(f"{device_id}/{match_id}/points_config.json", "w"))

    


if __name__ == '__main__':

    # 示例使用
    set_config_from_opencv()

    # match_id = "DanTAP7ePdK"


    # # points = [(685, 397), (215, 567), (1200, 419), (1305, 793), (1672, 417), (2121, 574)]
    
    # # left_most_setting = (910, 410, 37)
    # # middle_point_setting = (1250, 490, 42)
    # # right_most_setting = (1600, 440, 35)

    # points = [(699, 185), (87, 500), (1283, 151), (1048, 984), (1218, 190), (2542, 514)]
    # left_most_setting = (870, 360, 34)
    # middle_point_setting = (1350, 485, 48)
    # right_most_setting = (1750, 355, 36)

    # points_config = get_points_config(points, left_most_setting, middle_point_setting, right_most_setting)

    # json.dump(points_config, open("points_config.json", "w"))