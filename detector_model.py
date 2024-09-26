import os
import sys
import cv2
import numpy as np
import math
# from ultralytics import YOLOv10, YOLO
from ultralytics import YOLO

# from roboflowoak import RoboflowOak

def mask_points(results, quad_points):
        """
        将检测结果中的点限制在一个四边形内部。
        :param results: 包含检测结果的点列表，每个点为 (x, y)
        :param quad_points: 四边形的四个顶点坐标 [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
        :return: 在四边形内部的点列表
        """
        filtered_results = []

        # 检查每个点是否在四边形内部
        for point in results:
            # print("________")
            # print(point, is_point_inside_polygon(point, quad_points))
            # print(quad_points)
            # print("________\n")
            if is_point_inside_polygon(point, quad_points):
                filtered_results.append(point)

        return filtered_results

def is_point_inside_polygon(point, polygon):
    """
    判断一个点是否在多边形内部。
    :param point: 点的坐标 (x, y)
    :param polygon: 多边形的顶点坐标 [(x1, y1), (x2, y2), ...]
    :return: 布尔值，表示点是否在多边形内部
    """
    n = len(polygon)
    inside = False

    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if point[1] > min(p1y, p2y):
            if point[1] <= max(p1y, p2y):
                if point[0] <= max(p1x, p2x):
                    if p1y != p2y:
                        xints = (point[1] - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or point[0] <= xints:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside

def draw_parsed_boxes(image, boxes, color=None):
        if color is None:
            color = (0, 0, 255)
        for box in boxes:
            cv2.circle(image, tuple(box), 10, (0, 0, 255), -1)

class BaseDetector(object):
    def __init__(self, direction=None, to_skip=0):
        self.direction = direction
        self.to_skip = to_skip
        self.skiped = to_skip
        self.last = None

    def predict(self, image, direction=None, draw=False, crop_box=None):
        if self.skiped < self.to_skip:
            self.skiped += 1
            return self.last
        else:
            self.skiped = 0

        orig_image = image
        if crop_box is not None:
            if direction is None:
                raise ValueError("direction must be specified when crop_box is specified")
            if direction == "left":
                image = image[crop_box[1]:, crop_box[0]:, :]
            else:
                image = image[crop_box[1]:, :crop_box[0], :]
        boxes = self._predict(image)
        if crop_box is not None:
            if direction == "left":
                boxes = [[box[0] + crop_box[0], box[1] + crop_box[1]] for box in boxes]
            else:
                boxes = [[box[0], box[1] + crop_box[1]] for box in boxes]
        if draw:
            draw_parsed_boxes(orig_image, boxes, color=(0, 0, 255))

        self.last = boxes
        
        return boxes
    
    def draw_detections(self, image, detections):
        for box in detections.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            conf = box.conf[0].item()
            cls_id = box.cls[0].item()
            class_name = self.model.names[int(cls_id)]

            # 绘制边界框
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 绘制标签
            label = f"{class_name}: {conf:.2f}"
            text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(image, (x1, y1 - text_size[1] - 5), (x1 + text_size[0], y1), (0, 255, 0), -1)
            cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    def predict_and_draw_detections(self, image):
        detections = self.model.predict(source=image, verbose=False)[0]
        for box in detections.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            conf = box.conf[0].item()
            cls_id = box.cls[0].item()
            class_name = self.model.names[int(cls_id)]

            # 绘制边界框
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 绘制标签
            label = f"{class_name}: {conf:.2f}"
            text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(image, (x1, y1 - text_size[1] - 5), (x1 + text_size[0], y1), (0, 255, 0), -1)
            cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        return image


    def map_points(self, results, stitcher, direction=None, court_points=None):
        
        direction = direction if direction is not None else self.direction

        if direction is None:
            raise Exception("direction is None")
            
        panorama_points = []
        for point in results:
            try:
                panorama_points.append(list(map(int, stitcher.map_point_to_parorama(point, direction, True))))
            except:
                continue

        if court_points is not None:
            panorama_points = mask_points(panorama_points, court_points)
        return panorama_points
    
    def draw_det(self, image):
        results = self._predict(image)


    
        



class PlayerDetector(BaseDetector):
    def __init__(self, direction=None):
        super(PlayerDetector, self).__init__(direction)
        # self.model = YOLOv10("yolov10m.pt")
        self.model = YOLO("weights/yolov8n.pt")


    def _predict(self, image):
        

        results = self.model.predict(source=image, verbose=False)  # save predictions as labels
        
        boxes = np.array(results[0].boxes.xywh.cpu())
        # boxes = [[int(boxes[i][0] - boxes[i][2] / 2), int(boxes[i][1] - boxes[i][3] / 2), int(boxes[i][0] + boxes[i][2] / 2), int(boxes[i][1] + boxes[i][3] / 2), confidences[i]] for i in range(len(boxes)) if not classes[i]]

        
        # print(boxes)
        classes = np.array(results[0].boxes.cls.cpu())

        boxes = [list(map(int, list([boxes[i][0], boxes[i][1] + boxes[i][3] / 2]))) for i in range(len(boxes)) if not classes[i]]  # only return people boxes of which id is 0
        
        # print(boxes)
        return boxes



class BallDetector(BaseDetector):
    def __init__(self, direction=None, threshold=0.6):
        super(BallDetector, self).__init__(direction)
        self.model = YOLO("weights/basketball.pt")
        self.threshold = threshold

    def _predict(self, image):
        results = self.model.predict(source=image, verbose=False)  # save predictions as labels
        boxes = np.array(results[0].boxes.xywh.cpu())
        confidences = np.array(results[0].boxes.conf.cpu())

        # print(boxes)
        """0: ball, 1: rim"""
        classes = np.array(results[0].boxes.cls.cpu())
        # print(classes)

        # print(boxes, confidences, classes)

        return [list(map(int, list(boxes[i][:2]))) for i in range(len(boxes)) if (not classes[i] and confidences[i] >= self.threshold)]  





# class BaseDetector(object):
#     def __init__(self, stitcher):
#         self.stitcher = stitcher

#     def predict(self, images, masks=None):
#         """
#         :param images: [left_image, right_image]
#                masks: [left_mask, right_mask]
#         :return: points on panorama --- [[x1,y1], [x2, y2]...]
#         """

#         panorama_points = []
#         results_left = self._predict(images[0])
#         results_right = self._predict(images[1])

#         # print(results_left, results_right)

#         if masks is not None:
#             results_left = mask_points(results_left, masks[0])
#             results_right = mask_points(results_right, masks[1])

#         for point in results_left:
#             try:
#                 panorama_points.append(list(map(int, self.stitcher.map_point_to_parorama(point, "left", True))))
#             except:
#                 continue

#         for point in results_right:
#             try:
#                 panorama_points.append(list(map(int, self.stitcher.map_point_to_parorama(point, "right", True))))
#             except:
#                 continue

#         return panorama_points

#     def draw_det(self, points, panorama):
#         half_width = 30
#         half_height = 60
#         for point in points:
#             draw_point = [[point[0] - half_width, point[1] - half_height],
#                            [point[0] + half_width, point[1] - half_height],
#                            [point[0] + half_width, point[1] + half_height],
#                            [point[0] - half_width, point[1] + half_height]]
#             panorama = cv2.polylines(panorama, [np.int32(draw_point)], True, (100, 100, 255))
#         return panorama

#     def draw_results(self, stitcher, left_path, right_path, panorama):
#         results_left = self.predict(left_path)
#         results_right = self.predict(right_path)
#         left_points = []
#         right_points = []
#         for preds in results_left["predictions"]:
#             half_width = preds["width"] // 2
#             half_height = preds["height"] // 2
#             x, y = preds["x"], preds["y"]
#             left_points.append((x, y+half_height))
#         # left_points = mask_points(left_points, self.left_mask)
#         for points in left_points:
#             points = stitcher.map_point_to_parorama(points, "left", True)
#             if not points is None:
#                 points = list(points)
#             else:
#                 continue
#             draw_points = [[points[0] - half_width, points[1] - half_height],
#                            [points[0] + half_width, points[1] - half_height],
#                            [points[0] + half_width, points[1] + half_height],
#                            [points[0] - half_width, points[1] + half_height]]
#             panorama = cv2.polylines(panorama, [np.int32(draw_points)], True, (100, 100, 255))

#         for preds in results_right["predictions"]:
#             half_width = preds["width"] // 2
#             half_height = preds["height"] // 2
#             x, y = preds["x"], preds["y"]
#             right_points.append((x, y))
#         # right_points = mask_points(right_points, self.right_mask)
#         for points in right_points:
#             points = stitcher.map_point_to_parorama(points, "right", True)
#             if not points is None:
#                 points = list(points)
#             else:
#                 continue
#             draw_points = [[points[0] - half_width, points[1] - half_height],
#                            [points[0] + half_width, points[1] - half_height],
#                            [points[0] + half_width, points[1] + half_height],
#                            [points[0] - half_width, points[1] + half_height]]
#             panorama = cv2.polylines(panorama, [np.int32(draw_points)], True, (200, 200, 255))
#         return panorama

#     def _predict(self, image):
#         """
#         输入某一边的图像，返回检测结果
#         :param
#         :return:
#         list of points [[x1, y1], [x2, y2]...]
#         """
#         raise NotImplementedError


# class BallDetector(BaseDetector):
#     def __init__(self, stitcher):
#         super(BallDetector, self).__init__(stitcher)
#         self.model = YOLO("weights/basketball.pt")

#     def _predict(self, image):
#         results = self.model.predict(source=image, verbose=False)  # save predictions as labels
#         boxes = np.array(results[0].boxes.xywh.cpu())
#         # print(boxes)
#         """0: ball, 1: rim"""
#         classes = np.array(results[0].boxes.cls.cpu())
#         # print(classes)

#         return [list(map(int, list(boxes[i][:2]))) for i in range(len(boxes)) if not classes[i]]  # only return people boxes of which id is 0

# class PlayerDetector(BaseDetector):
#     def __init__(self, stitcher):
#         super(PlayerDetector, self).__init__(stitcher)
#         self.model = YOLO("weights/yolov8n.pt")

#     def _predict(self, image):
#         results = self.model.predict(source=image, verbose=False)  # save predictions as labels
#         boxes = np.array(results[0].boxes.xywh.cpu())
#         classes = np.array(results[0].boxes.cls.cpu())

#         return [list(map(int, list(boxes[i][:2]))) for i in range(len(boxes)) if not classes[i]]  # only return people boxes of which id is 0

if __name__ == "__main__":
    from utils import get_and_init_stitcher, get_points_config
    import json
    # from dp_stitching.details_stitcher import DetailsStitcher
    # from tqdm import tqdm

    # # init Stitcher
    # stitcher = DetailsStitcher()
    # stitcher.regist_image("data/left/10.png", "data/right/10.png")
    # stitcher.initialize_default_camera()
    # stitcher.initialize_warp()

    # init Detector
    play_det = PlayerDetector()
    ball_det = BallDetector()
    dp_live_config = json.load(open("dp_live_config.json", "r"))
    match_id = dp_live_config["match_id"]
    device_id = dp_live_config["device_id"]
    points_config = json.load(open(f"{device_id}/{match_id}/points_config.json", "r"))

    stitcher = get_and_init_stitcher(device_id)
    
    img_left = cv2.imread(f"{device_id}/init_frame_left.png")
    img_right = cv2.imread(f"{device_id}/init_frame_right.png")
    # panorama = stitcher.stitch(img_left, img_right)


    # # boxes = ball_det.predict(img, draw=True)
    boxes_left = play_det.predict(img_left, draw=True, crop_box=points_config["left_crop_size"], direction="left")
    # boxes_left = play_det.map_points(boxes_left, stitcher, direction="left", court_points=points_config["polygon"])

    # # boxes = ball_det.predict(img, draw=True)
    boxes_right = play_det.predict(img_right, draw=True, crop_box=points_config["right_crop_size"], direction="right")
    # boxes_right = play_det.map_points(boxes_right, stitcher, direction="right", court_points=points_config["polygon"])

    # print(boxes_left, boxes_right)
    # # img = ball_det.predict_and_draw_detections(img)
    cv2.imwrite("test_left_det.png", img_left)
    cv2.imwrite("test_right_det.png", img_right)

    
    # draw_parsed_boxes(panorama, boxes_left + boxes_right)
    # cv2.imwrite("panorama1.png", panorama)


    # img_left = img_left[438:, 323:, :]
    # img_right = img_right[485:, 0:1234, :]
    # cv2.imwrite("test_left1.png", img_left)
    # cv2.imwrite("img_right1.png", img_right)

    boxes = ball_det.predict(img_left, draw=True, crop_box=points_config["left_crop_size"], direction="left")
    cv2.imwrite("test_left_ball.png", img_left)



    # print(boxes)


    # get court mask
    # left_mask = cv2.imread("left.jpg", 0)
    # right_mask = cv2.imread("right.jpg", 0)

    # print(ball_det._predict(cv2.imread("data/left/10.png")))
    # print(human_det._predict(cv2.imread("data/left/10.png")))

    # start testing

    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # out = cv2.VideoWriter("test_det_ball1.avi", fourcc, 25.0, (2930, 1274), True)
    # for i in tqdm(range(2200)):
    #     left, right = [cv2.imread(f"data/left/{i+1}.png"), cv2.imread(f"data/right/{i+1}.png")]
    #     panorama = stitcher.stitch(left, right)
    #     points = ball_det.predict([left, right], masks=[left_mask, right_mask])
    #     panorama = ball_det.draw_det(points, panorama)
    #     out.write(panorama)
    # out.release()



