# -- coding: utf-8 --
# @Time : 2025/8/29
# @Author : ykk648
from ultralytics import YOLO

face_det = YOLO('./checkpoints/yolov8n-face.pt', verbose=False)


def detect_face(face_img):
    # face_det = YOLO(f'{args.pretrained_model_dir}/yolov8n-face/yolov8n-face.pt')
    boxes = face_det(face_img,
                     imgsz=640,
                     conf=0.01,
                     iou=0.5,
                     half=False,  # cpu必须是False
                     augment=False,
                     verbose=False,
                     device='cuda')[0].boxes
    bboxes = boxes.xyxy.cpu().numpy()
    return bboxes
