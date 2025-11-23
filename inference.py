import argparse
import datetime
import json
import os
import random

import cv2
from ultralytics import YOLO

from mavlink_communication import MAVLinkCommunication


def parse_args():
    parser = argparse.ArgumentParser(description='UAV YOLO model inference')
    parser.add_argument('--path_to_model_w', type=str, required=True, help='model weight path')
    parser.add_argument('--from_cam', action='store_true', help='capture stream from camera')
    parser.add_argument('--cam_num', type=int, default=0, help='camera device number')
    parser.add_argument('--input_video_path', type=str, default=None, help='path to input video')
    parser.add_argument('--show_video', action='store_true', help='Show processing video')
    parser.add_argument('--save_video', action='store_true', help='save processed video')
    parser.add_argument('--save_logs', action='store_true', help='save logs or not')
    parser.add_argument('--output_video_path', type=str, default="output_video.mp4", help='path to output video')
    parser.add_argument('--input_dir', type=str, default=None, help='path to input directory with files')
    parser.add_argument('--mav_port', type=str, default=None, help='mavlink port')
    args = parser.parse_args()
    if sum(bool(x) for x in [args.from_cam, args.input_video_path, args.input_dir]) > 1:
        parser.error("Use exactly one of --from_cam, --input_video_path or --input_dir.")
    if args.save_video and args.output_video_path == 'output_video.mp4':
        print("[INFO] Output video will be saved to default path: ./output_video.mp4")
    return args


def log_detected_objects(model, results):
    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
    classes = results[0].boxes.cls.cpu().numpy().astype(int)
    conf = results[0].boxes.conf.cpu().numpy().astype(float)
    objects = []
    for box, clss, conf in zip(boxes, classes, conf):
        objects.append({
            "class": model.names[int(clss)],
            "confidence": conf.item(),
            "bounding_box": {
                "x1": box[0].item(),
                "y1": box[1].item(),
                "x2": box[2].item(),
                "y2": box[3].item()
            }
        })
    return objects


def draw_bounding_boxes(model, frame, results):
    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
    classes = results[0].boxes.cls.cpu().numpy().astype(int)

    for box, clss in zip(boxes, classes):
        random.seed(int(clss) + 8)
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3],), color, 2)
        cv2.putText(
            frame,
            f"{model.model.names[clss]}",
            (box[0], box[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (50, 255, 50),
            2,
        )
    return frame


def detect_dir_files(path_to_model_w, path_to_dir):
    logs_dir = "./logs"
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    model = YOLO(path_to_model_w)
    for filename in os.listdir(path_to_dir):
        file_path = os.path.join(path_to_dir, filename)
        result = model.predict(file_path, save=True, verbose=False)
        log_filename = filename.rsplit('.', 1)[0] + "_detected_objects.json"
        log_filepath = os.path.join(logs_dir, log_filename)
        with open(log_filepath, 'w+', encoding='utf-8') as file:
            json.dump(log_detected_objects(model, result), file, ensure_ascii=False, indent=4)

        print(f"[INFO] Лог сохранён: {log_filepath}")


def model_validation(path_to_model_w, path_to_data):
    model = YOLO(path_to_model_w)
    metrics = model.val(data=path_to_data)
    print(model.names)
    print(metrics.box.map, metrics.box.map50, metrics.box.map75, metrics.box.maps)


def process_video_with_detect(path_to_model_w, input_video_path, from_cam=False, cam_num=0, show_video=False,
                              save_video=False,
                              save_logs=False, output_video_path="output_video.mp4", mav_port='COM11'):
    model = YOLO(path_to_model_w)
    model.fuse()

    mav = MAVLinkCommunication(port=mav_port)

    result_json = []

    # Open the input video file
    if from_cam:
        cap = cv2.VideoCapture(cam_num)
    else:
        cap = cv2.VideoCapture(input_video_path)

    if not cap.isOpened():
        raise Exception("Error: Could not open video file.")

    # Get input video frame rate and dimensions
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the output video writer
    if save_video:
        day = datetime.date.today()
        time = datetime.datetime.now().strftime("%H-%M")
        filename, ext = os.path.splitext(output_video_path)
        output_video_path = f"{filename}_{day}_{time}{ext}"

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model.predict(frame, iou=0.65, conf=0.570, imgsz=640, verbose=False, )  # half=True

        if results and results[0].boxes is not None and len(results[0].boxes) > 0:
            classes = results[0].boxes.cls.cpu().numpy().astype(int)
            class_names = [model.names[c] for c in classes]
            detection_count = len(class_names)

            print("Внимание, обнаружен объект!")

            mav.send_detection_alert(detection_count=detection_count, class_names=class_names)

        if results[0].boxes is not None:
            draw_bounding_boxes(model, frame, results)
            result_json.append({
                "detected_objects": log_detected_objects(model, results)
            })

        if save_video:
            out.write(frame)

        if save_logs:
            with open('detected_objects.json', 'w+', encoding='utf-8') as file:
                json.dump(result_json, file, ensure_ascii=False, indent=4)

        if show_video:
            frame = cv2.resize(frame, (0, 0), fx=0.8, fy=0.8)
            cv2.imshow("frame", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if save_video:
        out.release()

    mav.close()
    cv2.destroyAllWindows()


def main():
    args = parse_args()
    if args.input_dir:
        detect_dir_files(args.path_to_model_w, args.input_dir)
    else:
        process_video_with_detect(args.path_to_model_w, args.input_video_path, args.from_cam, args.cam_num,
                                  args.show_video, args.save_video, args.save_logs, args.output_video_path,
                                  args.mav_port)


if __name__ == '__main__':
    main()
