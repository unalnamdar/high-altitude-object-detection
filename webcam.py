import argparse
import logging
import time
import json
import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from torchvision.ops import nms

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='0')
    parser.add_argument('--model', type=str, default='yolov8n.pt')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--conf_thres', type=float, default=0.25)
    parser.add_argument('--iou_thres', type=float, default=0.45)
    parser.add_argument('--save_video', action='store_true')
    parser.add_argument('--save_json', action='store_true')
    parser.add_argument('--output', type=str, default='output')
    return parser.parse_args()

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

def load_model(path, device):
    model = YOLO(path)
    model.to(device)
    return model

def preprocess(frame, device):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    arr = img.astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    return tensor

def postprocess(result, conf_thres, iou_thres):
    boxes = result.boxes.xyxy
    scores = result.boxes.conf
    classes = result.boxes.cls
    keep = nms(boxes, scores, iou_thres)
    detections = []
    for i in keep:
        if scores[i] < conf_thres:
            continue
        b = boxes[i].cpu().numpy()
        s = float(scores[i].cpu())
        c = int(classes[i].cpu())
        detections.append({'bbox': b.tolist(), 'score': s, 'class_id': c})
    return detections

def draw(frame, detections, names):
    for det in detections:
        x1, y1, x2, y2 = map(int, det['bbox'])
        label = f"{names[det['class_id']]} {det['score']:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    return frame

def main():
    args = parse_args()
    setup_logging()

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    # Orijinal frame boyutları
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cv2.namedWindow('Inference', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Inference',
                     int(width * 2),   # %200 genişlik
                     int(height * 2))  # %200 yükseklik

    writer = None
    if args.save_video:
        os.makedirs(args.output, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(
            os.path.join(args.output, 'output.mp4'),
            fourcc, 30.0,
            (width, height)
        )

    results_list = []
    model = load_model(args.model, args.device)
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Inference
        tensor = preprocess(frame, args.device)
        start = time.time()
        with torch.no_grad():
            res = model(tensor)[0]
        detections = postprocess(res, args.conf_thres, args.iou_thres)

        frame = cv2.flip(frame, 1)

        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            new_x1 = width - x2
            new_x2 = width - x1
            det['bbox'] = [new_x1, y1, new_x2, y2]

        frame = draw(frame, detections, model.names)

        # FPS bilgisini ekle
        fps = 1.0 / (time.time() - start)
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('Inference', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if writer:
            writer.write(frame)
        if args.save_json:
            results_list.append({'frame': frame_id, 'detections': detections})

        frame_id += 1

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    # JSON olarak kaydet
    if args.save_json:
        os.makedirs(args.output, exist_ok=True)
        with open(os.path.join(args.output, 'results.json'), 'w') as f:
            json.dump(results_list, f, indent=2)

if __name__ == '__main__':
    main()
