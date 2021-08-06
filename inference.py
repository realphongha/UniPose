import torch
import cv2
import argparse
import numpy as np
import sys
import os
from time import sleep, time
from tqdm import tqdm
from imutils.video import FPS
from unipose import detect, get_model

parser = argparse.ArgumentParser(description="Run inference on videos")
parser.add_argument("--path", default="webcam", type=str, help="Path to video. Leave it empty to use webcam")
parser.add_argument("--ckpt", default="checkpoint_best.pth.tar", type=str, help="Path to pose pretrained model")
parser.add_argument("--ds", default="MPII", type=str, help="Pose model dataset")
parser.add_argument("--conf", default=0.25, type=int, help="Person detection confident")
parser.add_argument("--iou", default=0.45, type=int, help="Person detection IOU threshold")
parser.add_argument("--gpu", action="store_true", default=False, help="Use GPU or not?")
args = parser.parse_args()

person_detector = torch.hub.load('ultralytics/yolov5', 'yolov5s')
person_detector.conf = args.conf  # confidence threshold (0-1)
person_detector.iou = args.iou  # NMS IoU threshold (0-1)
person_detector.classes = [0] # person

pose_detector = get_model(args.ckpt, args.ds, not args.gpu)

if args.gpu:
    # person_detector = person_detector.cuda()
    pose_detector = pose_detector.cuda()

if args.path == "webcam":
    video_cap = cv2.VideoCapture(0)
    while True:
        if not video_cap.isOpened():
            print('Unable to load camera.')
            sleep(5)
            pass
    
        # Capture frame-by-frame
        ret, frame = video_cap.read()
        frame = cv2.flip(frame, 1)
        if args.gpu:
            results = person_detector(frame).xyxy[0].cpu().detach().numpy()
        else:
            results = person_detector(frame).xyxy[0].numpy()
        for res in results:
            x1, y1, x2, y2 = res[:-2]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cropped = frame[y1:y2, x1:x2]
            cropped = detect(cropped, pose_detector, args.ds, not args.gpu)
            cropped = cv2.resize(cropped, (x2-x1, y2-y1))
            frame[y1:y2, x1:x2] = cropped
            frame = cv2.rectangle(frame, (x1, y1, x2, y2), 
                (255, 0, 0), 2)
            break
        cv2.imshow('Video', frame)
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_cap.release()
    cv2.destroyAllWindows()
elif os.path.isdir(args.path):
    # directories of images
    save_path = "run/%i" % (int(time()))
    print("Saving results to %s" % save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    for fn in os.listdir(args.path):
        file = os.path.join(args.path, fn)
        new_filename = os.path.join(save_path, fn[:-4] + "_detected" + fn[-4:])
        image = cv2.imread(file)
        if args.gpu:
            results = person_detector(image).xyxy[0].cpu().detach().numpy()
        else:
            results = person_detector(image).xyxy[0].numpy()
        for res in results:
            x1, y1, x2, y2 = res[:-2]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cropped = image[y1:y2, x1:x2]
            cropped = detect(cropped, pose_detector, args.ds, not args.gpu)
            cropped = cv2.resize(cropped, (x2-x1, y2-y1))
            image[y1:y2, x1:x2] = cropped
            image = cv2.rectangle(image, (x1, y1, x2, y2), 
                (255, 0, 0), 2)
        cv2.imwrite(new_filename, image)
else:
    # video
    video_out = args.path[:-4] + '_detected' + args.path[-4:]
    video_reader = cv2.VideoCapture(args.path)

    nb_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_h = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_w = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = int(video_reader.get(cv2.CAP_PROP_FPS))
    
    video_writer = cv2.VideoWriter(video_out,
                            cv2.VideoWriter_fourcc(*'MPEG'), 
                            fps, 
                            (frame_w, frame_h))

    for i in tqdm(range(nb_frames)):
        _, frame = video_reader.read()
        if frame is None:
            continue
        if args.gpu:
            results = person_detector(frame).xyxy[0].cpu().detach().numpy()
        else:
            results = person_detector(frame).xyxy[0].numpy()
        for res in results:
            x1, y1, x2, y2 = res[:-2]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cropped = frame[y1:y2, x1:x2]
            cropped = detect(cropped, pose_detector, args.ds, not args.gpu)
            cropped = cv2.resize(cropped, (x2-x1, y2-y1))
            frame[y1:y2, x1:x2] = cropped
            frame = cv2.rectangle(frame, (x1, y1, x2, y2), 
                (255, 0, 0), 2)
        video_writer.write(np.uint8(frame))

    video_reader.release()
    video_writer.release()
