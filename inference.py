import torch
import cv2
import argparse
import numpy as np
import sys
from time import sleep
from tqdm import tqdm
from imutils.video import FPS
from unipose import detect, get_model

parser = argparse.ArgumentParser(description="Run inference on videos")
parser.add_argument("--path", default="webcam", type=str, help="Path to video. Leave it empty to use webcam")
parser.add_argument("--gpu", action="store_true", default=False, help="Use GPU or not?")
args = parser.parse_args()

person_detector = torch.hub.load('ultralytics/yolov5', 'yolov5s')
pose_detector = get_model("checkpoint_best.pth.tar")

if args.path == "webcam":
    video_cap = cv2.VideoCapture(0)
    fps = FPS().start()

    while True:
        if not video_cap.isOpened():
            print('Unable to load camera.')
            sleep(5)
            pass
    
        # Capture frame-by-frame
        ret, frame = video_cap.read()
        frame = cv2.flip(frame, 1)
        results = person_detector(frame).xyxy[0].numpy()
        cropped = None
        for res in results:
            if res[-1] == 0:
                x1, y1, x2, y2 = res[:-2]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                # frame = cv2.rectangle(frame, (x1, y1, x2, y2), 
                #     (255, 0, 0), 2)
                cropped = frame[y1:y2, x1:x2]
                break
        
        
        # Display the resulting frame
        if cropped is not None:
            cropped = detect(cropped, pose_detector)
            cropped = cv2.resize(cropped, (x2-x1, y2-y1))
            frame[y1:y2, x1:x2] = cropped
        cv2.imshow('Video', frame)
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
        # Display the resulting frame
        # cv2.imshow('Video', frame)
        fps = FPS().start()

    fps.stop()
    video_cap.release()
    cv2.destroyAllWindows()

else:
    video_out = args.path[:-4] + '_detected' + args.path[-4:]
    video_reader = cv2.VideoCapture(args.path)

    nb_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_h = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_w = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))

    video_writer = cv2.VideoWriter(video_out,
                            cv2.VideoWriter_fourcc(*'MPEG'), 
                            50.0, 
                            (frame_w, frame_h))

    for i in tqdm(range(nb_frames)):
        if i > 20:
            break
        _, frame = video_reader.read()
        
        # frame = cv2.flip(frame, 1)
        results = person_detector(frame).xyxy[0].numpy()
        cropped = None
        for res in results:
            if res[-1] == 0:
                x1, y1, x2, y2 = res[:-2]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                # frame = cv2.rectangle(frame, (x1, y1, x2, y2), 
                #     (255, 0, 0), 2)
                cropped = frame[y1:y2, x1:x2]
                cropped = detect(cropped, pose_detector)
                cropped = cv2.resize(cropped, (x2-x1, y2-y1))
                frame[y1:y2, x1:x2] = cropped
        video_writer.write(np.uint8(frame))
        fps = FPS().start()

    video_reader.release()
    video_writer.release()