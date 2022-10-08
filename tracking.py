import os.path
import time

import cv2
import torch
import torchvision
from deep_sort_realtime.deepsort_tracker import DeepSort
import paho.mqtt.publish as publish


object_detector = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
tracker = DeepSort(max_age=10)
# video_path = os.path.join("videos/Traffic.mp4")

CAR_INDEX = 2


hostname = "broker.hivemq.com"
port = 1883
topic = "traffic_tracking"


def track_label(img, track_id, ltrb):
    ltrb = ltrb.astype(int)
    thickness = 2
    color = (0, 255, 0)
    cv2.rectangle(img, (ltrb[0], ltrb[1]), (ltrb[2], ltrb[3]), color, thickness)
    cv2.putText(
        img,
        "{}".format(track_id),
        (ltrb[0], ltrb[1]),
        cv2.FONT_HERSHEY_DUPLEX,
        1,
        color,
        1,
    )
    return img


def main(video_path):
    cap = cv2.VideoCapture(video_path)
    if (cap.isOpened() == False):
        print("Error opening video stream or file")

    frame_count = 0
    # Read until video is completed
    while (cap.isOpened()):
        frame_count += 1
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == False:
            break
        else:
            results = object_detector([frame])
            im = results.ims[0]
            list_xyxy = results.xyxy[0][results.xyxy[0][:, -1] == CAR_INDEX].tolist()

            detections = []
            for xyxy in list_xyxy:
                detections.append(
                    (
                        [xyxy[0], xyxy[1], xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]],
                        xyxy[4],
                        xyxy[5],
                    )
                )

            tracks = tracker.update_tracks(detections, frame=im)
            count_confirmed = 0
            for track in tracks:
                if not track.is_confirmed():
                    continue
                track_id = track.track_id
                ltrb = track.to_ltrb()
                im = track_label(im, track_id, ltrb)
                count_confirmed += 1

            # Publish message
            publish.single(
                topic, f"Number of car: {count_confirmed}", hostname=hostname, port=port
            )
            resized = cv2.resize(im, (int(im.shape[1]/3*2), int(im.shape[0]/3*2)), interpolation=cv2.INTER_AREA)
            if frame_count % 2 == 0:
                cv2.imshow("Video", resized)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break


if __name__ == "__main__":
    import sys
    main(sys.argv[1])
