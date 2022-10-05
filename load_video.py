import os
import os.path

import cv2
import torch
import torchvision

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
video_path = os.path.join('videos/Traffic.mp4')

stream = "video"

video = torchvision.io.VideoReader(video_path, stream)
print(video.get_metadata())

for frame in video:
    # Inference
    results = model([frame['data'].numpy()])

    # cv2.imshow('img', results.ims[0])
    # cv2.waitKey()

    # Results
    results.print()
    # results.save()  # or .show()
    results.show()

    xyxy_pd = results.pandas().xyxy[0]

    xyxy_pd_car = xyxy_pd[xyxy_pd['name'] == 'car']