import os.path
import cv2
import torch
import torchvision
from deep_sort_realtime.deepsort_tracker import DeepSort

object_detector = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
tracker = DeepSort(max_age=1)
video_path = os.path.join("videos/Traffic.mp4")


def track_label(img, track_id, ltrb):
    ltrb = ltrb.astype(int)
    thickness = 1
    color = (0, 255, 0)
    cv2.rectangle(img, (ltrb[0], ltrb[1]), (ltrb[2], ltrb[3]), color, thickness)
    cv2.putText(
        img,
        "{}".format(track_id),
        (ltrb[0], ltrb[1]),
        cv2.FONT_HERSHEY_DUPLEX,
        0.4,
        color,
        1,
    )
    return img


def main():
    video = torchvision.io.VideoReader(video_path, stream="video")
    frame = next(video)
    _, frame_height, frame_width = frame["data"].shape
    fps = video.get_metadata()["video"]["fps"][0]

    out = cv2.VideoWriter(
        "videos/Traffic_output.avi",
        cv2.VideoWriter_fourcc("M", "J", "P", "G"),
        fps,
        (frame_width, frame_height),
    )

    video.seek(0)
    for frame in video:
        results = object_detector([frame["data"].numpy()])
        im = results.ims[0]
        list_xyxy = results.xyxy[0][results.xyxy[0][:, -1] == 2].tolist()

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
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb()
            im = track_label(im, track_id, ltrb)
        out.write(im)

        # cv2.imshow("img", im)
        # cv2.waitKey()
    out.release()


if __name__ == "__main__":
    main()
