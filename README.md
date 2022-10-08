

How to install:

```commandline
git clone git@github.com:binhnq94/object_tracking_test.git
git submodule init 
git submodule update

conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
pip install -r requirements.txt
```

How to train:

```commandline
python yolov5/train.py --img 640 --batch 16 --epochs 3 --data coco128.yaml --weights yolov5s.pt --cache
```

Run MQTT client:

```commandline
python mqtt_client.py
```

Run object tracking:

```commandline
 python tracking.py videos/Traffic.mp4
```