

How to install:

```commandline
conda install pytorch torchvision cpuonly -c pytorch
pip install -r requirements.txt
```

How to train:

```commandline
python yolov5/train.py --img 640 --batch 16 --epochs 3 --data coco128.yaml --weights yolov5s.pt --cache
```

