# mask-rcnn
train mask-rcnn custom dataset

# requirements

- `miniconda`
- `python=3.6`
- `tensorflow 1.x`

# guides

`pip install --upgrade pip setuptools wheel`

`pip install -r Mask_RCNN/requirements.txt`

`cd Mask_RCNN && python setup.py install`

`python3 train.py train --dataset=dataset --weights=mask_rcnn_coco.h5 --logs logs`

# pretrain model

Download coco weights: `wget --quiet https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5`

# references

[Mask R-CNN for Object Detection and Segmentation](https://github.com/matterport/Mask_RCNN)

[Mask R-CNN for Object Detection and Segmentation using TensorFlow 2.0](https://github.com/ahmedfgad/Mask-RCNN-TF2)

https://github.com/matterport/Mask_RCNN/issues/363
