'''
python3 predict.py dataset/pred/img4.jpeg
'''
import sys
import cv2

# from mrcnn import utils
from mrcnn import visualize
# from mrcnn.visualize import display_images
# from mrcnn.visualize import display_instances
# from mrcnn.model import log
from mrcnn.config import Config
from mrcnn import model as modellib

MODEL_PATH = "logs/object20230615T1514"
WEIGHTS_PATH = "logs/object20230615T1514/mask_rcnn_object_0001.h5"
IMG_PATH =  sys.argv[1] #"dataset/pred/img4.jpeg"

class_names = ['BG', 'baseball', 'tennis']

class CustomConfig(Config):
    """Configuration for training on the custom  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "object"

    IMAGES_PER_GPU = 1

    NUM_CLASSES = 1 + 2  # Background + Car and truck

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 10

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

# CREATE CONFIG    
config = CustomConfig()

#LOAD MODEL. Create model in inference mode
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_PATH, config=config)
model.load_weights(WEIGHTS_PATH, by_name=True)

# LOAD IMAGE
img = cv2.imread(IMG_PATH)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# PREDICT
result = model.detect([img], verbose=1)

# Visualize results
r = result[0]
visualize.display_instances(img, r['rois'], r['masks'], r['class_ids'], 
                            class_names, r['scores'])
