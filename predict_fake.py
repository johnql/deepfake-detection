# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 23:32:40 2020

@author: johnw
"""
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pathlib import Path
from facenet_pytorch import MTCNN
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from PIL import Image
from tqdm.notebook import tqdm
from time import time
import shutil
import datetime
import warnings
import glob
import os
import time
import json

from facenet_pytorch import MTCNN, InceptionResnetV1, extract_face
from efficientnet_pytorch import EfficientNet

os.sep='/'
os.path.sep='/'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Running on device: {device}')

def confident_strategy(pred, t=0.8):
    pred = np.array(pred)
    sz = len(pred)
    fakes = np.count_nonzero(pred > t)
    # 11 frames are detected as fakes with high probability
    if fakes > sz // 2.5 and fakes > 11:
        return np.mean(pred[pred > t])
    elif np.count_nonzero(pred < 0.2) > 0.9 * sz:
        return np.mean(pred[pred < 0.2])
    else:
        return np.mean(pred)


# Load facial recognition model
resnet = InceptionResnetV1(pretrained='vggface2', device=device).eval()

model = EfficientNet.from_pretrained('efficientnet-b7').eval()
# Unfreeze model weights
for param in model.parameters():
    param.requires_grad = True
num_ftrs = model._fc.in_features
model._fc = nn.Linear(num_ftrs, 1)
model = model.to('cpu')
optimizer = torch.optim.Adam(model.parameters())
loss_func = nn.BCELoss()


margin = 80
image_size = 250
input_size = 380
strategy = confident_strategy
test_videos = 'dataset/test_videos'
train_sample_videos = 'dataset/train_sample_videos'
train_videos = 'dataset/deepfake'
faces_path = 'dataset/faces'
batch_size = 60
scale = 0.25
n_frames = 32 # The number of frames extracted from each video, 'None' means get all available frames
stime = time.time()


# Load face detector
detector = MTCNN(keep_all=False, select_largest=False, post_process=False, device=device, min_face_size=100, margin=margin, image_size=image_size).eval()

def detect_facenet_pytorch(detector, images):
    faces = []

    faces.extend(detector(images))
    return faces
    
class DetectionPipeline:
    """Pipeline class for detecting faces in the frames of a video file."""
    
    def __init__(self, detector, n_frames, batch_size, resize=None):
        """Constructor for DetectionPipeline class.
        
        Keyword Arguments:
            n_frames {int} -- Total number of frames to load. These will be evenly spaced
                throughout the video. If not specified (i.e., None), all frames will be loaded.
                (default: {None})
            batch_size {int} -- Batch size to use with MTCNN face detector. (default: {32})
            resize {float} -- Fraction by which to resize frames from original prior to face
                detection. A value less than 1 results in downsampling and a value greater than
                1 result in upsampling. (default: {None})
        """
        self.detector = detector
        self.n_frames = n_frames
        self.batch_size = batch_size
        self.resize = resize
    
    def __call__(self, filename):
        """Load frames from an MP4 video and detect faces.

        Arguments:
            filename {str} -- Path to video.
        """
        # Create video reader and find length
        v_cap = cv2.VideoCapture(filename)
        v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        faces = []
        frames = []
        # Pick 'n_frames' evenly spaced frames to sample
        if self.n_frames is None:
            sample = np.arange(0, v_len)
        else:
            sample = np.linspace(0, v_len - 1, self.n_frames).astype(int)

        # Loop through frames
        if n_frames is None:
            sample = np.arange(0, v_len)
        else:
            sample = np.linspace(0, v_len - 1, n_frames).astype(int)
        for i in tqdm(sample):
            _, image = v_cap.read()
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            frames.append(image)
            #images_720_1280.append(cv2.resize(image, (1280, 720)))
            #images_540_960.append(cv2.resize(image, (960, 540)))
        v_cap.release()
        
        frames = np.stack(frames)
        #images_720_1280 = np.stack(images_720_1280)
        #images_540_960 = np.stack(images_540_960)
            
        
        
        
        

        v_cap.release()

        return detect_facenet_pytorch(self.detector, frames)




def plot_faces(images, figsize=(10.8/2, 19.2/2)):
    shape = images[0].shape
    images = images[np.linspace(0, len(images)-1, 16).astype(int)]
    im_plot = []
    for i in range(0, 16, 4):
        im_plot.append(np.concatenate(images[i:i+4], axis=0))
    im_plot = np.concatenate(im_plot, axis=1)
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(im_plot)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    ax.grid(False)
    fig.tight_layout()


def timer(detector, detect_fn, images, *args):
    start = time.time()
    #begin_time = datetime.datetime.now()
    faces = detect_fn(detector, images, *args)
    elapsed = time.time() - start
    #elapsed = datetime.datetime.now() - begin_time
    print(f', {elapsed:.3f} seconds')
    #print('elapsed:', elapsed )
    return faces, elapsed

    
# Source: https://www.kaggle.com/timesler/facial-recognition-model-in-pytorch
def process_faces(faces, feature_extractor):
    # Filter out frames without faces
    faces = [f for f in faces if f is not None]
    print('------', len(faces))
    if len(faces) == 0:
        return None
    
    
    # Generate facial feature vectors using a pretrained model
    from torchvision.transforms import ToTensor
    from torchvision import transforms
    faces = torch.Tensor(np.stack(faces))
    faces = torch.cat(faces).to(device)
    
    

    embeddings = feature_extractor(faces)

    # Calculate centroid for video and distance of each face's feature vector from centroid
    centroid = embeddings.mean(dim=0)
    x = (embeddings - centroid).norm(dim=1).cpu().numpy()
    
    return x

# Load face detector
#mtcnn = MTCNN(margin=margin, keep_all=True, factor=0.5, device=device).eval()

# Load facial recognition model
feature_extractor = InceptionResnetV1(pretrained='vggface2', device=device).eval()

# Define face detection pipeline
detection_pipeline = DetectionPipeline(detector=detector, n_frames=n_frames, batch_size=batch_size, resize=scale)

# Get the paths of all train videos
all_train_videos = glob.glob(os.path.join(train_sample_videos, '*.mp4'))[:5]

# Get path of metadata.json
metadata_path = train_sample_videos + '/metadata.json'

# Get metadata
#with open(metadata_path, 'r') as f:
#    metadata = json.load(f)
metadata= pd.read_json(metadata_path).transpose()
    
    
df = pd.DataFrame(columns=['filename', 'distance', 'predicted_label'])
tf_img = lambda i: ToTensor()(i).unsqueeze(0)
embeddings = lambda input: feature_extractor(input)

with torch.no_grad():
    for path in tqdm(all_train_videos):
        file_name = path.split('\\')[-1]

        # Detect all faces occur in the video
        faces = detection_pipeline(path)
        #plot_faces(torch.stack(faces).permute(0, 2, 3, 1).int().numpy())
        plot_faces(torch.stack(faces).permute(0, 2, 3, 1).int().numpy())
        # Calculate the distances of all faces' feature vectors to the centroid
        distances = process_faces(faces, feature_extractor)
        if distances is None:
            continue

        for distance in distances:
            row = [
                file_name,
                distance,
                1 if metadata[file_name]['label'] == 'FAKE' else 0
            ]

            # Append a new row at the end of the data frame
            df.loc[len(df)] = row
            
            
df.head()