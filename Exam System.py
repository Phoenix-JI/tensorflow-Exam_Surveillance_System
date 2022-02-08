#!/usr/bin/env python
# coding: utf-8

# In[ ]:


pip install opencv-python


# In[10]:


import dlib
import numpy as np
import tensorflow as tf
import cv2


# In[11]:


def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)
    
#     mou = []
    
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
#         if (i==62|i==63|i==64|i=66|i=67|i=68):
#             mou.append(i)
    return coords

predictor = dlib.shape_predictor('/Users/phoenixji/Desktop/Individual Project/shape_predictor_68_face_landmarks.dat')


# In[12]:


import struct
import numpy as np
from keras.layers import Conv2D
from keras.layers import Input
from keras.layers import BatchNormalization
from keras.layers import LeakyReLU
from keras.layers import ZeroPadding2D
from keras.layers import UpSampling2D
from keras.layers.merge import add, concatenate
from keras.models import Model


# In[13]:


from numpy import expand_dims
from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from matplotlib import pyplot
from matplotlib.patches import Rectangle


# In[14]:


model = load_model('model.h5')


# In[15]:


class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, objness = None, classes = None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.objness = objness
        self.classes = classes
        self.label = -1
        self.score = -1

    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)

        return self.label
 
    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.get_label()]

        return self.score


# In[16]:


def _sigmoid(x):
    return 1. / (1. + np.exp(-x))
 
def decode_netout(netout, anchors, obj_thresh, net_h, net_w):
    grid_h, grid_w = netout.shape[:2]
    nb_box = 3
    netout = netout.reshape((grid_h, grid_w, nb_box, -1))
    nb_class = netout.shape[-1] - 5
    boxes = []
    netout[..., :2]  = _sigmoid(netout[..., :2])
    netout[..., 4:]  = _sigmoid(netout[..., 4:])
    netout[..., 5:]  = netout[..., 4][..., np.newaxis] * netout[..., 5:]
    netout[..., 5:] *= netout[..., 5:] > obj_thresh

    for i in range(grid_h*grid_w):
        row = i / grid_w
        col = i % grid_w
        for b in range(nb_box):
            # 4th element is objectness score
            objectness = netout[int(row)][int(col)][b][4]
            if(objectness.all() <= obj_thresh): continue
            # first 4 elements are x, y, w, and h
            x, y, w, h = netout[int(row)][int(col)][b][:4]
            x = (col + x) / grid_w # center position, unit: image width
            y = (row + y) / grid_h # center position, unit: image height
            w = anchors[2 * b + 0] * np.exp(w) / net_w # unit: image width
            h = anchors[2 * b + 1] * np.exp(h) / net_h # unit: image height
            # last elements are class probabilities
            classes = netout[int(row)][col][b][5:]
            box = BoundBox(x-w/2, y-h/2, x+w/2, y+h/2, objectness, classes)
            boxes.append(box)
    return boxes
 
def correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w):
    new_w, new_h = net_w, net_h
    for i in range(len(boxes)):
        x_offset, x_scale = (net_w - new_w)/2./net_w, float(new_w)/net_w
        y_offset, y_scale = (net_h - new_h)/2./net_h, float(new_h)/net_h
        boxes[i].xmin = int((boxes[i].xmin - x_offset) / x_scale * image_w)
        boxes[i].xmax = int((boxes[i].xmax - x_offset) / x_scale * image_w)
        boxes[i].ymin = int((boxes[i].ymin - y_offset) / y_scale * image_h)
        boxes[i].ymax = int((boxes[i].ymax - y_offset) / y_scale * image_h)
 
def _interval_overlap(interval_a, interval_b):
    
    x1, x2 = interval_a
    x3, x4 = interval_b
    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2,x4) - x1
    else:
        if x2 < x3:
             return 0
        else:
            return min(x2,x4) - x3
 
def bbox_iou(box1, box2):
    intersect_w = _interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
    intersect_h = _interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])
    intersect = intersect_w * intersect_h
    w1, h1 = box1.xmax-box1.xmin, box1.ymax-box1.ymin
    w2, h2 = box2.xmax-box2.xmin, box2.ymax-box2.ymin
    union = w1*h1 + w2*h2 - intersect
    return float(intersect) / union
 
def do_nms(boxes, nms_thresh):
    if len(boxes) > 0:
        nb_class = len(boxes[0].classes)
    else:
        return
    for c in range(nb_class):
        sorted_indices = np.argsort([-box.classes[c] for box in boxes])
        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]
            if boxes[index_i].classes[c] == 0: continue
            for j in range(i+1, len(sorted_indices)):
                index_j = sorted_indices[j]
                if bbox_iou(boxes[index_i], boxes[index_j]) >= nms_thresh:
                    boxes[index_j].classes[c] = 0
 
# load and prepare an image
def load_image_pixels(filename, shape):
    # load the image to get its shape
    image = load_img(filename)
    width, height = image.size
    # load the image with the required size
    image = load_img(filename, target_size=shape)
    # convert to numpy array
    image = img_to_array(image)
    # scale pixel values to [0, 1]
    image = image.astype('float32')
    image /= 255.0
    # add a dimension so that we have one sample
    image = expand_dims(image, 0)
    return image, width, height
 
# get all of the results above a threshold
def get_boxes(boxes, labels, thresh):
    v_boxes, v_labels, v_scores = list(), list(), list()
    # enumerate all boxes
    for box in boxes:
        # enumerate all possible labels
        for i in range(len(labels)):
            # check if the threshold for this label is high enough
            if box.classes[i] > thresh:
                v_boxes.append(box)
                v_labels.append(labels[i])
                v_scores.append(box.classes[i]*100)
                # don't break, many labels may trigger for one box
    return v_boxes, v_labels, v_scores
 
# draw all results
def draw_boxes(filename, v_boxes, v_labels, v_scores):
    # load the image
    data = pyplot.imread(filename)
    # plot the image
    pyplot.imshow(data)
    # get the context for drawing boxes
    ax = pyplot.gca()
    # plot each box
    for i in range(len(v_boxes)):
        box = v_boxes[i]
        # get coordinates
        y1, x1, y2, x2 = box.ymin, box.xmin, box.ymax, box.xmax
        # calculate width and height of the box
        width, height = x2 - x1, y2 - y1
        # create the shape
        rect = Rectangle((x1, y1), width, height, fill=False, color='blue')
        # draw the box
        ax.add_patch(rect)
        # draw text and score in top left corner
        label = "%s (%.3f)" % (v_labels[i], v_scores[i])
        pyplot.text(x1, y1, label, color='white')
    # show the plot
    pyplot.show()


# In[17]:


labels = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]


# In[18]:


def getImage(frame,class_threshold=0.4,label = labels):
    
    #image = cv2.imread(photo_filename)
    #image.shape
    img = cv2.resize(frame,(416,416))
    #img.shape
    img = img.astype('float32')
    img /= 255.0
    img = expand_dims(img, 0)
    #img.shape

    yhat = model.predict(img)
    #print([a.shape for a in yhat])
    
    anchors = [[116,90, 156,198, 373,326], [30,61, 62,45, 59,119], [10,13, 16,30, 33,23]]
    
    
    input_w, input_h = 416, 416
    image_h, image_w = frame.shape[0],frame.shape[1]
    
    boxes = list()
    for i in range(len(yhat)):
        # decode the output of the network
        boxes += decode_netout(yhat[i][0], anchors[i], class_threshold, input_h, input_w)
        
    #correct_yolo_boxes(boxes, image_h, image_w, input_h, input_w)
    
    v_boxes, v_labels, v_scores = get_boxes(boxes, labels, class_threshold)
    
    #for i in range(len(v_boxes)):
        #print(v_labels[i], v_scores[i])
        
    if 'cell phone' in v_labels:
        #print('Yes')
        return 'yes'
    else:
        #print('No')
        return 'No'


# In[19]:


#photo_filename = 'Test_1.png'


# In[20]:


#class_threshold = 0.4


# In[21]:


#image = cv2.imread(photo_filename)


# In[22]:


#getImage(photo_filename,class_threshold=0.4,label = labels)


# In[28]:


cap = cv2.VideoCapture('/Users/phoenixji/Desktop/Demo Final.mp4')


# In[ ]:



detector = dlib.get_frontal_face_detector()

mou_frequency = 0

eye_dis = []

i = 0

while cap.isOpened():
    
    i=i+1
    
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    rects = detector(gray, 1)
    
    
    if len(rects)==0:
        
        print("No People")
        cv2.putText(frame,'Warning // No People',(50,100),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        
    elif len(rects)==1:
        
        print("")
    
    elif len(rects)>1:
        
        print("Multiple People")
        cv2.putText(frame,'Warning // Multiple People',(50,100),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        
    ans = getImage(frame,class_threshold=0.4,label = labels)
    
    if ans == 'yes':
        cv2.putText(frame,'Warning // Cell Phone Detected',(50,100),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

   
    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = shape_to_np(shape)

        eye_dis.append(shape[43][0])
        

        if (len(eye_dis) > 6 ):

            x = abs(eye_dis[i]-eye_dis[i-5])

            print(x)

            if (x > 30) :

                cv2.putText(frame,'Warning // Eye Moving',(50,150),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,0),2)
                print('Eye')
                #print(x)

       
        if (shape[67][1]-shape[61][1]>=8):
            mou_frequency = mou_frequency + 1
            
        
        if(mou_frequency == 40):
            cv2.putText(frame,'Warning // Mouth opening',(50,200),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
            mou_frequency = 0
            print('Mouth')
        
        for (i,(x, y)) in  enumerate(shape):

            
            if (i==43 or i==44 or i==46 or i==47 or i==37 or i==38 or i==40 or i==41):
                cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)
                continue
                
            if (i==61 or i==62 or i==63 or i==67 or i==66 or i==65):
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
                continue
           
            
            cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)

    cv2.imshow('frame',frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# In[ ]:


while cap.isOpened():
    
    ret, frame = cap.read()
    ans = getImage(frame,class_threshold=0.4,label = labels)
    
    if ans == 'yes':
        cv2.putText(frame,'Warning // Cell Phone Detected',(50,100),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),3)
    
    cv2.imshow('frame',frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
    
    
    
    


# In[ ]:




