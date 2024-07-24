import cv2
import numpy as np
import urllib.request
# cv2.startWindowThread()

url = 'http://129.161.208.44/stream'
print("hello world")

cap = cv2.VideoCapture(url)

classesfile = 'coco.names'
classNames = []
with open(classesfile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

# get a different color array for each of the classes
COLORS = np.random.uniform(0, 255, size=(len(classNames), 3))

modelConfig = 'yolov3.cfg'
modelWeights = 'yolov3.weights'
net = cv2.dnn.readNetFromDarknet(modelConfig, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

confThreshold = 0.5
nmsThreshold = 0.3

def findObject(outputs, im):
    hT, wT, cT = im.shape
    bbox = []
    classIds = []
    confs = []
    found_cat = False
    found_bird = False

    # for output in outputs:
    #     for det in output:
    #         scores = det[5:]
    #         classId = np.argmax(scores)
    #         confidence = scores[classId]
    #         if confidence > confThreshold:
    #             w,h = int(det[2]*wT), int(det[3]*hT)
    #             x,y = int((det[0]*wT)-w/2), int((det[1]*hT)-h/2)
    #             bbox.append([x,y,w,h])
    #             classIds.append(classId)
    #             confs.append(float(confidence))

    # loop over each of the detection
    for detection in outputs:
    # extract the confidence of the detection
        confidence = detection[2]
        if confidence > .4:
            # get the class id
            class_id = detection[1]
            # map the class id to the class
            className = classNames[int(class_id)-1]
            print(className)
            color = COLORS[int(class_id)]
            # get the bounding box coordinates
            box_x = detection[3] * wT
            box_y = detection[4] * hT
            # get the bounding box width and height
            box_width = detection[5] * wT
            box_height = detection[6] * hT
            # draw a rectangle around each detected object
            cv2.rectangle(im, (int(box_x), int(box_y)), (int(box_width), int(box_height)), color, thickness=2)
            # put the FPS text on top of the frame
            cv2.putText(im, className, (int(box_x), int(box_y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # #print(len(bbox))
    # indices = cv2.dnn.NMSBoxes(bbox,confs,confThreshold,nmsThreshold)
    # print(indices)
   
    # for i in indices:
    #     i = i[0]
    #     box = bbox[i]
    #     x,y,w,h = box[0],box[1],box[2],box[3]
    #     if classNames[classIds[i]] == 'bird':
    #         found_bird = True
    #     elif classNames[classIds[i]] == 'cat':
    #         found_cat = True
            
    #     if classNames[classIds[i]]=='bird':
            
    #         cv2.rectangle(im,(x,y),(x+w,y+h),(255,0,255),2)
    #         cv2.putText(im, f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%', (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,255), 2)
    #         print('bird')
    #         print(found_bird)
            
    #     if classNames[classIds[i]]=='cat':
             
    #         cv2.rectangle(im,(x,y),(x+w,y+h),(255,0,255),2)
    #         cv2.putText(im, f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%', (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,255), 2)
    #         print('cat')
    #         print(found_cat)
            
            
    #     if found_cat and found_bird:
    #         print('alert')


while (True):

    # Capture the video frame 
    # by frame q
    ret, im = cap.read()

    # create blob from image
    # blob = cv2.dnn.blobFromImage(image=im, size=(320, 320), mean=(104, 117, 123), swapRB=True, crop=False)
    # blob = cv2.dnn.blobFromImage(image=im, scalefactor=0.01, size=(224, 224), mean=(104, 117, 123))
    blob = cv2.dnn.blobFromImage(im,1/255,(320,320),[0,0,0],1,crop=False)

    # set the input blob for the neural network
    net.setInput(blob)

    # Get the name of all layers of the network
    layernames = net.getLayerNames()
    # print(layernames) #***********************

    # getUnconnectedOutLayers() => Returns names of layers with unconnected outputs.
    # print(net.getUnconnectedOutLayers())
    outputNames = [layernames[i-1] for i in net.getUnconnectedOutLayers()]

    # forward pass image blog through the model
    # outputs = net.forward()
    outputs = net.forward(outputNames)
    # print(outputs.shape)
    print(outputs[1].shape)
    print(outputs[0].shape)
    print(outputs[2].shape)
    print(outputs[0][0])

    # findObject(outputs, im)

    # Display the resulting frame 
    cv2.imshow('frame', im) 
      
    # the 'q' button is set as the 
    # quitting button you may use any 
    # desired button of your choice 
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
  
# After the loop release the cap object 
cap.release() 
# Destroy all the windows 
cv2.destroyAllWindows() 
