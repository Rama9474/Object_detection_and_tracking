# %%
import cv2


# %%
import matplotlib.pyplot as plt

# %%
config_file= 'C:/Users/swarn/OneDrive/Desktop/object detection/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model= 'c:/Users/swarn/OneDrive/Desktop/object detection/frozen_inference_graph.pb'

# %%
model = cv2.dnn_DetectionModel(frozen_model,config_file)

# %%
classLabels = []
file_name = 'C:/Users/swarn/OneDrive/Desktop/object detection/Labels.txt'
with open(file_name,'rt') as fpt:
    classLabels = fpt.read().rstrip('\n').split('\n')


# %%
print(classLabels)

# %%
print(len(classLabels))

# %%
model.setInputSize(320,320)
model.setInputScale(1.0/127.5)
model.setInputMean((127.5,127.5,127.5))
model.setInputSwapRB(True)

# %%
img=cv2.imread('C:/Users/swarn/OneDrive/Desktop/object detection/traffic.jpg')

# %%
plt.imshow(img)

# %%
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

# %%

# Example: Assuming expected_size is the size expected by the model
#expected_size = (10, 10)
#img_resized = cv2.resize(img, expected_size)
ClassIndex, confidence, bbox = model.detect(img,confThreshold=0.5)



# %%
print(ClassIndex)

# %%
'''font_scale = 3
font =cv2.FONT_HERSHEY_PLAIN
for classInd, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
    cv2.rectangle(img,boxes,(255,0,0),2)
    cv2.putText(img,classLabels[ClassIndex-1],(boxes[0]+10,boxes[1]+40),font,fontScale=font_scale,color=(0,255,0),thickness=3)
    '''
font_scale = 3
font = cv2.FONT_HERSHEY_PLAIN

for classInd, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
    cv2.rectangle(img, boxes, (255, 0, 0), 2)
    cv2.putText(img, classLabels[int(classInd) - 1], (boxes[0] + 10, boxes[1] + 40), font, fontScale=font_scale, color=(0, 255, 0), thickness=3)


# %%
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

# %%
#cap = cv2.VideoCapture('C:/Users/swarn/OneDrive/Desktop/object detection/videotr.mp4')#to detect video feed
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open camera")#you can use "can't open video" for video feed

font_scale = 3
font = cv2.FONT_HERSHEY_PLAIN

while True:
    ret, frame = cap.read()

    #Check if the frame is successfully read
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    ClassIndex, confidence, bbox = model.detect(frame, confThreshold=0.5)

    print(ClassIndex)
    if len(ClassIndex) != 0:
        for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
            if ClassInd <= 80:
                cv2.rectangle(frame, boxes, (255, 0, 0), 2)
                cv2.putText(frame, classLabels[int(ClassInd) - 1], (boxes[0] + 10, boxes[1] + 40), font, fontScale=font_scale, color=(0, 255, 0), thickness=3)

    cv2.imshow('Object Detection', frame)

    if cv2.waitKey(2) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



