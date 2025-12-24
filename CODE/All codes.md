###### **1. Environment Setup**

###### **1.1 Install Required Libraries**





!pip install -U ultralytics





###### 1.2 Verify GPU Availability



!nvidia-smi



###### ---------------------------------------------------------------------------------------------

###### 2\. Dataset Preparation

###### 2.1 Dataset Directory Structure



yolo\_data/

├── images/

│   ├── train/

│   └── val/

├── labels/

│   ├── train/

│   └── val/



###### 2.2 Create Dataset Folders





import os



base\_path = "/kaggle/working/yolo\_data"



folders = \[

&nbsp;   "images/train",

&nbsp;   "images/val",

&nbsp;   "labels/train",

&nbsp;   "labels/val"

]



for folder in folders:

&nbsp;   os.makedirs(os.path.join(base\_path, folder), exist\_ok=True)



-------------------------------------------------------------------------------------------------------------------------

###### 2.3 Split Dataset (Train / Validation)



import shutil

import random



data\_path = "/kaggle/input/crop-weed-data"

images = \[f for f in os.listdir(data\_path) if f.endswith(".jpeg")]



random.shuffle(images)



train\_split = int(0.8 \* len(images))

train\_imgs = images\[:train\_split]

val\_imgs = images\[train\_split:]



def move\_files(file\_list, split):

&nbsp;   for img in file\_list:

&nbsp;       label = img.replace(".jpeg", ".txt")

&nbsp;       shutil.copy(f"{data\_path}/{img}", f"{base\_path}/images/{split}/{img}")

&nbsp;       shutil.copy(f"{data\_path}/{label}", f"{base\_path}/labels/{split}/{label}")



move\_files(train\_imgs, "train")

move\_files(val\_imgs, "val")



----------------------------------------------------------------------------------------------------------------------------------------

###### 3\. Dataset Configuration File

###### 3.1 Create data.yaml





data\_yaml = """

path: /kaggle/working/yolo\_data

train: images/train

val: images/val



nc: 2

names: \['crop', 'weed']

"""



with open("/kaggle/working/data.yaml", "w") as f:

&nbsp;   f.write(data\_yaml)



------------------------------------------------------------------------------------------------------------------------------

###### 4\. Model Training (Initial Training)

###### 4.1 Import YOLO





from ultralytics import YOLO



4.2 Train YOLOv8 (10 Epochs – Baseline)

model = YOLO("yolov8n.pt")



model.train(

&nbsp;   data="/kaggle/working/data.yaml",

&nbsp;   epochs=10,

&nbsp;   imgsz=416,

&nbsp;   batch=16,

&nbsp;   name="baseline\_crop\_weed"

)



---------------------------------------------------------------------------------------------------------------------------

###### 5\. Improved Training (Higher Resolution)

###### 5.1 Train with 512×512 Resolution (20 Epochs)



model = YOLO("yolov8n.pt")



model.train(

&nbsp;   data="/kaggle/working/data.yaml",

&nbsp;   epochs=20,

&nbsp;   imgsz=512,

&nbsp;   batch=16,

&nbsp;   name="crop\_weed\_512"

)



----------------------------------------------------------------------------------------------------------------------------------

###### 6\. Fine-Tuning from Best Weights

###### 6.1 Continue Training from best.pt



model = YOLO("/kaggle/working/runs/detect/crop\_weed\_512/weights/best.pt")



model.train(

&nbsp;   data="/kaggle/working/data.yaml",

&nbsp;   epochs=50,

&nbsp;   imgsz=512,

&nbsp;   batch=16,

&nbsp;   name="crop\_weed\_512\_finetune"

)



------------------------------------------------------------------------------------------------------------------------------------

###### 7\. Model Evaluation (Performance Testing)

###### 7.1 Validate Model



model = YOLO("/kaggle/working/runs/detect/crop\_weed\_512\_finetune/weights/best.pt")



metrics = model.val(

&nbsp;   data="/kaggle/working/data.yaml",

&nbsp;   imgsz=512

)



print(metrics)



---------------------------------------------------------------------------------------------------------------------------------------

###### **8. Performance Metrics (Manual Extraction)**

###### **8.1 Display Key Metrics**



print("Precision:", metrics.box.precision)

print("Recall:", metrics.box.recall)

print("mAP50:", metrics.box.map50)

print("mAP50-95:", metrics.box.map)



----------------------------------------------------------------------------------------------------------------

###### **9. Inference on Sample Images**

###### **9.1 Run Prediction**



model.predict(

&nbsp;   source="/kaggle/working/yolo\_data/images/val",

&nbsp;   imgsz=512,

&nbsp;   conf=0.25,

&nbsp;   save=True

)

