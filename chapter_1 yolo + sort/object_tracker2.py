from ultralytics import YOLO
import cv2
import cvzone
import numpy as np
from sort import *
vid = cv2.VideoCapture("./4K Video of Highway Traffic!.mp4")
vid.set(3,1280)
vid.set(4,720)
model = YOLO("./yolov8n.pt")
classes_names=["person","bicycle","car","motorbike","aeroplane","bus","train","truck",
               "boat","traffic light","fire hydrant","stop sign","parking meter","bench","bird",
               "cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
               "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
               "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle",
               "wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange",
               "broccoli","carrot","hot dog","pizza","donut","cake","chair","sofa","pottedplant","bed",
               "diningtable","toilet","tvmonitor","laptop","mouse","remote","keyboard","cell phone","microwave",
               "oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier",
               "toothbrush"]
mask= cv2.imread("./main_loc_1.png")
tracker1 = Sort(max_age=20)
car_id_list1=[]
car_id_list2=[]
limits1=[50,420,620,450]
limits2=[700,410,1200,400]
while True:
    success, image = vid.read()
    reimage= cv2.bitwise_and(image,mask)
    #cv2.imshow("reimage",reimage)
    result = model(reimage)
    detection = np.empty((0,5))

    for i in result:
        print(result)
        for j in i.boxes:
            #adding the height and width
            x1,y1,x2,y2= j.xyxy[0]
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
            width,height = x2-x1,y2-y1
            #cvzone.cornerRect(image, (x1,y1,width,height),l=5)#,colorC=(220,100,21),colorR=(100,90,80)
            #adding the confidence
            confidence = round(float(j.conf[0]),2)
            #cvzone.putTextRect(image,f"{confidence}",(max(0,x1),max(35,y1)))
            #adding the class to the bounding dox
            class_id = j.cls[0]
            print(class_id,"\n",j.cls)

            if classes_names[int(class_id)]== "car" and confidence > 0.5:
                #scikit - image == 0.19.
                #cvzone.cornerRect(image, (x1, y1, width, height), l=5)
                #cvzone.putTextRect(image, f"{classes_names[int(class_id)]}:{confidence}",(max(0, x1), max(35, y1)),scale=1,thickness=1,offset=2)#,colorB=(90,90,12),thickness=3
                current_array = np.array((x1,y1,x2,y2,confidence))
                detection = np.vstack((detection,current_array))
    result=tracker1.update(detection)
    cv2.line(image, (limits1[0], limits1[1]), (limits1[2], limits1[3]),(255,0,255),5)
    cv2.line(image, (limits2[0], limits2[1]), (limits2[2], limits2[3]), (255, 0, 255), 5)
    for i in result:
        x1,y1,x2,y2,id=i
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print("in i ",x1,y1,x2,y2,id)
        width, height = x2 - x1, y2 - y1
        cvzone.cornerRect(image, (x1, y1, width, height), l=5)
        id = int(id)
        cvzone.putTextRect(image, f"{id}", (max(0, x1), max(35, y1)), scale=1,thickness=1, offset=2)
        #calculating the center
        cx,cy= x1+int((width/2)),y1+int((height/2))
        cv2.circle(image,(cx,cy),5,(255,0,0),cv2.FILLED)
        if (cx > limits1[0] -10and cx < limits1[2]+10)  and (cy > limits1[1]-5 and cy < limits1[3]+5):
            if id not in car_id_list1:
                car_id_list1.append(id)
        cvzone.putTextRect(image,f"{len(car_id_list1)}",(15,40))
        if (cx > limits2[0]-10 and cx < limits2[2]+10)  and (cy < limits2[1]+5 and cy > limits2[3]-5):
            if id not in car_id_list2:
                car_id_list2.append(id)
        cvzone.putTextRect(image,f"{len(car_id_list2)}",(1210,40))
        cvzone.putTextRect(image, f"{len(car_id_list2)+len(car_id_list1)}", (650, 40))




    cv2.imshow("this is the image",image)
    #cv2.waitKey(1)
    if cv2.waitKey(1)  & 0xFF == ord("q"):
        break
