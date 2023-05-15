from ultralytics import YOLO
import cv2
import cvzone
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
mask= cv2.imread("./mask.png")
while True:
    success, image = vid.read()
    reimage= cv2.bitwise_and(image,mask)
    cv2.imshow("reimage",reimage)
    result = model(reimage)

    for i in result:
        print(result)
        for j in i.boxes:
            #adding the height and width
            x1,y1,x2,y2= j.xyxy[0]
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
            width,height = x2-x1,y2-y1
            cvzone.cornerRect(image, (x1,y1,width,height),l=5)#,colorC=(220,100,21),colorR=(100,90,80)
            #adding the confidence
            confidence = round(float(j.conf[0]),2)
            #cvzone.putTextRect(image,f"{confidence}",(max(0,x1),max(35,y1)))
            #adding the class to the bounding dox
            class_id = j.cls[0]
            print(class_id,"\n",j.cls)
            cvzone.putTextRect(image, f"{classes_names[int(class_id)]}:{confidence}",(max(0, x1), max(35, y1)),scale=1,thickness=1,offset=2)#,colorB=(90,90,12),thickness=3

    cv2.imshow("this is the image",image)
    #cv2.waitKey(1)
    if cv2.waitKey(1)  & 0xFF == ord("q"):
        break
