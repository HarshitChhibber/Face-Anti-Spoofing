import cvzone
from cvzone.FaceDetectionModule import FaceDetector
import cv2
from time import time

##############################

classID = 1 # 0 = Fake, 1 = Real
outputFolderPath = 'DataSet/Data Collection'
offsetPercentageW = 10
offsetPercentageH = 20
confidence = 0.8
camwidth, camheight = 640, 480
floatingpoints = 6
save = True
blurThreashold = 35
debug = True

##############################

cap = cv2.VideoCapture(0)
cap.set(3, camwidth)
cap.set(4, camheight)
detector = FaceDetector(minDetectionCon=0.5, modelSelection=0)

while True:
    listBlur = [] # True / False Values if faces are blur or not
    listInfo = [] # Normalized Values
    success, img = cap.read()
    imgOut = img.copy()

    img, bboxs = detector.findFaces(img, draw=False)

    if bboxs:
        for bbox in bboxs:
            x, y, w, h = bbox["bbox"]
            score = bbox["score"][0]
            print(x,y,w,h)

            # ------------------- Score ----------------------

            if score > confidence:

                # ----------------- Offset for Faces --------------------
                offsetW = (offsetPercentageW/100) * w
                x = int(x - offsetW)
                w = int(w + offsetW * 2)

                offsetH = (offsetPercentageH / 100) * h
                y = int(y - offsetH * 3)
                h = int(h + offsetH * 3)


                # ------------------- Value Below 0 ---------------------
                if x < 0: x = 0
                if y < 0: y = 0
                if w < 0: w = 0
                if h < 0: h = 0


                # -------------------- Blurriness ---------------------------
                imgFace = img[y:y + h, x:x + w]
                cv2.imshow("Face", imgFace)
                blurValue = int(cv2.Laplacian(imgFace, cv2.CV_64F).var())
                if blurValue > blurThreashold:
                    listBlur.append (True)
                else:
                    listBlur.append (False)

                # ------------------------ Normalize ----------------------

                ih, iw, _ = img.shape
                xc, yc = x + w/2, y +h/2
                xcn, ycn = round(xc/iw, floatingpoints), round(yc/ih, floatingpoints)
                wn, hn = round(w/iw, floatingpoints), round(h/ih, floatingpoints)


                # ------------------- Value Below 0 ---------------------
                if xcn > 1: xcn = 1
                if ycn > 1: ycn = 1
                if wn > 1: wn = 1
                if hn > 1: hn = 1

                listInfo.append(f'{classID} {xcn} {ycn} {wn} {hn}\n')

                # ------------------------- Drawing ---------------------------
                cv2.rectangle(imgOut, (x,y,w,h),(255,0,0),3)
                cvzone.putTextRect(imgOut, f'Score {int(score * 100)}% Blur: {blurValue}', (x, y-20),
                                   scale=1, thickness=2)

                if debug:
                    cv2.rectangle(imgOut, (x, y, w, h), (255, 0, 0), 3)
                    cvzone.putTextRect(imgOut, f'Score {int(score * 100)}% Blur: {blurValue}', (x, y - 20),
                                       scale=1, thickness=2)
    # -------------------- To Save --------------------

    if save:
        if all(listBlur) and listBlur != []:
            # ------------------------ Save Image ---------------------
            timeNow = time()
            timeNow = str(timeNow).split('.')
            timeNow = timeNow[0] + timeNow[1]
            print(time())
            cv2.imwrite(f'{outputFolderPath}/{timeNow}.jpg', img)

            # --------------------------- Save Label Text File --------------------------
            for info in listInfo:
                f = open(f'{outputFolderPath}/{timeNow}.txt', "a")
                f.write(info)
                f.close()


    cv2.imshow("Image", imgOut)
    cv2.waitKey(1)