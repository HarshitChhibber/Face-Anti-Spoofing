from cvzone.FaceDetectionModule import FaceDetector
import cv2

def main():
    cap = cv2.VideoCapture(0)

    detector = FaceDetector(minDetectionCon=0.5, modelSelection=0)

    while True:
        success, img = cap.read()

        img, bboxs = detector.findFaces(img, draw=True)

        if bboxs:
            for bbox in bboxs:

                center = bbox["center"]
                x, y, w, h = bbox['bbox']
                score = int(bbox['score'][0] * 100)

        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()