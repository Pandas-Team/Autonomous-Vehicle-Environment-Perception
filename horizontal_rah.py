
import numpy as np
import cv2

def detect_lines(image):
    rho = 1  # precision in pixel, i.e. 1 pixel
    angle = np.pi / 180  # degree in radian, i.e. 1 degree
    min_threshold = 10  # minimal of votes
    lines = cv2.HoughLinesP(image, rho, angle, min_threshold, np.array([]), minLineLength=8,
                                    maxLineGap=4)
            
    return lines

def horiz_lines(input_frame, out_image):
    frame = input_frame.copy()
    mask = cv2.inRange(frame, np.array([70,70,90]), np.array([165,140,135]))
    roi = mask[560:, 230:1100]
    lines = detect_lines(roi)
    lines = lines.reshape(-1,2,2)
    slope = (lines[:,1,1]-lines[:,0,1]) / (lines[:,1,0]-lines[:,0,0])
    if (lines[np.where(abs(slope)<0.2)]).shape[0] > 60:
        xmin = lines[np.where(abs(slope)<0.2)].reshape(-1,4)[:,0].min(0)
        ymin = lines[np.where(abs(slope)<0.2)].reshape(-1,4)[:,1].min(0)
        xmax = lines[np.where(abs(slope)<0.2)].reshape(-1,4)[:,2].max(0)
        ymax = lines[np.where(abs(slope)<0.2)].reshape(-1,4)[:,3].max(0)
        
        mean_slope = np.mean(slope[np.where(abs(slope)<0.2)])
        xy = np.array([xmin,ymin,xmax,ymax]) + [230,560,230,560]
        y0,x0 = np.mean(np.where(roi>0), axis=1) + [560,230]
        x = np.array([xy[0],xy[2]])
        y = mean_slope*(x - x0) + y0

        xmin,xmax = x
        ymin,ymax = y.astype(int)
        out_points = np.array([xmin,ymin,xmax,ymax])
        if out_points is not None:
            out_image=cv2.line(out_image, (out_points[0],out_points[1]), (out_points[2],out_points[3]), [0,0,255], 5)
        test_img = np.zeros_like(mask)
        test_img[560:, 230:1100] = mask[560:, 230:1100]
        points,_ = cv2.findContours(test_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for point in points:
            if cv2.contourArea(point)>3700:
                cv2.fillPoly(out_image, pts =[point], color=(0,0,255))

    return out_image


cap = cv2.VideoCapture("test.mov")
counter = 0
while(cap.isOpened()):
    ret , frame = cap.read()
    frame = cv2.rotate(frame,cv2.ROTATE_90_COUNTERCLOCKWISE)

    frame = horiz_lines(frame)
    cv2.imshow('output', frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

