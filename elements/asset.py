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
    try:
        np.seterr(divide='ignore', invalid='ignore')
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

            test_img = np.zeros_like(mask)
            test_img[560:, 230:1100] = mask[560:, 230:1100]
            points,_ = cv2.findContours(test_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for point in points:
                if cv2.contourArea(point)>2500:
                    cv2.fillPoly(out_image, pts =[point], color=(0, 255,0))
            if out_points is not None:
                out_image=cv2.line(out_image, (out_points[0],out_points[1]), (out_points[2],out_points[3]), [0,0,255], 5)
    exceot:
        pass
    return out_image




#Cityscape information
def cityscape_xyz(disp, xl, yl, b = 0.22 ,f = 2262):
  z = (b*f)/(disp/1.6)
  x = (xl*z)/f
  y = (yl*z)/f

  dist = np.sqrt(x**2+y**2+z**2)
  return x, y, z, dist

#Kitti information
def kitti_xyz(disp, xl, yl, b = 0.5707 ,f = 645.24):
  z = (b*f)/(disp/1.6)
  x = (xl*z)/f
  y = (yl*z)/f

  dist = np.sqrt(x**2+y**2+z**2)
  return x, y, z, dist

def apply_mask1(image, seg_img, color = [244, 35, 232], alpha=0.5):
    img = image.copy()
    mask = (seg_img == np.array([244, 35, 232]))[...,1].astype('uint8')
    mask = (cv2.resize(mask, (img.shape[1], img.shape[0])))
             
    for c in range(3):
        img[:,:,c]= np.where(mask == 1, img[:,:,c]*(1 - alpha)+alpha*color[c],img[:,:,c])
    return img

def apply_mask(image, seg_img, color = [244, 35, 232], alpha=0.5):

    img = image.copy()
    # print(seg_img.shape)
    # print(seg_img.dtype)
    # np.save('img.npy', img)
    np.save('seg.npy',seg_img)

    # img = np.load('img.npy')
    seg_img = np.load('seg.npy')
    mask = (seg_img == np.array([244, 35, 232]))[...,1].astype('uint8')
    contours, _ = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    canvas = np.zeros_like(seg_img, np.uint8)

    cnts = sorted(contours, key = cv2.contourArea, reverse=True)
    n = len(cnts)

    if n < 5:
        for i in range(2):
            cv2.drawContours(canvas, cnts, i,  (0,255,0), -1, cv2.LINE_AA)
    else:
        for i in range(2):
            cv2.drawContours(canvas, cnts, i,  (0,255,0), -1, cv2.LINE_AA)
        
    mask_new = (canvas == np.array([0, 255, 0]))[...,1].astype('uint8')
    mask_new = (cv2.resize(mask_new, (img.shape[1], img.shape[0])))

    for c in range(3):
        img[:,:,c]= np.where(mask_new == 1, img[:,:,c]*(1 - alpha)+alpha*color[c],img[:,:,c])
    return img

