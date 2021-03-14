import numpy as np
import cv2

def classic_distance(image, pts):
	pts = pts.astype(dtype = "float32")
	(tl, tr, br, bl) = pts
	widthA = np.sqrt(((br[0] - bl[0]) * 2) + ((br[1] - bl[1]) * 2))
	widthB = np.sqrt(((tr[0] - tl[0]) * 2) + ((tr[1] - tl[1]) * 2))
	maxWidth = max(int(widthA), int(widthB))
	heightA = np.sqrt(((tr[0] - br[0]) * 2) + ((tr[1] - br[1]) * 2))
	heightB = np.sqrt(((tl[0] - bl[0]) * 2) + ((tl[1] - bl[1]) * 2))
	maxHeight = max(int(heightA), int(heightB))
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
	M = cv2.getPerspectiveTransform(pts, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
	ppm=100
	distance = (warped.shape[0]/ppm)
	return distance

    
def detect_lines(image):
    rho = 1  # precision in pixel, i.e. 1 pixel
    angle = np.pi / 180  # degree in radian, i.e. 1 degree
    min_threshold = 10  # minimal of votes
    lines = cv2.HoughLinesP(image, rho, angle, min_threshold, np.array([]), minLineLength=8,
                                    maxLineGap=4)
            
    return lines


def horiz_lines(input_frame, out_image, mode = 1):
    try:
        np.seterr(divide='ignore', invalid='ignore')
        frame = input_frame.copy()
        #mode1
        if mode == 1:
            mask = cv2.inRange(frame, np.array([70,70,90]), np.array([165,140,135]))
        #mode2
        if mode == 2:
            mask = cv2.inRange(frame, np.array([50,65,90]), np.array([135,205,170]))
        #mode3
        if mode == 3:
            mask = cv2.inRange(frame, np.array([180,180,180]), np.array([255,255,255]))
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
    except:
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


def apply_all_mask(image, seg_img, color = [244, 35, 232], alpha=0.5):
    img = image.copy()
    mask = (seg_img == np.array([244, 35, 232]))[...,1].astype('uint8')
    mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
             
    for c in range(3):
        img[:,:,c]= np.where(mask == 1, img[:,:,c]*(1 - alpha)+alpha*color[c],img[:,:,c])
    return img


def apply_mask(image, seg_img, masked_image, color = [244, 35, 232], alpha=0.5):
    img = image.copy()
    mask = (seg_img == np.array([244, 35, 232]))[...,1].astype('uint8')
    contours, _ = cv2.findContours(mask.copy() ,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    canvas = np.zeros((seg_img.shape[0],seg_img.shape[1],seg_img.shape[2]), np.uint8)

    cnts = sorted(contours, key = cv2.contourArea, reverse=True)

    if len(cnts) > 2:
        for i in range(2):
            cv2.drawContours(canvas, cnts, i, (0,255,0), -1, cv2.LINE_AA)
    else:
        cv2.drawContours(canvas, cnts, -1, (0,255,0), -1, cv2.LINE_AA)

    mask_new = (canvas == np.array([0, 255, 0]))[...,1].astype('uint8')
    mask_new = cv2.resize(mask_new, (img.shape[1], img.shape[0]))
    masked_image = cv2.bitwise_and(masked_image, masked_image, mask =mask_new)
    mask = (masked_image > 0)[...,1].astype('uint8')

    for c in range(3):
        img[:,:,c]= np.where(mask == 1, img[:,:,c]*(1 - alpha)+alpha*color[c],img[:,:,c])

    return img


def ROI(frame):
    mask = np.zeros(frame.shape, dtype=np.uint8)
    roi_corners = np.array([[(180,720), (432,300), (1000,300), (1200,720)]], dtype=np.int32)
    channel_count = frame.shape[2]
    ignore_mask_color = (255,)*channel_count
    cv2.fillPoly(mask, roi_corners, ignore_mask_color)
    masked_image = cv2.bitwise_and(frame, mask)

    return masked_image
  

def kitti_xyz_dist(disp, b = 0.5707 ,f = 645.24):
    dist = []
    for data in disp:
        
        z = (b*f)/(data[2]/1.6)
        x = (data[0]*z)/f
        y = (data[1]*z)/f

        dist.append(np.sqrt(x**2+y**2+z**2))
    
    return dist


def cityscape_xyz_dist(disp, b = 0.22 ,f = 2262):
    dist = []
    for data in disp:
        
        z = (b*f)/(data[2]/1.6)
        x = (data[0]*z)/f
        y = (data[1]*z)/f

        dist.append(np.sqrt(x**2+y**2+z**2))
    
    return dist


#From Yolov5 Github
def plot_one_box(x, img, distance=None, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled

        if distance:
            cv2.putText(img, f'{label}, {np.round(distance, 2)}m', (c1[0], c1[1] - 2), 0, tl/3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
        else:
            cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl/3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
