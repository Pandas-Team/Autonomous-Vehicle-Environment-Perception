import numpy as np
import cv2
import random 
from skimage.measure import block_reduce
from PIL import ImageFont, ImageDraw, Image


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


def horiz_lines(input_frame, out_image, mode = 'day'):
    try:
        np.seterr(divide='ignore', invalid='ignore')
        frame = input_frame.copy()
        
        #mode1
        if mode == 'day':
            mask = cv2.inRange(frame, np.array([70,70,90]), np.array([165,140,135]))
        #mode2
        if mode == 'night':
            mask = cv2.inRange(frame, np.array([50,65,90]), np.array([135,205,170]))

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


def apply_all_mask(image, seg_img, color = [244, 35, 232], alpha=0.5):
    img = image.copy()
    mask = np.where(seg_img==1, seg_img, 0).astype('uint8') #sidewalk
    mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
             
    for c in range(3):
        img[:,:,c]= np.where(mask == 1, img[:,:,c]*(1 - alpha)+alpha*color[c],img[:,:,c])
    return img


def apply_mask(image, seg_img, mode = 'day', color = [244, 35, 232], alpha=0.5):
    img = image.copy()
    
    mask = np.where(seg_img==1, seg_img, 0).astype('uint8') #sidewalk
    canvas = np.zeros((mask.shape[0], mask.shape[1], 3), np.uint8)

    # preprocessing
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.erode(mask, kernel, iterations=3)

    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(contours, key = cv2.contourArea, reverse=True)

    if len(cnts) > 2:
        if mode == 'day':
            for i in range(2):
                cv2.drawContours(canvas, cnts, i, (0, 255, 0), -1, cv2.LINE_AA)
        else:
            cv2.drawContours(canvas, cnts, 0, (0,255,0), -1, cv2.LINE_AA)
    else:
        cv2.drawContours(canvas, cnts, -1, (0,255,0), -1, cv2.LINE_AA)

    new_mask = (canvas == np.array([0, 255, 0]))[...,1].astype('uint8')
    new_mask = cv2.resize(new_mask, (img.shape[1], img.shape[0]))
    mask = (new_mask > 0).astype('uint8') 

    masked_img = np.where(mask[..., None], color, img).astype(np.uint8)
    # masked_img = cv2.addWeighted(img, 0.5, masked_img, 0.5, 0)

    return masked_img


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
            cv2.putText(img, f'{label}, {distance:.2f}m', (c1[0], c1[1] - 2), 0, tl/4, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
        else:
            cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl/4, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def plot_object_colors():
    obj_names = ['pedestrian', 'vehicle', 'traffic light', 'stop sign']
    obj_colors = {item: [random.randint(0, 255) for _ in range(3)] for item in obj_names}

    signs = ['Give Way', 'No Left', 'No Right', 'SL30', 'No Stopping',
            'No Entry', 'Ahead Only', 'SL40', 'SL50', 'SL60', 'SL70', 'SL80', 'SL100', 'No U-Turn']

    sign_colors = {item: [random.randint(0, 255) for _ in range(3)] for item in signs}

    return obj_colors, sign_colors


###### Depth Estimation ######
labels = {
               'person': 11,
               'car': 13,
               'truck': 14,
               'bus': 15,
              }
def depth_estimator(xyxy, depth, seg, obj_name, mask_state=True):
    Ry = 192/720
    Rx = 640/1280

    obj_mask = np.where(seg==labels[obj_name], seg, 0).astype('uint8')//labels[obj_name]
    obj_mask = obj_mask[int(xyxy[1]*Ry):int(xyxy[3]*Ry), int(xyxy[0]*Rx):int(xyxy[2]*Rx)]

    obj_depth = depth[int(xyxy[1]*Ry):int(xyxy[3]*Ry), int(xyxy[0]*Rx):int(xyxy[2]*Rx)]
    out_obj = obj_depth * obj_mask if mask_state else obj_depth

    # Zero-2-NaN 
    out_obj[out_obj==0]=np.nan 
    
    # Min pooling with the kernel size of 3x3
    min_pool = block_reduce(out_obj, block_size=(3,3), func=np.min)
    
    # Gaussian Noise Removal
    std  =  np.std(min_pool[~np.isnan(min_pool)])
    mean = np.mean(min_pool[~np.isnan(min_pool)])
    obj_GNM = np.where((mean - 2*std < min_pool) & (min_pool < mean + 2*std), min_pool, np.nan)
    
    # Group Avg pooling, 2x2, 3x3, and 5x5
    mean_pool_2x2 = block_reduce(obj_GNM, block_size=(2,2), func=np.mean)
    mean_pool_3x3 = block_reduce(obj_GNM, block_size=(3,3), func=np.mean)
    mean_pool_5x5 = block_reduce(obj_GNM, block_size=(5,5), func=np.mean)

    # Flattening and Concatenating
    if mean_pool_5x5 is None:
        obj_avg = np.concatenate([mean_pool_2x2.flatten(), mean_pool_3x3.flatten()], axis=0)
    else:
        obj_avg = np.concatenate([mean_pool_2x2.flatten(), mean_pool_3x3.flatten(), mean_pool_5x5.flatten()], axis=0)
    
    # NaN Removal and Global Avg
    g_avg = obj_avg[~np.isnan(obj_avg)].mean()

    if g_avg is None:
        g_avg = mean
    return g_avg
    

###### ROI Generation ######
def static_ROI(frame):
    mask = np.zeros(frame.shape, dtype=np.uint8)
    roi_corners = np.array([[(180,720), (432,300), (1000,300), (1200,720)]], dtype=np.int32)
    channel_count = frame.shape[2]
    ignore_mask_color = (255,)*channel_count
    cv2.fillPoly(mask, roi_corners, ignore_mask_color)
    
    return mask


def dynamic_ROI(frame, road_mask):
    # Finding Contours
    contours, _ = cv2.findContours(road_mask, 1, 2)
    contours = sorted(contours, key=cv2.contourArea)

    # Convex Hull
    hull = [cv2.convexHull(contours[i], False) for i in range(len(contours))]

    # Fillpoly
    drawing = np.zeros((road_mask.shape[0], road_mask.shape[1], 3), np.uint8)
    cv2.fillPoly(drawing, pts =hull, color=(255,255,255))
    cv2.fillPoly(drawing, pts =contours, color=(255,255,255))

    # Preprocessing: Erosion, Dialation, and Resizing 
    img_dilation = cv2.dilate(drawing, np.ones((5,5), np.uint8), iterations=1)
    img_erosion = cv2.erode(img_dilation, np.ones((5,5), np.uint8), iterations=1)
    resized_mask = cv2.resize(img_erosion, (frame.shape[1], frame.shape[0]))

    return resized_mask
    
    
def ROI(frame, seg):
    # Define Road Mask
    road_mask = 1 - np.where(seg==0, seg, 1).astype('uint8')
    
    try:
        # Dynamic ROI
        ROI_output = dynamic_ROI(frame, road_mask)
        return ROI_output * frame
    except:
        # Static ROI
        ROI_output = static_ROI(frame)
        return ROI_output * frame


###### UI ######
# Set font
font = ImageFont.truetype("./elements/Roboto-Medium.ttf", 25)
sign_font = ImageFont.truetype("./elements/Roboto-Medium.ttf", 20)
light_msg = {'green': 'GO', 'red': 'STOP', 'yellow': 'WARNING', 'off': 'OFF'}


def obj_display(obj, type):
    if type == 'light':
        light_img = cv2.imread(f'./imgs/{obj}.jpg')
        light_img = cv2.resize(light_img, (130, 280))
        return light_img

    else:
        sign_img = cv2.imread(f'./imgs/{obj}.png')
        sign_img = cv2.resize(sign_img, (150, 150))
        return sign_img


def text_display(obj, draw, type):
    if type == 'light':
        draw.text((10, 10), f"Traffic light: {light_msg[obj]}!", (0,0,0) , font=font)
    if type == 'sign':
        draw.text((10, 380), f"Detected Sign: {obj}!", (0,0,0) , font=sign_font)


import warnings
warnings.filterwarnings('ignore')
def ui(frame, yoloOutput, light_detector, signOutput, depth_values):

    background = 255*np.ones((720, 280, 3)).astype('uint8')
    bg_rgb = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
    pil_im = Image.fromarray(bg_rgb)
    draw = ImageDraw.Draw(pil_im)
    
    # Traffic Light
    lights = yoloOutput.loc[yoloOutput['class'] == 9]
    df= lights.copy()
    
    if len(lights) >= 1:
        df['area'] = (lights["ymax"] - lights["ymin"]) * (lights["xmax"] - lights["xmin"])
        lights_1 = lights.loc[df['area'] == df['area'].max()]

        if len(df) > 1:
            lights_2 = lights.loc[df['area'] == df['area'].nlargest(2).values[1]] 
            light_color_1 = light_detector.classify(frame[int(lights_1["ymin"]): int(lights_1["ymax"]), int(lights_1["xmin"]): int(lights_1["xmax"])])
            light_color_2 = light_detector.classify(frame[int(lights_2["ymin"]): int(lights_2["ymax"]), int(lights_2["xmin"]): int(lights_2["xmax"])])
            light_color_3 = None

            if len(df) > 2:
                lights_3 = lights.loc[df['area'] == df['area'].nlargest(3).values[2]] 
                light_color_3 = light_detector.classify(frame[int(lights_3["ymin"]): int(lights_3["ymax"]), int(lights_3["xmin"]): int(lights_3["xmax"])])


            if (light_color_1 == 'off') and (light_color_3 == 'None'):
                light_color = light_color_2
            elif (light_color_1 == 'off') and (light_color_2 == 'off') and (light_color_3 != None):
                light_color = light_color_3
            else:
                light_color = light_color_1
        else:  
            light_color = light_detector.classify(frame[int(lights_1["ymin"]): int(lights_1["ymax"]), int(lights_1["xmin"]): int(lights_1["xmax"])])

        light_img = obj_display(light_color, type='light')
        background[60:340, 75:205] = light_img
        text_display(light_color, draw, type='light')
        
    else:
        off_img = obj_display('off', type='light')
        background[60:340, 75:205] = off_img
        draw.text((10, 10), f"Not Detected!", (0,0,0) , font=font)
    

    # Sign
    stop_sign = yoloOutput.loc[yoloOutput['class'] == 11]
    
    if len(stop_sign) >= 1:
        stop_sign['area'] = (stop_sign["ymax"] - stop_sign["ymin"]) * (stop_sign["xmax"] - stop_sign["xmin"])
        stop_sign = stop_sign.loc[stop_sign['area'] == stop_sign['area'].max()]

    if len(signOutput) >= 1:
        signOutput['area'] = (signOutput["ymax"] - signOutput["ymin"]) * (signOutput["xmax"] - signOutput["xmin"])
        signOutput = signOutput.loc[signOutput['area'] == signOutput['area'].max()]

    if len(stop_sign) >= 1  and len(signOutput) >= 1:
        if (stop_sign['area'].values[0] > signOutput['area'].values[0]) and (stop_sign['confidence'].values[0] > signOutput['confidence'].values[0]) : 
            sign_img = obj_display('stop sign', type='sign')
            background[430:580, 65:215] = sign_img
            text_display('stop sign', draw, type='sign')
        else: 
            sign_img = obj_display(signOutput['name'].values[0], type='sign')
            background[430:580, 65:215] = sign_img
            text_display(signOutput['name'].values[0], draw, type='sign')

    elif len(stop_sign) >= 1 and len(signOutput) < 1:
            sign_img = obj_display('stop sign', type='sign')
            background[430:580, 65:215] = sign_img
            text_display('stop sign', draw, type='sign')

    elif len(signOutput) >= 1 and len(stop_sign) < 1:
            sign_img = obj_display(signOutput['name'].values[0], type='sign')
            background[430:580, 65:215] = sign_img
            text_display(signOutput['name'].values[0], draw, type='sign')

    else:
        draw.text((10, 380), f"No Sign Detected!", (0,0,0) , font=font)


    # Distance Measurement Display
    safe_zone_thresh = 5 # meter
    draw.text((10, 620), "Nearest Distance:", (0,0,0) , font=font)
    if depth_values:
        if min(depth_values) <= safe_zone_thresh: # set safe zone threshold
            draw.text((100, 660), f"{min(depth_values):.2f} m", (0,0,0) , font=font)
        else:
            draw.text((80, 660), "Safe Zone :)", (0,0,0) , font=font)
    else: 
        draw.text((80, 660), "Safe Zone :)", (0,0,0) , font=font)

    # Boundary between light and sign   
    draw.line([(25, 360), (255, 360)], fill =(0,0,0), width = 2)

    # Boundary between sign and distance  
    draw.line([(25, 600), (255, 600)], fill =(0,0,0), width = 2)

    cv2_format = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)/255.
    background = background / 255.
    
    out = np.clip(background * cv2_format * 255, 0, 255).astype('uint8')

    return  out