import numpy as np
import cv2

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

def apply_mask(image, seg_img, color = [244, 35, 232], alpha=0.5):
    img = image.copy()
    mask = (seg_img == np.array([244, 35, 232]))[...,1].astype('uint8')
    mask = (cv2.resize(mask, (img.shape[1], img.shape[0])))
             
    for c in range(3):
        img[:,:,c]= np.where(mask == 1, img[:,:,c]*(1 - alpha)+alpha*color[c],img[:,:,c])
    return img

