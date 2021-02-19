#############################################################################################################
##
##  Parameters
##
#############################################################################################################
import numpy as np

class Parameters():

    x_size = 512
    y_size = 256
    resize_ratio = 8
    grid_x = x_size//resize_ratio  #64
    grid_y = y_size//resize_ratio  #32
    feature_size = 4
    threshold_point = 0.81 #0.35 #0.5 #0.57 #0.64 #0.35
    threshold_instance = 0.08

    # test parameter
    color = [(0,0,0), (255,0,0), (0,255,0),(0,0,255),(255,255,0),
            (255,0,255),(0,255,255),(255,255,255),(100,255,0),
            (100,0,255),(255,100,0),(0,100,255),(255,0,100),(0,255,100)]
    grid_location = np.zeros((grid_y, grid_x, 2))
    for y in range(grid_y):
        for x in range(grid_x):
            grid_location[y][x][0] = x
            grid_location[y][x][1] = y
