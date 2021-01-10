import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def new_xyz(disp, xl, yl, b = 0.5707 ,f = 645.24):
  z = (b*f)/(disp/1.6)
  x = (xl*z)/f
  y = (yl*z)/f

  dist = np.sqrt(x**2+y**2+z**2)
  return x, y, z, dist

def get_xyz(disparity, b = 0.5707 ,f = 645.24):
  disparity = disparity.astype(np.float32)
  disparity[disparity==0]= -1

  # calculate depth from disparity
  z = (b*f)/(disparity/1.6) 
  # calculate x,y
  grid = np.mgrid[0:disparity.shape[0],0:disparity.shape[1]]
  xl = grid[1] - disparity.shape[1]/2
  x = (xl*z)/f
  yl = grid[0] - disparity.shape[0]/2
  y = (yl*z)/f
  # delete empty pixels
  cliche = np.zeros(np.shape(disparity))
  
  cliche[disparity>0]= 1
  x = cliche*x
  y = cliche*y
  z = cliche*z

  xyz = np.zeros((x.shape[0],x.shape[1],3))
  xyz[:,:,0] = x
  xyz[:,:,1] = y
  xyz[:,:,2] = z
  return x,y,z
def create_pointcloud(x,y,z,ts):
  # create a grid mask for 3D-plot
  grid = np.mgrid[0:x.shape[0],0:x.shape[1]]
  mask = np.zeros_like(x)
  mask[(x!=0)&(y!=0)&(z!=0)] = 1
  mask[(grid[0]%ts!=0)|(grid[1]%ts!=0)] = 0 # sample every ts pixels
  x_sample = (x*mask).flatten(); x_sample = x_sample[x_sample!=0]
  y_sample = (y*mask).flatten(); y_sample = y_sample[y_sample!=0]
  z_sample = (z*mask).flatten(); z_sample = z_sample[z_sample!=0]
  sampleCount =x_sample.shape[0]; 
  #print('count of samples:',sampleCount)
  
  # prepare points for kmeans clustering and other processes
  X = np.zeros((sampleCount,3))
  X[:,0]=x_sample
  X[:,1]=y_sample
  X[:,2]=z_sample
  return X

def show_pointcloud(X,title=None):
  max_range = np.array([X[:,0].max()-X[:,0].min(), X[:,1].max()-X[:,1].min(), X[:,2].max()-X[:,2].min()]).max() / 2.0
  mid_x = (X[:,0].max()+X[:,0].min()) * 0.5
  mid_y = (X[:,1].max()+X[:,1].min()) * 0.5
  mid_z = (X[:,2].max()+X[:,2].min()) * 0.5
      
  fg = plt.figure()
  ax = Axes3D(fg)
  ax.scatter(X[:,0], X[:,1], X[:,2],edgecolor='k')
  ax.set_xlabel('X-axis')
  ax.set_ylabel('Y-axis')
  ax.set_zlabel('Z-axis')
  # set limit to equal
  ax.set_xlim(mid_x - max_range, mid_x + max_range)
  ax.set_ylim(mid_y - max_range, mid_y + max_range)
  ax.set_zlim(mid_z - max_range, mid_z + max_range)
  if title is not None:
      plt.title(title)
  # plt.show()