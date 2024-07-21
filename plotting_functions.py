import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
import numpy


def plot3D(x,t,y, title:str, z_label:str):
  x_plot =x.squeeze(1)
  t_plot =t.squeeze(1)
  X,T= torch.meshgrid(x_plot,t_plot)
  F_xt = y
  fig,ax = plt.subplots(1,1)
  cp = ax.contourf(X, T, F_xt,20,cmap="jet")
  fig.colorbar(cp) # Add a colorbar to a plot
  ax.set_title(title)
  ax.set_xlabel('x')
  ax.set_ylabel('y')
  plt.show()
  ax = plt.axes(projection='3d')
  ax.plot_surface(X.numpy(), T.numpy(), F_xt.numpy(),cmap="jet")
  ax.set_xlabel('x')
  ax.set_ylabel('y')
  ax.set_zlabel(z_label)


def plot3D_Matrix(x,y,u, title:str, z_label:str):
  X, Y = x, y
  F_xy = u
  fig,ax=plt.subplots(1,1)
  cp = ax.contourf(X, Y, F_xy, 20, cmap="jet")
  fig.colorbar(cp) # Add a colorbar to a plot
  ax.set_title(title)
  ax.set_xlabel('x')
  ax.set_ylabel('y')
  plt.show()
  ax = plt.axes(projection='3d')
  ax.plot_surface(X.numpy(), Y.numpy(), F_xy.numpy(), cmap="jet")
  ax.set_xlabel('x')
  ax.set_ylabel('y')
  ax.set_zlabel(z_label)
