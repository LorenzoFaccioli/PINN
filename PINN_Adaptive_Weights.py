#------------------ Necessary imports --------------------------------
import torch
import torch.autograd as autograd
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker
from sklearn.model_selection import train_test_split
import itertools

import numpy as np
import time
from pyDOE import lhs # Latin Cube Hypersampling
import rff

from plotting_functions import plot3D, plot3D_Matrix

#------------------- Make code device agnostic -----------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

#---------------------- Generate synthetic data ------------------------------

# defining boundaries of the domain and number of discretization points
x_min = 0
x_max = 1
y_min = 0
y_max = 1

N_x = 200
N_y = 200

#Nu: Number of training points # Nf: Number of collocation points (Evaluate PDE)
Nu = 500
Nf = 10000

def ground_truth(x,y):
  return torch.mul(torch.sin(2*np.pi*y), torch.sin(2*np.pi*x))

x = torch.linspace(x_min, x_max, N_x).view(-1,1)    #.view is equivalent to .reshape, the resulting tensor has size (Nx, 1)
y = torch.linspace(y_min, y_max, N_y).view(-1,1)

X,Y = torch.meshgrid(x.squeeze(1), y.squeeze(1))    # creating the mesh

u_real = ground_truth(X,Y)  # evaluating the real function


#----------------------- create a TEST dataset ---------------------------------

# transform the x,y into a 2-column feature vector (X is going to be used to denote the feature vector)
X_test = torch.hstack((X.transpose(1, 0).flatten()[:, None], Y.transpose(1, 0).flatten()[:, None])) # now x_test is a [N_x*N_y, 2] tensor
u_test = u_real.transpose(1,0).flatten()[:, None]  # Colum major Flatten (so we transpose it)

# extracting the domain bounds
lower_bound = X_test[0]  # tensor([0, 0])
upper_bound = X_test[-1]  # tensor[(1,1])

# ------------------------ create a TRAINING dataset (BC, collocation points) ----------------------

# Boundary Conditions (we will consider this as our training data)
left_edge = torch.hstack((X[0][:, None], Y[0][:, None])) # the None is so that they have the right dimensions [N_y,1]
right_edge = torch.hstack((X[-1][:, None], Y[-1][:, None]))
bottom_edge = torch.hstack((X[:, 0][:, None], Y[:, 0][:, None]))
top_edge = torch.hstack((X[:, -1][:, None], Y[:, -1][:, None]))

u_vertical_edges = torch.zeros(N_y, 1)
u_horizontal_edges = torch.zeros(N_x, 1)

# put all the BC data into one feature vector
X_BC = torch.vstack([left_edge, right_edge, bottom_edge, top_edge])
u_BC = torch.vstack([u_horizontal_edges, u_horizontal_edges, u_vertical_edges, u_vertical_edges])

# randomly sample Nu points on the boundary to use as our training data
idx = np.random.choice(X_BC.shape[0], Nu, replace=False).astype('float64') # it's randomly choosing Nu numbers between 0 and 800 (excluded) in this case
X_BC = X_BC[idx, :]
u_BC = u_BC[idx, :]

# define Nf the collocation points where to verify the PDE
X_PDE = lower_bound+(upper_bound-lower_bound)*lhs(2, Nf) # 2 as the inputs are x and y
X_PDE = torch.vstack((X_PDE, X_BC)) # Add the training points to the collocation points
u_PDE = torch.zeros(X_PDE.shape[0], 1)

#------------------------ creating a Neural Network model ----------------------------
def init_weights(m):
  if isinstance(m, nn.Linear):
    torch.nn.init.xavier_uniform_(m.weight)
    m.bias.data.fill_(0.01)

class PINN(nn.Module):
  def __init__(self, hidden_units):
    super().__init__()

    self.loss_function = nn.MSELoss(reduction='mean')

    self.layer_stack = nn.Sequential(
      nn.Linear(in_features=2, out_features=hidden_units),
      nn.Tanh(),
      nn.Linear(in_features=hidden_units, out_features=hidden_units),
      nn.Tanh(),
      nn.Linear(in_features=hidden_units, out_features=hidden_units),
      nn.Tanh(),
      nn.Linear(in_features=hidden_units, out_features=1),
      )

    self.layer_stack.apply(init_weights)  # initializing the network with Xavier initialization

  # Loss BC
  def lossBC(self, x_BC, u_BC):
    loss_BC = self.loss_function(self.forward(x_BC), u_BC)
    return loss_BC

  # Loss PDE
  def lossPDE(self, x_PDE):
    #g = x_PDE.clone().detach().requires_grad_(True)   # create a detached clone for differentiation
    x = x_PDE[:, [0]].requires_grad_(True).type(torch.float)
    y = x_PDE[:, [1]].requires_grad_(True).type(torch.float)
    u = self.forward(x,y)
    u_x = autograd.grad(u, x, torch.ones_like(u).to(device), retain_graph=True, create_graph=True)[0]  # first derivative
    u_y = autograd.grad(u, y, torch.ones_like(u).to(device), retain_graph=True, create_graph=True)[0]
    u_xx = autograd.grad(u_x, x, torch.ones_like(u_x).to(device), create_graph=True)[0]  # second derivative
    u_yy = autograd.grad(u_y, y, torch.ones_like(u_y).to(device), create_graph=True)[0]  # second derivative
    f = u_xx + u_yy + torch.mul(torch.mul(torch.sin(2*np.pi*x), torch.sin(2*np.pi*y)), 8*np.pi**2)
    return self.loss_function(f, torch.zeros(f.shape[0], 1).to(device)) # make sure the target has the same shape as f

  def loss(self, x_BC, u_BC, x_PDE):
    loss_bc = self.lossBC(x_BC, u_BC)
    loss_pde = self.lossPDE(x_PDE)

    if ADAPTIVE_WEIGHTS:

        weight_pde = loss_pde/(loss_pde + loss_bc)
        weight_bc = loss_bc/(loss_pde + loss_bc)
        return loss_pde*weight_pde, loss_bc*weight_bc, (loss_bc*weight_bc + loss_pde*weight_pde)

    else:
        return loss_pde, loss_bc*1e7, (loss_bc*1e7 + loss_pde)


  def forward(self, x: torch.Tensor, y:torch.Tensor=None):
    if y is not None:
        X = torch.hstack((x,y))
        return self.layer_stack(X)
    else:
        return self.layer_stack(x)

#------------------------- train the neural network ------------------------------------
torch.manual_seed(42) # pseudo-randomization for reproducible code
torch.set_default_dtype(torch.float)

HIDDEN_UNITS = 100
LR = 1e-4
DECAY_RATE = 0.96
EPOCHS_BEFORE_DECAY = 10000
EPOCHS = 50000
ADAPTIVE_WEIGHTS = True

# send data to GPU
X_PDE = X_PDE.to(device).type(torch.float)
u_PDE = u_PDE.to(device).type(torch.float)
X_BC = X_BC.to(device).type(torch.float)
u_BC = u_BC.to(device).type(torch.float)
X_test = X_test.to(device).type(torch.float)
u_test = u_test.to(device).type(torch.float)

torch.manual_seed(42) # pseudo-randomization for reproducible code
torch.set_default_dtype(torch.float)

# creating an instance of the PINN
model = PINN(hidden_units=HIDDEN_UNITS).to(device)
#print(model)

# defining an optimizer and learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=LR, amsgrad=False)
#optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=DECAY_RATE)


# define empty lists to track the value of the losses
PDE_loss_values = []
BC_loss_values = []
total_loss_values = []
test_loss_values = []

epoch_count = []

for epoch in range(EPOCHS):

    # keeping track of the number of epoch
    epoch_count.append(epoch)

    PDE_loss = 0
    BC_loss = 0
    total_loss = 0

    # Training
    model.train()

    # calculate the losses
    #PDE_loss += model.lossPDE(x_PDE).item()
    #BC_loss += model.lossBC(x_BC, u_BC).item()
    PDE_loss, BC_loss, total_loss = model.loss(X_BC, u_BC, X_PDE)

    PDE_loss_values.append(PDE_loss.item())
    BC_loss_values.append(BC_loss.item())
    total_loss_values.append(total_loss.item())

    # setting to zero the gradients
    optimizer.zero_grad()

    # backward propagation and optimization
    total_loss.backward(retain_graph=True)
    optimizer.step()

    # adjusting the learning rate
    if epoch % EPOCHS_BEFORE_DECAY == 0:
        lr_scheduler.step()


    if epoch % 100 == 0:

        # Testing
        test_loss = 0
        model.eval()  # put model in eval mode

        # Turn on inference context manager
        with torch.inference_mode():

                # Forward pass
                u_pred = model.forward(X_test)

                # Calculate loss
                test_loss = model.loss_function(u_pred, u_test)

        test_loss_values.append(test_loss)
        print(f"Epoch: {epoch} | PDE loss: {PDE_loss:.5f} | BC loss: {BC_loss:.5f} | Total loss: {total_loss:.5f} | Test loss: {test_loss:.5f}\n")

print(f"Epoch: {epoch_count[-1]+1} | PDE loss: {PDE_loss_values[-1]:.5f} | BC loss: {BC_loss_values[-1]:.5f} | Total loss: {total_loss_values[-1]:.5f} | Test loss: {test_loss_values[-1]:.5f}\n")

#---------------------------------- Plotting the results -------------------------------------
u_hat = model.forward(X_test.to(device))

u_hat = u_hat.reshape(N_x, N_y).detach().cpu()
u_real = u_real.reshape(N_x, N_y).detach().cpu()

error = np.subtract(u_real, u_hat)

x = X_test[:, 0]
y = X_test[:, 1]

x = x.reshape(shape=[N_x,N_y]).detach().cpu()
y = y.reshape(shape=[N_x,N_y]).detach().cpu()

plot3D_Matrix(x, y, u_hat, r"$\hat{u}(x,y)$", r"$\hat{u}(x,y)$")
plot3D_Matrix(x, y, u_real, "u(x,y)", "u(x,y)")
plot3D_Matrix(x, y, error, "Error", "error")

fig = plt.figure(10)
plt.semilogy(epoch_count, PDE_loss_values, label="PDE loss")
plt.semilogy(epoch_count, BC_loss_values, label="BC loss")
plt.semilogy(epoch_count, total_loss_values, label="total loss")
plt.title("Loss functions")
plt.xlabel("epochs")
plt.ylabel("losses")
plt.legend()
fig.savefig("Images/AdaptWeights/loss_functions_no_weight")


plt.show()

#--------------------------------- Save the model ------------------------------------------------

from pathlib import Path

# 1. Create models directory
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# 2. Create model save path
MODEL_NAME = "AdaptWeights.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# 3. Save the model state dict
print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model.state_dict(), # only saving the state_dict() only saves the models learned parameters
           f=MODEL_SAVE_PATH)








