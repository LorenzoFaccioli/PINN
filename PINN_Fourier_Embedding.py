#------------------ Necessary imports --------------------------------
import torch
import torch.autograd as autograd
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils import clip_grad_norm_

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

torch.set_default_dtype(torch.float)
# defining boundaries of the domain and number of discretization points
x_min = 0
x_max = 1

#N_train: Number of training points # N_test: number of test points # Nf: Number of collocation points (Evaluate PDE)
N_train = 50
N_test = 200
Nf = 1000
def ground_truth(x):
  return torch.sin(2*np.pi*x) + torch.mul(torch.sin(50*np.pi*x), 0.1)

x_train = torch.linspace(x_min, x_max, N_train).view(-1,1)  #.view is equivalent to .reshape, the resulting tensor has size (N_train, 1)
x_PDE = torch.from_numpy(x_min+(x_max-x_min)*lhs(1, Nf)).type(torch.float)# 1 as the input is only x and
x_test = torch.linspace(x_min, x_max, N_test).view(-1,1)
x_bc = torch.Tensor([0,1]).view(-1,1).type(torch.float)

u_train = ground_truth(x_train)
u_test = ground_truth(x_test)  # evaluating the real function
u_bc = torch.Tensor([0, 0]).view(-1, 1).type(torch.float)

#adding noise to training data
noise = torch.randn(u_train.shape[0], 1)
u_train = u_train + noise



#------------------------ creating a Neural Network model ----------------------------
def init_weights(m):
  if isinstance(m, nn.Linear):
    torch.nn.init.xavier_uniform_(m.weight)
    m.bias.data.fill_(0.01)

class PINN(nn.Module):
  def __init__(self, hidden_units, input_size):
    super().__init__()

    self.loss_function = nn.MSELoss(reduction='mean')
    self.input_size = input_size

    self.layer_stack = nn.Sequential(
      nn.Linear(in_features=self.input_size, out_features=hidden_units),
      nn.Tanh(),
      nn.Linear(in_features=hidden_units, out_features=hidden_units),
      nn.Tanh(),
      nn.Linear(in_features=hidden_units, out_features=hidden_units),
      nn.Tanh(),
      nn.Linear(in_features=hidden_units, out_features=hidden_units),
      nn.Tanh(),
      nn.Linear(in_features=hidden_units, out_features=hidden_units),
      nn.Tanh(),
      nn.Linear(in_features=hidden_units, out_features=1),
      )

    self.layer_stack.apply(init_weights)  # initializing the network with Xavier initialization

  def lossTrain(self, x_train, u_train):
      loss_train = self.loss_function(self.forward(x_train), u_train)
      return(loss_train)

  def lossBC(self, x_bc, u_bc):
      loss_bc = self.loss_function(self.forward(x_bc), u_bc)
      return(loss_bc)

  # Loss PDE
  def lossPDE(self, x_PDE):
    x = x_PDE.clone().detach().requires_grad_(True)   # create a detached clone for differentiation
    u = self.forward(x.type(torch.float))
    u_x = autograd.grad(u, x, torch.ones_like(u).to(device), retain_graph=True, create_graph=True)[0]  # first derivative
    u_xx = autograd.grad(u_x, x, torch.ones_like(u_x).to(device), create_graph=True)[0]  # second derivative
    f = u_xx + torch.mul(torch.sin(2*np.pi*x), 4*np.pi**2) + torch.mul(torch.sin(50*np.pi*x), 250*np.pi**2)
    return self.loss_function(f, torch.zeros(f.shape[0], 1).to(device)) # make sure the target has the same shape as f

  def loss(self, x_train, u_train, x_bc, u_bc,  x_PDE):

      loss_pde = self.lossPDE(x_PDE)
      loss_bc = self.lossBC(x_bc, u_bc)
      loss_train = self.lossTrain(x_train, u_train)

      match WEIGHTS:
          case "static":
              weight_pde = 1e-2
              weight_bc = 1e6
              weight_train = 1
              return loss_train * weight_train, loss_bc * weight_bc, loss_pde * weight_pde, (loss_pde * weight_pde + loss_train * weight_train + loss_bc * weight_bc)

          case "adaptive simple":
            weight_pde = loss_pde/(loss_pde + loss_train + loss_bc)
            weight_bc = loss_bc/(loss_pde + loss_train + loss_bc)
            weight_train = loss_train/(loss_pde + loss_train + loss_bc)
            return loss_train*weight_train, loss_bc*weight_bc, loss_pde*weight_pde, (loss_pde*weight_pde + loss_train*weight_train + loss_bc*weight_bc)

  def forward(self, x: torch.Tensor):
      if x.shape[1] == self.input_size:
          return self.layer_stack(x)
      else:
          x_50 = fourier_encoder_50(x)
          x_2 = fourier_encoder_2(x)
          X = torch.hstack((x_2, x_50))
          return self.layer_stack(X)

#------------------------- train the neural network ------------------------------------
torch.manual_seed(42) # pseudo-randomization for reproducible code
torch.set_default_dtype(torch.float)

HIDDEN_UNITS = 100
LR = 1e-4
DECAY_RATE = 0.96
EPOCHS_BEFORE_DECAY = 10000
EPOCHS = 50000
ENCODED_SIZE = 2
INPUT_SIZE = 4*ENCODED_SIZE
WEIGHTS = "static"


# Fourier embedding of the inputs
B_50 = 50*torch.randn(ENCODED_SIZE, 1)
B_2 = 2*torch.randn(ENCODED_SIZE,1)

fourier_encoder_50 = rff.layers.GaussianEncoding(b=B_50)
fourier_encoder_2 = rff.layers.GaussianEncoding(b=B_2)

x_test_50 = fourier_encoder_50(x_test)    # returns a tensor of shape (N, 2*encoded_size)
x_test_2 = fourier_encoder_2(x_test)
x_test_encoded = torch.hstack((x_test_2, x_test_50))

x_train_50 = fourier_encoder_50(x_train)    # returns a tensor of shape (N, 2*encoded_size)
x_train_2 = fourier_encoder_2(x_train)
x_train_encoded = torch.hstack((x_train_2, x_train_50))

x_bc_50 = fourier_encoder_50(x_bc)
x_bc_2 = fourier_encoder_2(x_bc)
x_bc_encoded = torch.hstack((x_bc_2, x_bc_50))


#u_train = torch.vstack((u_train, u_train))
#u_test = torch.vstack((u_test, u_test))


# send data to GPU
x_train = x_train.to(device).type(torch.float)
x_PDE = x_PDE.to(device).type(torch.float)
x_test = x_test.to(device).type(torch.float)
x_bc = x_bc.to(device).type(torch.float)
x_train_encoded = x_train_encoded.to(device).type(torch.float)
x_test_encoded = x_test_encoded.to(device).type(torch.float)
x_bc_encoded = x_bc_encoded.to(device).type(torch.float)


u_train = u_train.to(device).type(torch.float)
u_test = u_test.to(device).type(torch.float)
u_bc = u_bc.to(device).type(torch.float)

torch.manual_seed(42) # pseudo-randomization for reproducible code
torch.set_default_dtype(torch.float)

# creating an instance of the PINN
model = PINN(hidden_units=HIDDEN_UNITS, input_size=INPUT_SIZE).to(device)
#print(model)

# defining an optimizer and learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=LR, amsgrad=False)
#optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)
lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=DECAY_RATE)

# define empty lists to track the value of the losses
PDE_loss_values = []
train_loss_values = []
total_loss_values = []
test_loss_values = []
BC_loss_values = []

u_evolution = torch.zeros(x_test_encoded.shape[0], 1)

epoch_count = []

for epoch in range(EPOCHS):

    # keeping track of the number of epoch
    epoch_count.append(epoch)

    PDE_loss = 0
    total_loss = 0
    train_loss = 0
    BC_loss = 0

    # Training
    model.train()

    # calculate the losses
    #PDE_loss += model.lossPDE(x_PDE).item()
    #BC_loss += model.lossBC(x_BC, u_BC).item()
    train_loss, BC_loss, PDE_loss, total_loss = model.loss(x_train_encoded, u_train, x_bc_encoded, u_bc, x_PDE)

    PDE_loss_values.append(PDE_loss.item())
    total_loss_values.append(total_loss.item())
    train_loss_values.append(train_loss.item())
    BC_loss_values.append(BC_loss.item())

    # setting to zero the gradients
    optimizer.zero_grad()

    # backward propagation and optimization
    total_loss.backward(retain_graph=True)
    #clip_grad_norm_(model.parameters(), 1)  # avoiding exploding/vanishing gradient
    optimizer.step()

    # adjusting the learning rate
    if epoch % EPOCHS_BEFORE_DECAY == 0:
        lr_scheduler.step()


    if epoch % 500 == 0:

       # Testing
        test_loss = 0
        model.eval()  # put model in eval mode

        # Turn on inference context manager
        with torch.inference_mode():

                # Forward pass
                u_pred = model.forward(x_test_encoded)
                u_evolution = torch.hstack((u_evolution, u_pred))

                # Calculate loss
                test_loss = model.loss_function(u_pred, u_test)

        test_loss_values.append(test_loss)

        print(f"Epoch: {epoch} | PDE loss: {PDE_loss:.5f} | Train loss: {train_loss:.5f} | BC loss: {BC_loss:.5f} | Total loss: {total_loss:.5f}\n")

print(f"Epoch: {epoch_count[-1]+1} | PDE loss: {PDE_loss_values[-1]:.5f} | Train loss: {train_loss_values[-1]:.5f} | BC loss: {BC_loss_values[-1]:.5f} | Total loss: {total_loss_values[-1]:.5f}\n")

#---------------------------------- Plotting the results -------------------------------------
u_hat = model.forward(x_test_encoded.to(device))

u_hat = u_hat.detach().cpu()
u_test = u_test.detach().cpu()
u_train = u_train.detach().cpu()

error = np.subtract(u_test, u_hat)

fig_1 = plt.figure(10)
plt.plot(x_train.detach().cpu(), u_train, label="Noisy data", color='red')
plt.plot(x_test.detach().cpu(), u_test, label="Exact", color='orange')
plt.plot(x_test.detach().cpu(), u_hat, label="Final estimate", linestyle='dashed', linewidth=1.2, color='k')
#plt.plot(x_test.detach().cpu(), u_evolution[:, 1], label="500 epochs", linestyle='dashed', color='b')
#plt.plot(x_test.detach().cpu(), u_evolution[:, 2], label="1000 epochs", linestyle='dashed', color='red')
plt.title("u(x)")
plt.xlabel("x")
plt.ylabel("u(x)")
plt.legend()
plt.show()
fig_1.savefig('images/Noisy_data/prediction')


fig_2 = plt.figure(20)
plt.plot(x_test.detach().cpu(), error)
plt.title("Point_wise_error")
plt.xlabel("x")
plt.ylabel("error")
plt.ticklabel_format(style='sci', axis='y')
plt.show()
fig_2.savefig('images/Noisy_data/error')

fig_3 = plt.figure(100)
plt.semilogy(epoch_count, BC_loss_values, label="BC loss")
plt.semilogy(epoch_count, PDE_loss_values, label="PDE loss")
plt.semilogy(epoch_count, total_loss_values, label="total loss")
plt.semilogy(epoch_count, train_loss_values, label="train loss")
plt.title("Loss functions")
plt.xlabel("epochs")
plt.ylabel("losses")
plt.legend()
plt.show()
fig_3.savefig('images/Noisy_data/loss_functions')

#--------------------------------- Save the model ------------------------------------------------

from pathlib import Path

# 1. Create models directory
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# 2. Create model save path
MODEL_NAME = "Noisy_data.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# 3. Save the model state dict
print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model.state_dict(), # only saving the state_dict() only saves the models learned parameters
           f=MODEL_SAVE_PATH)
