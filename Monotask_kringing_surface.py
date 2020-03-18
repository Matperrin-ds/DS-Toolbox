import gpytorch
import torch
import math
import numpy as np
import matplotlib.gridspec as gridspec
from mpl_toolkits import mplot3d
from matplotlib import pyplot as plt

############################
# Useful functions
############################

# Define a plotting function
def ax_plot(f, ax, y_labels, title, colormap = "coolwarm"):
    im = ax.imshow(y_labels, cmap = colormap)
    ax.set_title(title)
    f.colorbar(im)

############################
# Model
############################

class GridGPRegressionModel(gpytorch.models.ExactGP):
    def __init__(self, grid, train_x, train_y, likelihood):
        super(GridGPRegressionModel, self).__init__(train_x, train_y, likelihood)
        num_dims = train_x.size(-1)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.GridKernel(gpytorch.kernels.RBFKernel(), grid=grid)
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def fit(self, train_x, train_y, training_iter = 100, plot = True, verbose = True):
        self.train()
        self.likelihood.train()
        # Use the adam optimizer
        optimizer = torch.optim.Adam([
            {'params': self.parameters()},  # Includes GaussianLikelihood parameters
        ], lr=0.1)

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, self)

        # Store Loss, Lengthscale and noise
        Loss_vector = []
        Lscale_vector = []
        noise_vector = []

        for i in range(training_iter):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = self(train_x)
            # Calc loss and backprop gradients
            loss = -mll(output, train_y)
            loss.backward()
            print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
                i + 1, training_iter, loss.item(),
                self.covar_module.base_kernel.lengthscale.item(),
                self.likelihood.noise.item()
            ))
            optimizer.step()
            # Print results if required
            if verbose == True :
                print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
                    i + 1, training_iter, loss.item(),
                    self.covar_module.base_kernel.lengthscale.item(),
                    self.likelihood.noise.item()
                ))
            
            Loss_vector.append(loss.item())
            Lscale_vector.append(self.covar_module.base_kernel.lengthscale.item())
            noise_vector.append(self.likelihood.noise.item())
            

        # plot training results if required
        if plot == True :
            # Create 2x2 sub plots
            gs = gridspec.GridSpec(2, 2)
            fig = plt.figure()
            ax1 = fig.add_subplot(gs[0, 0]) # row 0, col 0
            ax1.plot(Lscale_vector)
            ax1.set_xlabel("Iterations")
            ax1.set_ylabel("Lengthscale")

            ax2 = fig.add_subplot(gs[0, 1]) # row 0, col 1
            ax2.plot(noise_vector)
            ax2.set_xlabel("Iterations")
            ax2.set_ylabel("Noise")

            ax3 = fig.add_subplot(gs[1, :]) # row 1, span all columns
            ax3.plot(Loss_vector)
            ax3.set_xlabel("Iterations")
            ax3.set_ylabel("Loss")

            plt.show()

    def predict(self, test_x):
        self.eval()
        self.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
             observed_pred = self.likelihood(self(test_x))
        return(observed_pred)

    def plot_prediction(self, train_x, train_y, test_x):
        observed_pred = self.predict(test_x)
        with torch.no_grad():
            # Initialize plot
            fig = plt.figure()
            ax = plt.axes(projection='3d')
            # Plot training data as red stars
            ax.scatter(train_x.numpy()[:,0], train_x.numpy()[:,1], train_y.numpy(), color = "orangered", marker = "o")
            # Plot predictive means as blue line
            ax.scatter(test_x.numpy()[:,0], test_x.numpy()[:,1], observed_pred.mean.numpy(), color = "darkslategray", marker = ".")


    def prediction_heatmap(self, train_x, train_y, test_x):
        observed_pred = self.predict(test_x)
        lower, upper = observed_pred.confidence_region()
        margin = (upper - lower) / 2
        n = int(
            ( observed_pred.mean.numpy().shape[0] )**0.5
            )
        # Plot our predictive means
        f1, observed_ax1 = plt.subplots(1, 1, figsize=(4, 3))
        ax_plot(f1, observed_ax1, observed_pred.mean.view(n,n), 'Predicted Values')

        # Plot Margin
        f2, observed_ax2 = plt.subplots(1, 1, figsize=(4, 3))
        ax_plot(f2, observed_ax2, margin.view(n,n), 'Confidence Margin', colormap = "Reds")

        plt.show()

############################
# Data
############################

train_x = torch.rand(20,2)
grid = train_x.clone()
train_y = torch.sin((train_x[:, 0] + train_x[:, 1]) * (2 * math.pi)) + torch.randn_like(train_x[:, 0]).mul(0.01)
#test_x = torch.rand(400,2)
n = 20
test_x = torch.zeros(int(pow(n, 2)), 2)
for i in range(n):
    for j in range(n):
        test_x[i * n + j][0] = float(i) / (n-1)
        test_x[i * n + j][1] = float(j) / (n-1)

############################
# Model use
############################

likelihood = gpytorch.likelihoods.GaussianLikelihood()

model = GridGPRegressionModel(grid, train_x, train_y, likelihood)

model.fit(train_x, train_y, training_iter = 100)

model.plot_prediction(train_x, train_y, test_x)

model.prediction_heatmap(train_x, train_y, test_x)
