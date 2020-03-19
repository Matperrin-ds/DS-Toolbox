import math
import torch
import gpytorch
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

############################
# Data
############################

# Time steps at which there are observations
train_x = torch.rand(15)
# Noisy sinus
train_y = torch.sin(train_x * (2 * math.pi)) + torch.randn(train_x.size()) * math.sqrt(0.04)
# Time steps at which we wish to infer y's value
test_x = torch.linspace(0, 1, 30)

############################
# Model
############################

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        # ConstantMean ne permet pas de faire du krigeage universel
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        # Kernel Radial Based Function

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def fit(self, train_x, train_y, training_iter = 100, plot = True, verbose = True):
        self.train()
        self.likelihood.train()

        # Use the adam optimizer
        optimizer = torch.optim.Adam([
        {'params': model.parameters()},  # Includes GaussianLikelihood parameters
        ], lr=0.1)

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        Loss_vector = []
        Lscale_vector = []
        noise_vector = []

        for i in range(training_iter):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = model(train_x)
            # Calc loss and backprop gradients
            loss = -mll(output, train_y)
            loss.backward()
    
            print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
                i + 1, training_iter, loss.item(),
                model.covar_module.base_kernel.lengthscale.item(),
                model.likelihood.noise.item()
            ))
    
            Loss_vector.append(loss.item())
            Lscale_vector.append(model.covar_module.base_kernel.lengthscale.item())
            noise_vector.append(model.likelihood.noise.item())
    
            optimizer.step()

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
            f, ax = plt.subplots(1, 1, figsize=(13, 8))

            # Get upper and lower confidence bounds
            lower, upper = observed_pred.confidence_region()
            # Plot training data as black stars
            ax.plot(train_x.numpy(), train_y.numpy(), 'r*')
            # Plot predictive means as blue line
            ax.plot(test_x.numpy(), observed_pred.mean.numpy(), 'b.')
            # Shade between the lower and upper confidence bounds
            ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
            ax.set_ylim([-3, 3])
            ax.legend(['Observed Data', 'Predicted data(Mean)', 'Confidence Interval'])

        plt.show()



############################
# Model use
############################

likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = ExactGPModel(train_x, train_y, likelihood)
model.fit(train_x, train_y, training_iter = 100, plot = True, verbose = True)
model.plot_prediction(train_x, train_y, test_x)