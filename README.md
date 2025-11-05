# Constellaration Fusion Challenge
This project is in collaboration with Prof.Dr.Nabil Iqbal. Our goal is tackle the highly promising stellarator approach to nuclear fusion. More specifically this repository is dedicated to the hugging-face challenge: https://huggingface.co/blog/cgeorgiaw/constellaration-fusion-challenge . 

# (Non-technical) Overview 
The overall goal is to identify the optimal setup of the stellarator. The stellarator is a toric device designed to confine hot plasma for the purpose of achieving nuclear fusion with a standard reaction(deuterium + tritium -> helium + energy). Magnetic coils are twisted in a complex shape (which is to be determined) to ensure a magnetic field that traps the plasma in some way (that importantly avoids instabilities present in the more standard tokamak approach). Finding the best shape involves solving MHD/fluid and electromagnetic differential equations numerically. ML and or RL could search the space and potentially find the optimal construction (to be more precise we do supervised learning to find a relationship between fluid/plasma quantities and magnetic field quantities and then afterwards solve an optimization problem).

# More technical Overview
In this repository we only consider the Geometrically Optimized Stellarator Challenge of the overall challenge(the others will be in a different repository). We may think of the fluid as a g=1 handlebody with intrinsic coordinates given by so-called Boozer coordinates embedded in the 3d Lab-frame given in cylindrical coordinates. The Boozer coordinates give a D^2 x S^1 slicing foliation of the handlebody, such that we have periodic (theta,phi)-coordinates and a radial coordinate given by the poloidal flux.
For this challenge we have as input the boundary shape of the fluid given as Fourier coefficients of two of the cyclindrical coordinates expanded along (theta,phi). Three features contain more information of the fluid. The goal is to construct a model that minimizes the so-called elongation of the fluid. For this we first train a model to correctly predict the target from the four features (i.e elongation is part of the dataset;supervised training) and then we optimize for the smallest elongation while the three non-Fourier features are subject to constraints of the less-or-equal type.

# Our current approaches:
Overall we always have a standard training setup involving 4 features and one target i.e we train the features to predict the target elongation accurately on the test set. Then afterwards we have to optimize for minimal target i.e smallest elongation while preserving the constraints.



## BasicMLP:

Naturally we start with the simplest setup just to get a basic benchline (for the test loss). Just a (very) basic MLP, which takes in the main feature(fourier coefficients) and also the three frozen features. As loss function we take MSE loss. I still have to normalise the features in a correct way (which ideally takes into account the natural scaling of different units). For the optimization I also still might have to add a constraint/penalty on the fourier features as the optimization for smallest elongation might push them into a regime, which is outside of the set its trained on i.e the network is not reliable so we might have to put a "cut-off", have a closer look at the training data, which is related to normalising the features.
Moreover, we see many cases where the optimized value does not correspond to R and Z being smooth function.s
We should also do a search over good hyperparameters and add some different functional aspects to the MLP. For the moment we stick to optimization over the fourier coefficients, leaving the other features frozen. One could consider a different approach for which one optimizes in all four directions but adds constraints to the three "frozen" features to stay in a valid regime i.e fulfilling the 3 constraints.

## BoundaryCNN:

As a starting point to an approach, which is more naturally built on the physical setup, we implement a CNN of the boundary data. This means that we take in the 2 pairs of fourier coefficients and return the underlying local boundary fields (i.e undo the fourier transformation). These local fields are then treated via a periodic CNN. Conceptually speaking we are implementing the locality of the boundary data. A full geometric deep-learning setup would consider the complete 3d locality. We can see that the test loss is indeed lower than in the MLP approach.

## 3d Graph Neural Network: still be done.



# Folder Structure:
- `BasicMLP/` – **Basic MLP**: simple feedforward network to predict (and optimize) elongation. Serves as a baseline.  
- `BoundaryCNN/` – **Boundary CNN**: treats Fourier coefficients as local boundary fields for a periodic CNN. Captures locality of boundary shape and yields lower test loss.  
- Each `approach folder' has its own readme (still to be added).
- Each approach folder has its **own `notebooks/` subfolder** containing the script in one single runnable file.




