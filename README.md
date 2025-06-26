# VoxelMorph Smoothness Study

[**VoxelMorph**](https://github.com/voxelmorph/voxelmorph) is a learning-based framework for deformable image registration. It estimates spatial transformations between images using convolutional neural networks and is trained in an unsupervised manner using image similarity and deformation smoothness losses.

In this study, we investigate the effect of smoothness regularization in two key directions:

- **Regularization Weight Sweep**: Studying how increasing the weight λ of L2 regularization influences the trade-off between registration accuracy and deformation smoothness.
- **Regularization Type Comparison**: Comparing the effects of L1, L2, and Bending Energy regularizations on the resulting deformation fields.

We evaluate each experiment using:
- **Validation MSE Loss**: Measures alignment accuracy between registered image pairs.
- **Percentage of Non-Positive Jacobians**: Indicates local folding or irregularities in the deformation field (lower is smoother/more plausible).

A detailed summary of our findings, including analysis, and visualizations is available in our [**project report**](https://github.com/harel147/voxelmorph-smoothness-study/blob/main/our_project/VoxelMorph_Smoothness_Study_Course_Project.pdf).  

---

## Setup Instructions

We recommend using a Python virtual environment:

```bash
python -m venv venv
source venv/bin/activate
bash installation_script.sh
```

## Experiments

### 1. Regularization Weight Effect

To run the regularization weight sweep experiment (varying λ for L2 regularization), use:

```bash
python our_project/train_sweep_lambda.py
```
This script trains models using different values of the regularization weight λ and stores the results in:
```bash
models/sweep_lambda/
```

### 2. Regularization Type Comparison
To run the experiment comparing different regularization types (L1, L2, bending energy), use:
```bash
python our_project/train_different_regularizations.py
```
This script trains models for each regularization type and stores the results in:
```bash
models/regularizations/
```
## Results

After running the training scripts, you can visualize the results using the following command:

```bash
python our_project/visualize_results.py
```
This script reads the summary CSV files from the output directories and produces:

- A scatter plot showing the trade-off between validation loss (MSE) and the percentage of non-positive Jacobians for each regularization type.
Output: `models/regularizations/regularizations_scatter.png`.
- A line plot of validation loss (MSE) vs λ for L2 regularization.
Output: `models/sweep_lambda/sweep_val_loss_vs_lambda.png`.
- A line plot of non-positive Jacobians vs λ.
Output: `models/sweep_lambda/sweep_jacobian_vs_lambda.png`.
