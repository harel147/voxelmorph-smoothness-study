import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Load CSV files
sweep_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'sweep_lambda', 'summary.csv')
regu_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'regularizations', 'summary.csv')

sweep_df = pd.read_csv(sweep_path)
regu_df = pd.read_csv(regu_path)

# Plot 1: Scatter plot for regularization methods
plt.figure(figsize=(4, 3.5))
markers = ['o', 's', '^', 'D', 'v', '*']
colors = ['blue', 'green', 'red', 'purple', 'orange', 'cyan']
for i, regu in enumerate(regu_df['regularization']):
    plt.scatter(regu_df['val_loss'][i], regu_df['nonpositive_jacobian_mean'][i],
                label=regu, marker=markers[i % len(markers)],
                color=colors[i % len(colors)], s=150)

plt.xlabel("Validation MSE Loss")
plt.ylabel("Percentage of Non-Positive Jacobians")
plt.title("Regularization Type Effects")
plt.grid(True)
plt.legend()
plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:.0e}'))
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'regularizations', 'regularizations_scatter.png'))

# Plot 2: val_loss vs lambda
plt.figure(figsize=(6, 5))
plt.plot(sweep_df['lambda'], sweep_df['val_loss'], marker='o')
plt.xlabel("Lambda")
plt.ylabel("Validation MSE Loss")
plt.title("Validation Loss vs Lambda")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'sweep_lambda', 'sweep_val_loss_vs_lambda.png'))

# Plot 3: nonpositive_jacobian_mean vs lambda
plt.figure(figsize=(6, 5))
plt.plot(sweep_df['lambda'], sweep_df['nonpositive_jacobian_mean'], marker='o')
plt.xlabel("Lambda")
plt.ylabel("Percentage of Non-Positive Jacobians")
plt.title("Non-Positive Jacobians vs Lambda")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'sweep_lambda', 'sweep_jacobian_vs_lambda.png'))

