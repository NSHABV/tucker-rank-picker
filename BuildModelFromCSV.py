import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from matplotlib.colors import Normalize
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import pickle
from scipy.interpolate import griddata

data = pd.read_csv('metrics_practic_data_train.csv')

data['SAM_improvement'] = (data['SAM_before'] - data['SAM_after']) / data['SAM_before']
data['PSNR_Improvement'] = data['PSNR_after'] - data['PSNR_before']
data['SSIM_improvement'] = (data['SSIM_after'] - data['SSIM_before']) / data['SSIM_after']

filtered_data = data[
    (data['Feature_Loss_Metric'] < 0.25) &
    (data['PSNR_after'] >= 20) &
    (data['SSIM_after'] >= 0.4) &
    (data['SAM_improvement'] >= 0.3)
]

sorted_data = filtered_data.sort_values(
    by=['SAM_improvement', 'PSNR_Improvement', 'SSIM_improvement', 'Feature_Loss_Metric'],
    ascending=[False, False, False, True]
)

optimal_entries = sorted_data.groupby('PSNR_before').first().reset_index()

X = optimal_entries[['PSNR_before']]
y = optimal_entries['TuckerRank1']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = make_pipeline(
    PolynomialFeatures(degree=3),
    LinearRegression()
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
# with open('trained_model.pkl', 'wb') as f:
#     pickle.dump(model, f)

def predict_optimal_tucker_rank(psnr_before):
    return model.predict([[psnr_before]])[0]

print(f'Optimal TuckerRank1 for PSNR 17: {predict_optimal_tucker_rank(17):.2f}')

import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('metrics_practic_data_test.csv')

data['SAM_improvement'] = (data['SAM_before'] - data['SAM_after']) / data['SAM_before']
data['PSNR_Improvement'] = data['PSNR_after'] - data['PSNR_before']
data['SSIM_improvement'] = (data['SSIM_after'] - data['SSIM_before']) / data['SSIM_after']

filtered_data = data[
    (data['Feature_Loss_Metric'] < 0.25) &
    (data['PSNR_after'] >= 20) &
    (data['SSIM_after'] >= 0.4) &
    (data['SAM_improvement'] >= 0.3)
]

sorted_data = filtered_data.sort_values(
    by=['SAM_improvement', 'PSNR_Improvement', 'SSIM_improvement', 'Feature_Loss_Metric'],
    ascending=[False, False, False, True]
)

optimal_entries = sorted_data.groupby('PSNR_before').first().reset_index()
optimal_entries = filtered_data # Фильтр не нужен (мы хотим графики по неотфильтрованным данным)

X_full = optimal_entries[['PSNR_before']]
y_full = optimal_entries['TuckerRank1']
y_full_pred = model.predict(X_full)

psnr_range = np.linspace(X_full.min()[0], X_full.max()[0], 100).reshape(-1, 1)
tucker_pred_smooth = model.predict(psnr_range)

metrics = [
    ('SAM_improvement', 'plasma', 'SAM Improvement Ratio'),
    ('PSNR_Improvement', 'viridis', 'PSNR Improvement (dB)'),
    ('Feature_Loss_Metric', 'coolwarm', 'Feature Loss Metric'),
    ('SSIM_improvement', 'magma', 'SSIM Improvement')
]

fig, axs = plt.subplots(2, 2, figsize=(16, 14))
fig.suptitle('Optimal TuckerRank1 Selection Across Metrics', fontsize=16, y=1.02)

axs = axs.ravel()

for i, (metric, cmap, label) in enumerate(metrics):
    if metric not in optimal_entries.columns:
        print(f"Skipping {metric} (not found in data)")
        continue

    norm = Normalize(vmin=optimal_entries[metric].min(),
                     vmax=optimal_entries[metric].max())

    scatter = axs[i].scatter(
        optimal_entries['PSNR_before'],
        optimal_entries['TuckerRank1'],
        c=optimal_entries[metric],
        cmap=cmap,
        norm=norm,
        alpha=0.8,
        s=100,
        edgecolor='w',
        linewidth=0.5
    )

    xi = np.linspace(optimal_entries['PSNR_before'].min(), optimal_entries['PSNR_before'].max(), 100)
    yi = np.linspace(optimal_entries['TuckerRank1'].min(), optimal_entries['TuckerRank1'].max(), 100)
    zi = griddata((optimal_entries['PSNR_before'], optimal_entries['TuckerRank1']),
                  optimal_entries[metric],
                  (xi[None, :], yi[:, None]), method='cubic')

    contours = axs[i].contour(xi, yi, zi, levels=5, colors='k', linewidths=1, alpha=0.5)
    axs[i].clabel(contours, inline=True, fontsize=8, fmt='%1.1f')

    axs[i].plot(
        psnr_range, tucker_pred_smooth,
        'k--', lw=2,
        label='Model Prediction'
    )

    cbar = plt.colorbar(scatter, ax=axs[i])
    cbar.set_label(label, fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    cbar.add_lines(contours)

    axs[i].set_xlabel('PSNR Before Denoising', fontsize=10)
    axs[i].set_ylabel('Optimal TuckerRank1', fontsize=10)
    axs[i].set_title(f'Colored by: {label} with Isolines', fontsize=12)
    axs[i].grid(True, linestyle='--', alpha=0.3)
    axs[i].legend(fontsize=8)

plt.tight_layout()
plt.show()