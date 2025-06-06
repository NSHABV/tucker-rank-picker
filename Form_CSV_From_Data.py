import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import csv
from collections import defaultdict

def parse_data(file_path):
    data_by_tucker_rank1 = defaultdict(list)

    with open(file_path, 'r') as file:
        for line in file:
            parts = [part.strip() for part in line.strip().split(';')]
            if len(parts) < 9:
                continue

            tucker_part = parts[0]
            tucker_ranks = [rank.strip() for rank in tucker_part.split('x')]
            if len(tucker_ranks) != 3:
                continue

            tucker_rank1 = tucker_ranks[0]

            try:
                record = {
                    'TuckerRank1': tucker_rank1,
                    'TuckerRank2': tucker_ranks[1],
                    'TuckerRank3': tucker_ranks[2],
                    'PSNR_after': float(parts[1]),
                    'PSNR_before': float(parts[2]),
                    'Inlier_Count_after': float(parts[3]),
                    'Inlier_Count_before': float(parts[4]),
                    'SSIM_after': float(parts[5]),
                    'SSIM_before': float(parts[6]),
                    'SAM_after': float(parts[7]),
                    'SAM_before': float(parts[8]),
                    'Time_per_operation': float(parts[9]) if len(parts) > 9 else None
                }
            except (ValueError, IndexError):
                continue

            if float(parts[3]) > 1.0 and float(parts[4]) < 250.0:
                data_by_tucker_rank1[tucker_rank1].append(record)

    return dict(data_by_tucker_rank1)

def plot_correlations_with_reference_lines(parsed_data):
    tucker_rank1_values = sorted([int(rank) for rank in parsed_data.keys()])
    norm = Normalize(vmin=min(tucker_rank1_values), vmax=max(tucker_rank1_values))
    cmap = plt.cm.get_cmap('viridis')

    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Correlations with Reference Lines', fontsize=16)

    for tucker_rank1, records in parsed_data.items():
        initial_psnr = [record['PSNR_before'] for record in records]
        denoised_psnr = [record['PSNR_after'] for record in records]
        psnr_diff = [after - before for before, after in zip(initial_psnr, denoised_psnr)]

        initial_orb = [record['Inlier_Count_before'] for record in records]
        denoised_org = [record['Inlier_Count_after'] for record in records]
        metric_orbdiff = [(before - after) / before * 625 / after / after for before, after in zip(initial_orb, denoised_org)]

        initial_sam = [record['SAM_before'] for record in records]
        denoised_sam = [record['SAM_after'] for record in records]
        sam_diff = [(before - after) / before for before, after in zip(initial_sam, denoised_sam)]

        initial_ssim = [record['SSIM_before'] for record in records]
        denoised_ssim = [record['SSIM_after'] for record in records]

        color = cmap(norm(int(tucker_rank1)))

        axs[0, 0].scatter(initial_psnr, denoised_psnr, color=color, alpha=0.6, label=f'TuckerRank1={tucker_rank1}')
        axs[0, 0].set_title('Initial PSNR vs. Denoised PSNR')
        axs[0, 0].set_xlabel('Initial PSNR')
        axs[0, 0].set_ylabel('Denoised PSNR')
        axs[0, 0].grid(True)

        axs[0, 1].scatter(initial_psnr, metric_orbdiff, color=color, alpha=0.6, label=f'TuckerRank1={tucker_rank1}')
        axs[0, 1].set_title('Initial PSNR vs. Feature Metric (After - Before)')
        axs[0, 1].set_xlabel('Initial PSNR')
        axs[0, 1].set_ylabel('Feature Metric')
        axs[0, 1].set_yscale('log')
        axs[0, 1].grid(True)

        axs[1, 0].scatter(initial_psnr, sam_diff, color=color, alpha=0.6, label=f'TuckerRank1={tucker_rank1}')
        axs[1, 0].set_title('Initial PSNR vs. SAM Reduction')
        axs[1, 0].set_xlabel('Initial PSNR')
        axs[1, 0].set_ylabel('SAM Reduction')
        axs[0, 1].set_yscale('log')
        axs[1, 0].grid(True)

        axs[1, 1].scatter(initial_psnr, denoised_ssim, color=color, alpha=0.6, label=f'TuckerRank1={tucker_rank1}')
        axs[1, 1].set_title('Initial PSNR vs. Denoised SSIM')
        axs[1, 1].set_xlabel('Initial PSNR')
        axs[1, 1].set_ylabel('Denoised SSIM')
        axs[1, 1].grid(True)

    x_min, x_max = axs[0, 0].get_xlim()
    y_min, y_max = axs[0, 0].get_ylim()
    lim_min = min(x_min, y_min)
    lim_max = max(x_max, y_max)
    axs[0, 0].plot([lim_min, lim_max], [lim_min, lim_max], 'k--', label='x = y')
    axs[0, 0].axhline(y=20, color='r', linestyle=':', label='y = 20')
    axs[0, 0].axhline(y=25, color='b', linestyle=':', label='y = 25')
    axs[0, 0].set_xlim(lim_min, lim_max)
    axs[0, 0].set_ylim(lim_min, lim_max)

    axs[0, 1].axhline(y=0.25, color='g', linestyle=':', label='y = 15')

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.subplots_adjust(right=1.1)
    cbar = plt.colorbar(sm, ax=axs, orientation='vertical', fraction=0.02, pad=0.1)
    cbar.set_label('TuckerRank1 Value')

    plt.tight_layout()
    plt.savefig('correlations_with_reference_lines_test3.png', bbox_inches='tight')
    plt.close()

# file_path = 'metrics_practic_data_test_3.txt'
# parsed_data = parse_data(file_path)
# plot_correlations_with_reference_lines(parsed_data)

def parse_data_into_csv(file_path, output_csv_path):
    data_by_tucker_rank1 = defaultdict(list)
    all_records = []

    with open(file_path, 'r') as file:
        for line in file:
            parts = [part.strip() for part in line.strip().split(';')]
            if len(parts) < 9:
                continue

            tucker_part = parts[0]
            tucker_ranks = [rank.strip() for rank in tucker_part.split('x')]
            if len(tucker_ranks) != 3:
                continue

            tucker_rank1 = tucker_ranks[0]

            try:
                inlier_after = float(parts[3])
                inlier_before = float(parts[4])

                if inlier_before > 0 and inlier_after > 0:
                    feature_loss_metric = ((inlier_before - inlier_after) / inlier_before) * 625 / (
                                inlier_after * inlier_after)
                else:
                    feature_loss_metric = float('inf')

                record = {
                    'TuckerRank1': tucker_rank1,
                    'TuckerRank2': tucker_ranks[1],
                    'TuckerRank3': tucker_ranks[2],
                    'PSNR_after': float(parts[1]),
                    'PSNR_before': float(parts[2]),
                    'Inlier_Count_after': inlier_after,
                    'Inlier_Count_before': inlier_before,
                    'SSIM_after': float(parts[5]),
                    'SSIM_before': float(parts[6]),
                    'SAM_after': float(parts[7]),
                    'SAM_before': float(parts[8]),
                    'Time_per_operation': float(parts[9]) if len(parts) > 9 else None,
                    'Feature_Loss_Metric': feature_loss_metric
                }

                if float(parts[3]) > 1.0 and float(parts[4]) < 250.0:
                    data_by_tucker_rank1[tucker_rank1].append(record)
                    all_records.append(record)
            except (ValueError, IndexError) as e:
                print(f"Skipping line due to error: {e}")
                continue

    if all_records:
        fieldnames = list(all_records[0].keys())
        with open(output_csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_records)
        print(f"Successfully wrote data to {output_csv_path}")
    else:
        print("No valid records found to write to CSV")

    return data_by_tucker_rank1

parsed_data = parse_data_into_csv('metrics_practic_data_test_3.txt', 'metrics_practic_data_test_3.csv')
