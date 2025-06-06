import time

import cv2
import tifffile
import numpy as np
import tensorly as tl
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

def find_homography_withInliers(img1, img2):
    orb = cv2.ORB_create(nfeatures=1000)

    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    matches = bf.match(des1, des2)

    matches = sorted(matches, key=lambda x: x.distance)

    good = matches[:min(500, len(matches))]

    if len(good) > 10:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        mask = mask.ravel()

        inlier_src_pts = src_pts[mask == 1]
        inlier_dst_pts = dst_pts[mask == 1]

        inlier_count = np.sum(mask)
        outlier_count = len(src_pts) - inlier_count

        return H, inlier_count, inlier_src_pts, inlier_dst_pts
    else:
        return None, 0, None, None

def load_image_data():
    rgb = cv2.imread('data/TestPatch2/thumbnail.jpg', cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    ms_files = tifffile.imread('data/TestPatch2/spectre.TIF')

    ms_tensor = np.stack(ms_files, axis=-1)

    return ms_tensor, rgb

def add_gaussian_noise(image, mean=0, std=0.05):
    noise = np.random.normal(mean, std, image.shape).astype(np.float32)
    noisy_image = np.clip(image + noise, 0, 1)
    return noisy_image

def add_salt_pepper_noise(image, prob=0.01):
    noisy_image = np.copy(image)
    salt_pepper = np.random.rand(*image.shape)
    noisy_image[salt_pepper < prob/2] = 0.0
    noisy_image[salt_pepper > 1 - prob/2] = 1.0
    return noisy_image

def add_poisson_noise(image):
    scaled_image = image * 255.0
    noisy_image = np.random.poisson(scaled_image) / 255.0
    return np.clip(noisy_image, 0, 1).astype(np.float32)

def add_striping_noise(image, intensity=0.05, direction='horizontal'):
    noisy_image = np.copy(image)
    h, w, c = image.shape
    stripe = np.random.uniform(1 - intensity, 1 + intensity, size=(h if direction == 'horizontal' else w))

    if direction == 'horizontal':
        for i in range(h):
            noisy_image[i, :, :] *= stripe[i]
    else:
        for j in range(w):
            noisy_image[:, j, :] *= stripe[j]

    return np.clip(noisy_image, 0, 1).astype(np.float32)

def add_multispectral_noise(image, gauss_std=0.03, poisson_scale=True, salt_pepper_prob=0.01, striping_intensity=0.05):
    noisy_image = add_gaussian_noise(image, std=gauss_std)
    if poisson_scale:
        noisy_image = add_poisson_noise(noisy_image)
    if salt_pepper_prob > 0:
        noisy_image = add_salt_pepper_noise(noisy_image, prob=salt_pepper_prob)
    if striping_intensity > 0:
        noisy_image = add_striping_noise(noisy_image, intensity=striping_intensity)
    return noisy_image

def psnr(original, denoised, max_val=1.0):
    mse = np.mean((original - denoised) ** 2)
    return 10 * np.log10(max_val**2 / mse)

def sam(original, denoised):
    dot_product = np.sum(original * denoised, axis=2)
    norm_original = np.sqrt(np.sum(original**2, axis=2))
    norm_denoised = np.sqrt(np.sum(denoised**2, axis=2))
    angle = np.arccos(dot_product / (norm_original * norm_denoised + 1e-10))
    return np.mean(angle) * 180 / np.pi


def save_metrics_to_file(tensorrank1, tensorrank2, tensorrank3, psnr_values, psnr_noised, inlier_count_denoised, inlier_noised, ssim_values, ssim_values_noised, sam_value, sam_value_noised, elapsed_time,
                         filename='metrics_practic_data_test_3.txt'):
    with open(filename, 'a') as f:
        f.write(f"{tensorrank1} x {tensorrank2} x {tensorrank3}; ")
        f.write(f"{np.mean(psnr_values):.2f}; ")
        f.write(f"{np.mean(psnr_noised):.2f}; ")
        f.write(f"{np.mean(inlier_count_denoised):.2f}; ")
        f.write(f"{np.mean(inlier_noised):.2f}; ")
        f.write(f"{np.mean(ssim_values):.3f}; ")
        f.write(f"{np.mean(ssim_values_noised):.3f}; ")
        f.write(f"{sam_value:.2f}; ")
        f.write(f"{sam_value_noised:.2f}; ")
        f.write(f"{elapsed_time:.2f}\n")
        f.write("\n")
        f.flush()

def getInliers(rgb_data_orig, rgb_dest_orig):
    rgb_data = rgb_data_orig.astype(np.float32)
    rgb_dest = rgb_dest_orig.astype(np.float32)

    gray_origin = cv2.cvtColor(rgb_data, cv2.COLOR_BGR2GRAY)
    gray_dest = cv2.cvtColor(rgb_dest, cv2.COLOR_BGR2GRAY)

    gray_origin = cv2.normalize(gray_origin, None, 0, 255, cv2.NORM_MINMAX)
    gray_dest = cv2.normalize(gray_dest, None, 0, 255, cv2.NORM_MINMAX)

    gray_origin = gray_origin.astype(np.uint8)
    gray_dest = gray_dest.astype(np.uint8)

    H, inlier_count, inlier_src_pts, inlier_dst_pts  = find_homography_withInliers(gray_dest, gray_origin)
    return H, inlier_count, inlier_src_pts, inlier_dst_pts

def generate_model_orb_tucker_data(ranks = (5, 5, 31), point_count = 600):
    ms_data, rgb_data = load_image_data()
    ms_data = np.concatenate([ms_data, rgb_data], axis=2)

    array = np.linspace(2, 7.5, num=point_count)

    psnr_values = []
    point_values = []

    for blurfactor in array:
        new_ms_data = np.empty(ms_data.shape)
        old_ms_data = np.empty(ms_data.shape)
        for i in range(new_ms_data.shape[2]):
            sub_array = ms_data[:, :, i]
            data_min = np.min(sub_array)
            data_max = np.max(sub_array)
            normalized = (sub_array - data_min) / (data_max - data_min + 1e-10)
            old_ms_data[:, :, i] = normalized
            normalized = add_multispectral_noise(normalized, gauss_std=0.03 * blurfactor, poisson_scale=False, salt_pepper_prob=0.01 * blurfactor, striping_intensity=0)
            new_ms_data[:, :, i] = normalized
        old_rgb_data = old_ms_data[:, :, -3:]
        noisy_rgb = new_ms_data[:, :, -3:]

        tl.set_backend('numpy')

        psnr_values_old = [psnr(old_ms_data[:, :, i], new_ms_data[:, :, i]) for i in range(new_ms_data.shape[2])]
        ssim_values_old = []
        for i in range(new_ms_data.shape[2]):
            ssim_val = ssim(old_ms_data[:, :, i], new_ms_data[:, :, i], data_range=1.0, win_size=7)
            ssim_values_old.append(ssim_val)

        sam_value_old = sam(old_ms_data, new_ms_data)

        start_time = time.time()
        core, factors = tl.decomposition.tucker(
            new_ms_data,
            rank=ranks,
            init='svd',
            tol=1e-5,
            n_iter_max=75
        )

        denoised = tl.tucker_to_tensor((core, factors))
        rgb_denoised = denoised[:, :, -3:]

        end_time = time.time()
        elapsed_time = end_time - start_time

        H, inlier_count_old, inlier_src_pts_noisy, inlier_dst_pts_noisy = getInliers(old_rgb_data, noisy_rgb)
        if inlier_count_old == 0:
            break

        H, inlier_count_new, inlier_src_pts_noisy, inlier_dst_pts_noisy = getInliers(old_rgb_data, rgb_denoised)

        psnr_values = [psnr(old_ms_data[:, :, i], denoised[:, :, i]) for i in range(new_ms_data.shape[2])]

        ssim_values = []
        for i in range(new_ms_data.shape[2]):
            ssim_val = ssim(old_ms_data[:, :, i], denoised[:, :, i], data_range=1.0, win_size=7)
            ssim_values.append(ssim_val)

        sam_value = sam(old_ms_data, denoised)
        print(f"Blurfactor done: {blurfactor} \n")
        print("\n")

        save_metrics_to_file(ranks[0], ranks[1], ranks[2], psnr_values, psnr_values_old, inlier_count_new, inlier_count_old, ssim_values, ssim_values_old, sam_value, sam_value_old, elapsed_time)

    return psnr_values, point_values

def tiff_inlier_comparison(ranks = (5, 5, 227), point_count = 600):
    ms_data, rgb_data = load_image_data()
    ms_data = np.concatenate([ms_data, rgb_data], axis=2)

    array = np.linspace(0.25, 7, num=point_count)
    array = [8]
    for blurfactor in array:
        new_ms_data = np.empty(ms_data.shape)
        old_ms_data = np.empty(ms_data.shape)
        for i in range(new_ms_data.shape[2]):
            sub_array = ms_data[:, :, i]
            data_min = np.min(sub_array)
            data_max = np.max(sub_array)
            normalized = (sub_array - data_min) / (data_max - data_min + 1e-10)
            old_ms_data[:, :, i] = normalized
            normalized = add_multispectral_noise(normalized, gauss_std=0.03 * blurfactor, poisson_scale=False, salt_pepper_prob=0.01 * blurfactor, striping_intensity=0)
            new_ms_data[:, :, i] = normalized
        old_rgb_data = old_ms_data[:, :, -3:]
        noisy_rgb = new_ms_data[:, :, -3:]

        tl.set_backend('numpy')

        psnr_values_old = [psnr(old_ms_data[:, :, i], new_ms_data[:, :, i]) for i in range(34)]
        ssim_values_old = []
        for i in range(new_ms_data.shape[2]):
            ssim_val = ssim(old_ms_data[:, :, i], new_ms_data[:, :, i], data_range=1.0, win_size=7)
            ssim_values_old.append(ssim_val)

        sam_value_old = sam(old_ms_data, new_ms_data)

        start_time = time.time()
        core, factors = tl.decomposition.tucker(
            new_ms_data,
            rank=ranks,
            init='svd',
            tol=1e-5,
            n_iter_max=75
        )

        denoised = tl.tucker_to_tensor((core, factors))
        rgb_denoised = denoised[:, :, -3:]
        end_time = time.time()
        elapsed_time = end_time - start_time

        H, inlier_count_old, inlier_src_pts_noisy, inlier_dst_pts_noisy = getInliers(old_rgb_data, noisy_rgb)

        H, inlier_count_new, inlier_src_pts_denoised, inlier_dst_pts_denoised = getInliers(old_rgb_data, rgb_denoised)

        print(f"Ranks done: {ranks}; Time: {elapsed_time} \n")
        print(f"Pre Denoise Inliers: {inlier_count_old}; After: {inlier_count_new} \n")
        print("\n")

        # if inlier_count_new > inlier_count_old:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        axes[0].imshow(noisy_rgb)
        # axes[0].scatter(
        #     inlier_dst_pts_noisy[:, 0, 0],  # X-coords (dest points)
        #     inlier_dst_pts_noisy[:, 0, 1],  # Y-coords (dest points)
        #     color='red',
        #     s=10,
        #     label=f'Noisy Inliers ({inlier_count_old})'
        # )
        axes[0].set_title('Noisy RGB Image (Dest: Noisy)')
        axes[0].axis('off')
        axes[0].legend()

        # axes[1].imshow(rgb_denoised)
        # axes[1].scatter(
        #     inlier_dst_pts_denoised[:, 0, 0],  # X-coords (dest points)
        #     inlier_dst_pts_denoised[:, 0, 1],  # Y-coords (dest points)
        #     color='lime',
        #     s=10,
        #     label=f'Denoised Inliers ({inlier_count_new})'
        # )
        # axes[1].set_title('Denoised RGB Image (Dest: Denoised)')
        # axes[1].axis('off')
        # axes[1].legend()

        axes[1].imshow(old_rgb_data)
        # Plot noisy inliers (src points)
        # axes[2].scatter(
        #     inlier_src_pts_noisy[:, 0, 0],  # X-coords (src points)
        #     inlier_src_pts_noisy[:, 0, 1],  # Y-coords (src points)
        #     color='red',
        #     s=10,
        #     label=f'Noisy Matches ({inlier_count_old})'
        # )
        # # Plot denoised inliers (src points)
        # axes[2].scatter(
        #     inlier_src_pts_denoised[:, 0, 0],  # X-coords (src points)
        #     inlier_src_pts_denoised[:, 0, 1],  # Y-coords (src points)
        #     color='lime',
        #     s=10,
        #     label=f'Denoised Matches ({inlier_count_new})'
        # )
        axes[1].set_title('Original RGB Image')
        axes[1].axis('off')
        axes[1].legend()

        plt.tight_layout()
        plt.show()

def tiff_decomp_psnr_chart(point_count = 600):
    ms_data, rgb_data = load_image_data()
    ms_data = np.concatenate([ms_data, rgb_data], axis=2)

    array = np.linspace(0.1, 8, num=point_count)

    psnr_values = []
    inlier_values = []

    for blurfactor in array:
        new_ms_data = np.empty(ms_data.shape)
        old_ms_data = np.empty(ms_data.shape)
        for i in range(new_ms_data.shape[2]):
            sub_array = ms_data[:, :, i]
            data_min = np.min(sub_array)
            data_max = np.max(sub_array)
            normalized = (sub_array - data_min) / (data_max - data_min + 1e-10)
            old_ms_data[:, :, i] = normalized
            normalized = add_multispectral_noise(normalized, gauss_std=0.03 * blurfactor, poisson_scale=False, salt_pepper_prob=0.01 * blurfactor, striping_intensity=0)
            new_ms_data[:, :, i] = normalized
        old_rgb_data = old_ms_data[:, :, -3:]
        noisy_rgb = new_ms_data[:, :, -3:]

        tl.set_backend('numpy')

        psnr_values_old = [psnr(old_ms_data[:, :, i], new_ms_data[:, :, i]) for i in range(new_ms_data.shape[2])]
        psnr_values.append(np.mean(psnr_values_old))
        H, inlier_count_old, inlier_src_pts_noisy, inlier_dst_pts_noisy = getInliers(old_rgb_data, noisy_rgb)
        inlier_values.append(inlier_count_old)

        print(f"Ranks done: {blurfactor}\n")

    plt.figure(figsize=(8, 5))
    plt.plot(psnr_values, inlier_values, 'bo-', label='PSNR vs. Inliers')
    plt.xlabel('PSNR (dB)')
    plt.ylabel('Number of Inliers')
    plt.title('Feature Matching Robustness vs. Image Quality (PSNR)')
    plt.grid(True)
    plt.legend()
    plt.show()

# Генерация данных для построения модели
# linspace_array = [10, 15, 20, 25, 30, 35, 40, 50, 60, 70]
# point_values = []
#
# for rankval in linspace_array:
#     generate_model_orb_tucker_data(ranks=(rankval, rankval, 227), point_count=23)

# Демонстрация зависимости PSNR от признаков ORB
# tiff_decomp_psnr_chart(point_count = 100)

# Функция для генерации визуального сравнения изображений
# tiff_inlier_comparison(ranks=(27, 27, 227), point_count=10)