import time

import cv2
from glob import glob
import numpy as np
import tensorly as tl
from skimage.metrics import structural_similarity as ssim
import pywt
from skimage.restoration import denoise_nl_means
from sklearn.decomposition import PCA

def pca_denoise(new_ms_data, psnr=None):
    h, w, b = new_ms_data.shape
    X = new_ms_data.reshape(-1, b)

    n_components = 10 if (psnr is None or psnr > 25) else 5

    start_time = time.time()
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    X_denoised = pca.inverse_transform(X_pca)
    end_time = time.time()
    elapsed = end_time - start_time
    return X_denoised.reshape(h, w, b), elapsed


def hybrid_denoise(new_ms_data, psnr=None):
    start_time = time.time()
    wavelet_denoised = wavelet_denoise(new_ms_data, psnr)

    denoised, elapsed = pca_denoise(wavelet_denoised, psnr)
    end_time = time.time()
    elapsed = end_time - start_time
    return denoised, elapsed


def nlm_denoise(new_ms_data, psnr=None):
    denoised = np.empty_like(new_ms_data)

    h = 0.1 if (psnr is None or psnr > 30) else 0.2

    start_time = time.time()
    for b in range(new_ms_data.shape[2]):
        band = new_ms_data[..., b]
        denoised[..., b] = denoise_nl_means(
            band,
            patch_size=5,
            patch_distance=7,
            h=h * band.std()
        )
    end_time = time.time()
    elapsed = end_time - start_time
    return denoised, elapsed


def wavelet_denoise(new_ms_data, psnr=None):
    denoised = np.empty_like(new_ms_data)

    threshold = 0.05 if (psnr is None or psnr > 25) else 0.1
    start_time = time.time()
    for b in range(new_ms_data.shape[2]):
        coeffs = pywt.wavedec2(new_ms_data[..., b], 'db1', level=2)
        coeffs_thresh = [pywt.threshold(c, value=threshold, mode='soft')
                         for c in coeffs]
        denoised[..., b] = pywt.waverec2(coeffs_thresh, 'db1')
    end_time = time.time()
    elapsed = end_time - start_time
    return denoised, elapsed


def find_homography(img1, img2):
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

        return H, inlier_count
    else:
        return None, 0

def load_image_data(spectremaskname = 'data/TestPatch2/spectre.TIF', rgbmaskname = 'data/TestPatch2/thumbnail.jpg'):
    ms_files = sorted(glob(spectremaskname))
    ms_channels = [cv2.imread(f, cv2.IMREAD_UNCHANGED) for f in ms_files]

    ms_tensor = np.stack(ms_channels, axis=-1)

    rgb = cv2.imread(rgbmaskname, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    return ms_tensor, rgb


def add_gaussian_noise(image, mean=0, std=0.05):
    noise = np.random.normal(mean, std, image.shape).astype(np.float32)
    noisy_image = np.clip(image + noise, 0, 1)
    return noisy_image


def add_salt_pepper_noise(image, prob=0.01):
    noisy_image = np.copy(image)
    salt_pepper = np.random.rand(*image.shape)
    noisy_image[salt_pepper < prob / 2] = 0.0
    noisy_image[salt_pepper > 1 - prob / 2] = 1.0
    return noisy_image


def add_poisson_noise(image):
    scaled_image = image * 255.0
    noisy_image = np.random.poisson(scaled_image) / 255.0
    return np.clip(noisy_image, 0, 1).astype(np.float32)


def add_striping_noise(image, intensity=0.05, direction='horizontal'):
    """Add striping noise (common in push-broom sensors)."""
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
    """Combine all noise types for realistic multispectral noise."""
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
    return 10 * np.log10(max_val ** 2 / mse)


def sam(original, denoised):
    dot_product = np.sum(original * denoised, axis=2)
    norm_original = np.sqrt(np.sum(original ** 2, axis=2))
    norm_denoised = np.sqrt(np.sum(denoised ** 2, axis=2))
    angle = np.arccos(dot_product / (norm_original * norm_denoised + 1e-10))
    return np.mean(angle) * 180 / np.pi


def getInliers(rgb_data_orig, rgb_dest_orig):
    rgb_data = rgb_data_orig.astype(np.float32)
    rgb_dest = rgb_dest_orig.astype(np.float32)

    gray_origin = cv2.cvtColor(rgb_data, cv2.COLOR_BGR2GRAY)
    gray_dest = cv2.cvtColor(rgb_dest, cv2.COLOR_BGR2GRAY)

    gray_origin = cv2.normalize(gray_origin, None, 0, 255, cv2.NORM_MINMAX)
    gray_dest = cv2.normalize(gray_dest, None, 0, 255, cv2.NORM_MINMAX)

    gray_origin = gray_origin.astype(np.uint8)
    gray_dest = gray_dest.astype(np.uint8)

    H, inlier_count = find_homography(gray_dest, gray_origin)
    return H, inlier_count


def tucker_restore(new_ms_data, ranks=(50, 50, 227), device='cpu'):
    new_ms_data_tensor = tl.tensor(new_ms_data).to(device)
    start_time = time.time()
    core, factors = tl.decomposition.tucker(
        new_ms_data_tensor,
        rank=ranks,
        init='svd',
        tol=1e-5,
        n_iter_max=75,
        verbose=True
    )

    denoised_tensor = tl.tucker_to_tensor((core, factors))
    end_time = time.time()
    elapsed_time = end_time - start_time

    denoised = denoised_tensor.cpu().numpy()
    rgb_denoised = denoised[:, :, -3:]

    return denoised_tensor, rgb_denoised, elapsed_time


def generate_comparison_denoise_data(point_count=10):
    ms_data, rgb_data = load_image_data()
    ms_data = np.concatenate([ms_data, rgb_data], axis=2)

    array = np.linspace(1, 7, num=point_count)

    psnr_values = []
    point_values = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    old_ms_data = np.empty((128, 128, 227))
    for i in range(old_ms_data.shape[2]):
        sub_array = ms_data[:, :, i]
        data_min = np.min(sub_array)
        data_max = np.max(sub_array)
        normalized = (sub_array - data_min) / (data_max - data_min + 1e-10)
        old_ms_data[:, :, i] = normalized

    old_rgb_data = old_ms_data[:, :, -3:]
    new_ms_data = np.empty((128, 128, 227))
    for blurfactor in array:
        for i in range(old_ms_data.shape[2]):
            sub_array = old_ms_data[:, :, i]
            normalized = add_multispectral_noise(sub_array, gauss_std=0.03 * blurfactor, poisson_scale=False,
                                                 salt_pepper_prob=0.01 * blurfactor, striping_intensity=0)
            new_ms_data[:, :, i] = normalized
        noisy_rgb = new_ms_data[:, :, -3:]

        psnr_values_old = [psnr(old_ms_data[:, :, i], new_ms_data[:, :, i]) for i in range(227)]
        ssim_values_old = []
        for i in range(227):
            ssim_val = ssim(old_ms_data[:, :, i], new_ms_data[:, :, i], data_range=1.0, win_size=7)
            ssim_values_old.append(ssim_val)
        sam_value_old = sam(old_ms_data, new_ms_data)
        H, inlier_count_old = getInliers(old_rgb_data, noisy_rgb)
        if inlier_count_old == 0:
            break

        # Non-Blind Tucker
        denoised_tucker, rgb_denoised_tucker, elapsed_time_tucker = tucker_restore(new_ms_data)
        rgb_denoised_tucker = denoised_tucker[:, :, -3:]
        H, inlier_count_new = getInliers(old_rgb_data, rgb_denoised_tucker)
        psnr_values_tucker = [psnr(old_ms_data[:, :, i], denoised_tucker[:, :, i]) for i in range(227)]
        ssim_values_tucker = []
        for i in range(227):
            ssim_val = ssim(old_ms_data[:, :, i], denoised_tucker[:, :, i], data_range=1.0, win_size=7)
            ssim_values_tucker.append(ssim_val)
        sam_value_tucker = sam(old_ms_data, denoised_tucker)

        # Wavelet
        denoised_wavelet, elapsed_time_wavelet = wavelet_denoise(new_ms_data)
        rgb_denoised_wavelet = denoised_wavelet[:, :, -3:]
        H, inlier_count_wavelet = getInliers(old_rgb_data, rgb_denoised_wavelet)
        psnr_values_wavelet = [psnr(old_ms_data[:, :, i], denoised_wavelet[:, :, i]) for i in range(227)]
        ssim_values_wavelet = []
        for i in range(227):
            ssim_val = ssim(old_ms_data[:, :, i], denoised_wavelet[:, :, i],
                            data_range=1.0, win_size=7)
            ssim_values_wavelet.append(ssim_val)
        sam_value_wavelet = sam(old_ms_data, denoised_wavelet)

        # Non-Local-Means
        denoised_nlm, elapsed_time_nlm = nlm_denoise(new_ms_data)
        rgb_denoised_nlm = denoised_nlm[:, :, -3:]
        H, inlier_count_nlm = getInliers(old_rgb_data, rgb_denoised_nlm)
        psnr_values_nlm = [psnr(old_ms_data[:, :, i], denoised_nlm[:, :, i]) for i in range(227)]
        ssim_values_nlm = []
        for i in range(227):
            ssim_val = ssim(old_ms_data[:, :, i], denoised_nlm[:, :, i],
                            data_range=1.0, win_size=7)
            ssim_values_nlm.append(ssim_val)
        sam_value_nlm = sam(old_ms_data, denoised_nlm)

        # PCA
        denoised_pca, elapsed_time_pca = pca_denoise(new_ms_data)
        rgb_denoised_pca = denoised_pca[:, :, -3:]
        H, inlier_count_pca = getInliers(old_rgb_data, rgb_denoised_pca)
        psnr_values_pca = [psnr(old_ms_data[:, :, i], denoised_pca[:, :, i]) for i in range(227)]
        ssim_values_pca = []
        for i in range(227):
            ssim_val = ssim(old_ms_data[:, :, i], denoised_pca[:, :, i],
                            data_range=1.0, win_size=7)
            ssim_values_pca.append(ssim_val)
        sam_value_pca = sam(old_ms_data, denoised_pca)

        # Hybrid Wavelet + PCA
        denoised_hybrid, elapsed_time_hybrid = hybrid_denoise(new_ms_data)
        rgb_denoised_hybrid = denoised_hybrid[:, :, -3:]
        H, inlier_count_hybrid = getInliers(old_rgb_data, rgb_denoised_hybrid)
        psnr_values_hybrid = [psnr(old_ms_data[:, :, i], denoised_hybrid[:, :, i]) for i in range(227)]
        ssim_values_hybrid = []
        for i in range(227):
            ssim_val = ssim(old_ms_data[:, :, i], denoised_hybrid[:, :, i],
                            data_range=1.0, win_size=7)
            ssim_values_hybrid.append(ssim_val)
        sam_value_hybrid = sam(old_ms_data, denoised_hybrid)

        results = {
            'blur_factor': blurfactor,
            'noisy': {
                'mean_psnr': np.mean(psnr_values_old),
                'mean_ssim': np.mean(ssim_values_old),
                'sam': sam_value_old,
                'inliers': inlier_count_old
            },
            'tucker': {
                'mean_psnr': np.mean(psnr_values_tucker),
                'mean_ssim': np.mean(ssim_values_tucker),
                'sam': sam_value_tucker,
                'inliers': inlier_count_new,
                'time': elapsed_time_tucker
            },
            'wavelet': {
                'mean_psnr': np.mean(psnr_values_wavelet),
                'mean_ssim': np.mean(ssim_values_wavelet),
                'sam': sam_value_wavelet,
                'inliers': inlier_count_wavelet,
                'time': elapsed_time_wavelet
            },
            'nlm': {
                'mean_psnr': np.mean(psnr_values_nlm),
                'mean_ssim': np.mean(ssim_values_nlm),
                'sam': sam_value_nlm,
                'inliers': inlier_count_nlm,
                'time': elapsed_time_nlm
            },
            'pca': {
                'mean_psnr': np.mean(psnr_values_pca),
                'mean_ssim': np.mean(ssim_values_pca),
                'sam': sam_value_pca,
                'inliers': inlier_count_pca,
                'time': elapsed_time_pca
            },
            'hybrid': {
                'mean_psnr': np.mean(psnr_values_hybrid),
                'mean_ssim': np.mean(ssim_values_hybrid),
                'sam': sam_value_hybrid,
                'inliers': inlier_count_hybrid,
                'time': elapsed_time_hybrid
            }
        }

        point_values.append(results)

    return point_values

point_values = generate_comparison_denoise_data(point_count=2)