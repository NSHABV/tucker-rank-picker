import cv2
import matplotlib.pyplot as plt
from glob import glob
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import cross_val_score, KFold
import random
import tifffile

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

def load_pepper_data():
    rgb = cv2.imread('data/TestPatch/thumbnail.jpg', cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    ms_files = tifffile.imread('data/TestPatch/spectre.TIF')

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


def psnr_ignore_black_rgb(original, noisy, max_pixel=1.0):
    mask = np.any(original != 0, axis=-1)

    if np.sum(mask) == 0:
        return float('nan')

    mse = np.mean((original[mask] - noisy[mask]) ** 2)

    if mse == 0:
        return float('inf')

    psnr = 10 * np.log10((max_pixel ** 2) / mse)
    return psnr

def rand_angle():
    random_positive = random.randint(30, 150)

    random_negative = random.randint(-150, -30)

    if random.choice([True, False]):
        random_number = random_positive
    else:
        random_number = random_negative

    print(random_number)

def generate_model_orb_psnr_data(rot_angle = 50, point_count = 600):
    ms_data, rgb_data = load_pepper_data()

    array = np.linspace(0, 8, num=point_count)

    psnr_values = []
    point_values = []

    for blurfactor in array:
        noisy_rgb = rgb_data
        noisy_rgb = add_multispectral_noise(rgb_data, gauss_std=0.03 * blurfactor, poisson_scale=True, salt_pepper_prob=0.01 * blurfactor, striping_intensity=0.05 * blurfactor)

        rgb_dest = noisy_rgb

        gray_origin = cv2.cvtColor(rgb_data, cv2.COLOR_BGR2GRAY)
        gray_dest = cv2.cvtColor(rgb_dest, cv2.COLOR_BGR2GRAY)

        gray_origin = cv2.normalize(gray_origin, None, 0, 255, cv2.NORM_MINMAX)
        gray_dest = cv2.normalize(gray_dest, None, 0, 255, cv2.NORM_MINMAX)

        gray_origin = gray_origin.astype(np.uint8)
        gray_dest = gray_dest.astype(np.uint8)

        H, inlier_count = find_homography(gray_dest, gray_origin)
        if inlier_count == 0:
            break

        aligned_img = rgb_dest

        psnr = psnr_ignore_black_rgb(aligned_img, rgb_data, max_pixel=1.0)

        psnr_values.append(psnr)
        point_values.append(inlier_count)

    return psnr_values, point_values

psnr_values, point_values = generate_model_orb_psnr_data(0, 250)

data = np.column_stack((psnr_values, point_values))
psnr = data[:, 0]
orb_matches = data[:, 1]

orb_matches = np.clip(orb_matches, 0, 200)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(orb_matches.reshape(-1, 1))

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_scaled)
model = Ridge(alpha=1.0)
model.fit(X_poly, psnr)

model.fit(X_poly, psnr)
y_pred = model.predict(X_poly)

cv = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_poly, psnr, cv=cv, scoring='r2')
print(f"Cross-validated R²: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")

def predict_psnr(orb_matches, model, poly, scaler):
    orb_scaled = scaler.transform([[orb_matches]])
    orb_poly = poly.transform(orb_scaled)
    return model.predict(orb_poly)[0]

psnr_values_test, point_values_test = generate_model_orb_psnr_data('rand', 100)
test_data = np.column_stack((psnr_values_test, point_values_test))
test_psnr = test_data[:, 0]
test_orb = test_data[:, 1]

test_orb_scaled = scaler.transform(test_orb.reshape(-1, 1))

test_orb_poly = poly.transform(test_orb_scaled)

test_psnr_pred = model.predict(test_orb_poly)

plt.figure(figsize=(10, 6))

plt.scatter(orb_matches, psnr, s=30, alpha=0.5,
            label="Training Data", color="blue")

plt.scatter(test_orb, test_psnr, s=30, alpha=0.7,
            label="Test Data (Ground Truth)", color="red")

plt.scatter(test_orb, test_psnr_pred, s=30, alpha=0.7,
            label="Test Data (Predicted)", color="green")

orb_range = np.linspace(0, 200, 100).reshape(-1, 1)
orb_range_scaled = scaler.transform(orb_range)
orb_range_poly = poly.transform(orb_range_scaled)
psnr_range_pred = model.predict(orb_range_poly)
plt.plot(orb_range, psnr_range_pred, color="black",
         linewidth=2, label="Model Curve")

plt.xlabel("ORB Matches")
plt.ylabel("PSNR (dB)")
plt.title("PSNR vs. ORB Matches (Training + Test Data)")
plt.legend()
plt.grid(True)
plt.show()