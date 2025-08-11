import cv2
import numpy as np
import torch, lpips
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
from colormath.color_diff import delta_e_cie2000
from colormath.color_objects import LabColor
from torchvision.models import inception_v3
import torchvision.transforms as T
from scipy.linalg import sqrtm
from skimage.color import rgb2lab, deltaE_ciede2000

# ------------------------------------------------------------------
# 1. LPIPS
lpips_fn = lpips.LPIPS(net='alex')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lpips_fn = lpips_fn.to(device).eval()

def compute_lpips(img1, img2):
    for img in (img1, img2):
        assert img.dtype == np.uint8 and img.ndim == 3, "uint8 HxWx3 필요"
    t1 = torch.from_numpy(img1).permute(2,0,1).unsqueeze(0).float().to(device) / 127.5 - 1
    t2 = torch.from_numpy(img2).permute(2,0,1).unsqueeze(0).float().to(device) / 127.5 - 1
    with torch.no_grad():
        dist = lpips_fn(t1, t2).squeeze().item()
    return dist

# ------------------------------------------------------------------
# 2. CIEDE2000 (평균)

def compute_ciede2000(img1, img2):

    # 1. BGR -> LAB 변환 후 float32로 타입 변경
    lab1 = cv2.cvtColor(img1, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab2 = cv2.cvtColor(img2, cv2.COLOR_BGR2LAB).astype(np.float32)

    # 2. OpenCV의 LAB 범위를 CIE 표준 범위로 보정
    # L 채널: 0-255 -> 0-100
    lab1[..., 0] = lab1[..., 0] * 100 / 255
    lab2[..., 0] = lab2[..., 0] * 100 / 255
    # a, b 채널: 0-255 -> -128-127
    lab1[..., 1:] -= 128
    lab2[..., 1:] -= 128

    # 3. scikit-image를 사용해 CIEDE2000 계산
    delta_e = deltaE_ciede2000(lab1, lab2)

    # 4. 전체 픽셀의 평균 델타 E 값 반환
    return np.mean(delta_e)

# ------------------------------------------------------------------
# 3. PSNR
def compute_psnr(img1, img2):
    return psnr(img1, img2, data_range=255)

# ------------------------------------------------------------------
# 4. SSIM

def ssimScore_color(img1, img2):
    ssim_val = ssim(img1, img2,channel_axis=-1,data_range=255)     # scikit-image ≥0.19
    return ssim_val

# ------------------------------------------------------------------
# 5. 히스토그램 코사인 유사도
def compute_hist_similarity(img1, img2, bins=256):
    hist_pairs = []
    for ch in range(3):
        h1 = cv2.calcHist([img1],[ch],None,[bins],[0,256]).flatten()
        h2 = cv2.calcHist([img2],[ch],None,[bins],[0,256]).flatten()
        h1, h2 = h1/np.sum(h1), h2/np.sum(h2)
        hist_pairs.append((h1,h2))
    h1 = np.concatenate([hp[0] for hp in hist_pairs])
    h2 = np.concatenate([hp[1] for hp in hist_pairs])
    return float(np.dot(h1,h2) / (np.linalg.norm(h1)*np.linalg.norm(h2) + 1e-10))





# MSE (Mean Squared Error, 평균 제곱 오차)
def mse(image1, image2):
    assert image1.shape == image2.shape, "두 이미지 크기가 다릅니다."
    err = np.mean((image1.astype(np.float32) - image2.astype(np.float32)) ** 2)
    return float(err)


def histScore(image1,image2):
    ## 히스토그램 유사도
    hsv1 = cv2.cvtColor(image1, cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(image2, cv2.COLOR_BGR2HSV)

    # 히스토그램 계산
    hist1 = cv2.calcHist([hsv1], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist2 = cv2.calcHist([hsv2], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])


    hist_similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

    return hist_similarity



if __name__ == "__main__":
    img_path1 = "original.jpg"
    img_path2 = "restored.jpg"

    img1 = cv2.imread(img_path1)          # BGR uint8
    img2 = cv2.imread(img_path2)

    metrics = {
        "LPIPS"             : compute_lpips(img1, img2),
        "CIEDE2000 (avg)"   : compute_ciede2000(img1, img2),
        "PSNR (dB)"         : compute_psnr(img1, img2),
        "SSIM"              : ssimScore_color(img1, img2),
        "Hist. CosSim"      : compute_hist_similarity(img1, img2),
    }

    # (선택) FID – 이미지가 한 장씩이면 큰 의미 없음
    # metrics["FID"] = compute_fid([img1], [img2])

    print("\n=== Metric Results ===")
    for k,v in metrics.items():
        print(f"{k:18}: {v:.4f}")