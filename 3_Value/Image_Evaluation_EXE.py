import os, re, cv2, numpy as np, pandas as pd
from tqdm import tqdm
from typing import Optional

# ──────────────────────────────────────────────────────────
# 1.  지표 함수 (이미 만들어 두신 모듈로부터 import)
from Image_Evaluation_Funtion import (
    compute_lpips, compute_ciede2000, compute_psnr, ssimScore_color,
    compute_hist_similarity, histScore, mse
)
# ──────────────────────────────────────────────────────────

# ---------- 유틸 ----------
EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')

def imread_unicode(path):                 # 한글 경로 대응
    data = np.fromfile(path, np.uint8)
    return None if data.size == 0 else cv2.imdecode(data, cv2.IMREAD_COLOR)

def extract_key(fname: str) -> Optional[str]:
    """파일명에서 연속 10자리 숫자(예: 0124120252) 추출"""
    m = re.search(r'\d{10}', fname)
    return m.group(0) if m else None

# ---------- 매칭 ----------
def build_ref_map(ref_dir):
    """key → ref_filename  (중복 키가 있으면 첫 파일 사용)"""
    ref_map = {}
    for f in os.listdir(ref_dir):
        if f.lower().endswith(EXTS):
            k = extract_key(f)
            if k and k not in ref_map:
                ref_map[k] = f
    return ref_map

def paired_files(ref_map, cand_dir):
    """
    cand_dir 기준으로: 같은 키가 ref_map 에 있으면 1:1 쌍 생성
    반환 [(ref_fname, cand_fname, key), ...]
    """
    pairs = []
    for f in os.listdir(cand_dir):
        if f.lower().endswith(EXTS):
            k = extract_key(f)
            if k and k in ref_map:
                pairs.append((ref_map[k], f, k))
    return pairs


def downsample(img, max_side=512):
    """긴 변이 max_side 이상이면 비율 축소, 아니면 그대로 반환"""
    h, w = img.shape[:2]
    if max(h, w) <= max_side:
        return img
    scale = max_side / max(h, w)
    new_size = (int(w*scale), int(h*scale))
    return cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)


# ---------- 디렉터리 평가 ----------
def process_dir(ref_dir, cand_dir, ref_map):
    rows = []
    for ref_f, cand_f, key in tqdm(paired_files(ref_map, cand_dir),
                                   desc=os.path.basename(cand_dir)):
        img1 = imread_unicode(os.path.join(ref_dir,  ref_f))
        img2 = imread_unicode(os.path.join(cand_dir, cand_f))

        if img1 is None or img2 is None:
            print(f"[WARN] 이미지 로드 실패 → {ref_f}, {cand_f}")
            continue

        rows.append({
            "key"        : key,
            "ref_file"   : ref_f,
            "cand_file"  : cand_f,
            "LPIPS"      : compute_lpips(img1, img2),
            "CIEDE2000"  : compute_ciede2000(img1, img2),
            "PSNR(dB)"   : compute_psnr(img1, img2),
            "SSIM"       : ssimScore_color(img1, img2),
            "HistCosSim" : compute_hist_similarity(img1, img2),
            "HistSim"    : histScore(img1, img2),
            "mse"        : mse(img1, img2),
        })
    return pd.DataFrame(rows).set_index("key")


# ---------- 실행 & 엑셀 저장 ----------
def run(ref_dir, cand_dirs, out_path="result_metrics.xlsx"):
    ref_map = build_ref_map(ref_dir)
    writer  = pd.ExcelWriter(out_path, engine='openpyxl')
    summary = []

    for cdir in cand_dirs:
        df = process_dir(ref_dir, cdir, ref_map)
        sheet = os.path.basename(cdir)[:31] or "model"
        df.to_excel(writer, sheet_name=sheet)

        means = df.select_dtypes('number').mean().to_dict()
        means["model"] = sheet
        summary.append(means)

    pd.DataFrame(summary).set_index("model").to_excel(writer, sheet_name="_summary")
    writer.close()
    print(f"✅ 결과 저장 완료 → {out_path}")



# ──────────────────────────────────────────────────────────
# 2.  경로 설정 & 실행
if __name__ == "__main__":
    root = os.path.dirname(os.path.realpath(__file__))
    data_base_dir = os.path.join(os.path.dirname(root), 'ImageData')

    REF_DIR = os.path.join(data_base_dir, 'TestSet_New_3500_Ori')
    CAND_DIRS = [
        os.path.join(data_base_dir, "GeneratorOutputs","NoGenerator", "autoencoder", "1"),
        os.path.join(data_base_dir, "GeneratorOutputs","NoGenerator", "dncnn",       "1"),
        os.path.join(data_base_dir, "GeneratorOutputs","NoGenerator", "resnet",      "1"),
        os.path.join(data_base_dir, "GeneratorOutputs","NoGenerator", "resnet50",    "1"),
        os.path.join(data_base_dir, "GeneratorOutputs","NoGenerator", "srcnn",       "1"),
        os.path.join(data_base_dir, "GeneratorOutputs","NoGenerator", "unet",        "1"),
        os.path.join(data_base_dir, "TestSet_New_3500_Ori"),
    ]

    OUTPUT_XLSX = "result_metrics.xlsx"
    run(REF_DIR, CAND_DIRS, OUTPUT_XLSX)
