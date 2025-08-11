import os, re, cv2, numpy as np, pandas as pd
from tqdm import tqdm
from typing import Optional,Tuple

from datetime import datetime

# ──────────────────────────────────────────────────────────
# 1.  지표 함수 (이미 만들어 두신 모듈로부터 import)
from Image_Evaluation_Funtion import (
    compute_lpips, compute_ciede2000, compute_psnr, ssimScore_color,
    compute_hist_similarity, histScore, mse
)
# ──────────────────────────────────────────────────────────

# ---------- 유틸 ----------
EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')



def parse_fname(fname: str) -> Optional[Tuple[str, str, str]]:
    m_proj = re.match(r'([A-Za-z]+)-projected(\d{10})', fname)
    if m_proj:
        fruit, id10 = m_proj.groups()
        return fruit, id10[-6:], id10

    m_ori = re.match(r'([A-Za-z]+)-Ori(\d{10})', fname)
    if m_ori:
        fruit, id10 = m_ori.groups()
        return fruit, id10[-6:], id10
    return None


def imread_unicode(path):                 # 한글 경로 대응
    data = np.fromfile(path, np.uint8)
    return None if data.size == 0 else cv2.imdecode(data, cv2.IMREAD_COLOR)

def extract_key(fname: str) -> Optional[str]:
    """파일명에서 연속 10자리 숫자(예: 0124120252) 추출"""
    m = re.search(r'\d{10}', fname)
    return m.group(0) if m else None

# ---------- 매칭 ----------
def build_ref_map(ref_dir):
    """key = (id6, fruit)  → ref_filename"""
    ref_map = {}
    for f in os.listdir(ref_dir):
        info = parse_fname(f)
        if info and "-Ori" in f:
            fruit, id6, _ = info
            key = (id6, fruit)
            if key not in ref_map:       # 중복 방지
                ref_map[key] = f
    return ref_map


def paired_files(ref_map, cand_dir):
    """
    cand_dir 기준으로 (id6, fruit) 키가 ref_map 에 있으면 쌍 생성
    """
    pairs = []
    for f in os.listdir(cand_dir):
        info = parse_fname(f)
        if info and "-projected" in f:
            fruit, id6, _ = info
            key = (id6, fruit)
            ref_fname = ref_map.get(key)
            if ref_fname:
                pairs.append((ref_fname, f, f"{fruit}-{id6}"))
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


def write_additional_summaries(writer: pd.ExcelWriter, df: pd.DataFrame, sheet_base: str):
    overall_stat = generate_statistical_summary(df)
    overall_stat.to_excel(writer, sheet_name=f"{sheet_base}_stats")

    # 과일명 추출
    df["fruit"] = df["ref_file"].apply(lambda x: x.split('-')[0] if '-' in x else "Unknown")
    fruitwise_stat = generate_statistical_summary(df, groupby_col="fruit")
    fruitwise_stat.to_excel(writer, sheet_name=f"{sheet_base}_fruit_stats")


def generate_statistical_summary(df: pd.DataFrame, groupby_col: str = None) -> pd.DataFrame:
    """
    수치형 컬럼에 대해 평균, 최소, 최대를 계산
    - groupby_col이 주어지면 해당 컬럼 기준 그룹 통계 반환
    - 그렇지 않으면 전체 통계 반환
    """
    numeric_cols = df.select_dtypes(include='number')

    if groupby_col:
        grouped = df.groupby(groupby_col)
        summary_frames = []

        for name, group in grouped:
            mean = group[numeric_cols.columns].mean().rename("mean")
            min_ = group[numeric_cols.columns].min().rename("min")
            max_ = group[numeric_cols.columns].max().rename("max")
            stat_df = pd.concat([mean, min_, max_], axis=1)
            stat_df["group"] = name
            summary_frames.append(stat_df.reset_index())

        result_df = pd.concat(summary_frames, ignore_index=True)
        return result_df.set_index(["group", "index"])

    else:
        mean = numeric_cols.mean().rename("mean")
        min_ = numeric_cols.min().rename("min")
        max_ = numeric_cols.max().rename("max")
        return pd.concat([mean, min_, max_], axis=1)
    

# ---------- 실행 & 엑셀 저장 ----------
def run(ref_dir, cand_dirs, out_path="result_metrics.xlsx"):
    ref_map = build_ref_map(ref_dir)
    writer  = pd.ExcelWriter(out_path, engine='openpyxl')
    summary = []


    cand_dirs = cand_dirs[:1000]

    for cdir in cand_dirs:
        df = process_dir(ref_dir, cdir, ref_map)
        sheet = os.path.basename(cdir)[:31] or "model"
        df.to_excel(writer, sheet_name=sheet)

        write_additional_summaries(writer, df, sheet)

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
    data_base_dir = os.path.join(os.path.dirname(root), r"ImageData\GeneratorOutputs\Fin_0731_Crop")

    current_time = datetime.now()
    formatted_time = current_time.strftime("%m%d")


    REF_DIR = os.path.join(data_base_dir, "Ori")

    CAND_DIRS = []


            
    CAND_DIRS = [

        # os.path.join(data_base_dir, "Pro"),
        # os.path.join(data_base_dir, "autoencoder"),
        # os.path.join(data_base_dir, "dncnn"),
        # os.path.join(data_base_dir, "resnet50"),
        # os.path.join(data_base_dir, "srcnn"),
        # os.path.join(data_base_dir, "unet"),
        
        # os.path.join(data_base_dir, "bestmodel_Com3"),
        os.path.join(data_base_dir, "bestmodel_Com5_Wgan_All_v2_5"),
        os.path.join(data_base_dir, "bestmodel_Com5_Wgan_All_v2_30"),
        os.path.join(data_base_dir, "bestmodel_Com5_Wgan_All_v2_50")
        
    ]

    OUTPUT_XLSX = f"{formatted_time}_Gan_CropResult_Fin_3.xlsx"
    run(REF_DIR, CAND_DIRS, OUTPUT_XLSX)
