<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>GAN-based Image Restoration for Projector–Camera Systems</title>
  <style>
    :root {
      --fg:#111; --muted:#555; --bg:#fff; --card:#fafafa; --line:#e6e6e6; --accent:#2563eb;
    }
    html,body{margin:0;padding:0;background:var(--bg);color:var(--fg);font:16px/1.6 system-ui,-apple-system,Segoe UI,Roboto,Apple SD Gothic Neo,Noto Sans KR,Helvetica,Arial,sans-serif}
    main{max-width:960px;margin:48px auto;padding:0 20px}
    h1{font-size:2rem;margin:0 0 12px}
    h2{font-size:1.35rem;margin:28px 0 8px;border-bottom:1px solid var(--line);padding-bottom:6px}
    h3{font-size:1.05rem;margin:18px 0 6px;color:var(--muted)}
    p{margin:8px 0}
    ul{margin:8px 0 8px 20px}
    code,pre{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
    pre{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px;overflow:auto}
    .card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px;margin:12px 0}
    .grid{display:grid;gap:12px}
    @media(min-width:700px){.grid.cols-2{grid-template-columns:1fr 1fr}}
    table{width:100%;border-collapse:collapse;background:var(--bg);border:1px solid var(--line)}
    th,td{border-bottom:1px solid var(--line);padding:10px 8px;text-align:left}
    th{background:#f6f8fa;font-weight:600}
    .badge{display:inline-block;background:#eef2ff;color:#3730a3;border:1px solid #c7d2fe;border-radius:999px;padding:2px 8px;font-size:.8rem;margin-right:6px}
    .muted{color:var(--muted)}
    details{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px 12px}
    details summary{cursor:pointer;font-weight:600}
    a{color:var(--accent);text-decoration:none}
    a:hover{text-decoration:underline}
  </style>
</head>
<body>
<main>
  <h1>GAN-based Image Restoration for Projector–Camera Systems</h1>
  <p class="muted">프로젝터–카메라 환경의 색 왜곡을 복원하여 객체 검출 성능을 향상시키는 연구 코드</p>

  <div class="card">
    <span class="badge">GAN Restoration</span>
    <span class="badge">WGAN-GP + Perceptual</span>
    <span class="badge">Color Conditioning</span>
    <span class="badge">Attention + Residual</span>
    <span class="badge">Template Matching / YOLO</span>
  </div>

  <h2>프로젝트 소개</h2>
  <p>
    이 레포는 프로젝터–카메라(Projection–Camera) 환경에서 <b>투사광으로 왜곡된 영상</b>을 <b>GAN</b>으로 원본에
    가깝게 복원하고, 그 결과가 <b>객체 검출(Template Matching, YOLO)</b> 정확도를 실제로 향상시키는지 검증합니다.
    복원 단계와 디텍션 단계가 <b>모듈화</b>되어 있어, 복원만 거치면 기존의 사전학습 디텍터를
    <b>재학습 없이</b> 재사용할 수 있습니다.
  </p>

  <h2>왜 필요한가?</h2>
  <p>
    프로젝터 조명(색/세기/주변광)으로 <b>지각적 색 분포가 크게 변형</b>됩니다. 형태가 유사하고 <b>색으로 구분</b>해야 하는
    객체(예: 과일)에서 오인식이 늘어납니다. 본 프로젝트는 디텍터를 다시 학습하지 않고, 입력 영상을 <b>복원(원복)</b>하여 문제를 해결합니다.
  </p>

  <h2>핵심 아이디어</h2>
  <ul>
    <li><b>GAN 복원기</b>
      <ul>
        <li><b>Color Conditioning</b>: 프로젝터 조명 <code>RGB</code> 벡터 주입</li>
        <li><b>Attention + Residual</b> 블록: 세밀한 구조/색 보존</li>
        <li><b>WGAN-GP + Perceptual Loss</b>: 지각 품질 및 학습 안정성 확보</li>
      </ul>
    </li>
    <li><b>판별기 보조 신호</b>: <b>Similarity Map</b> 분기(실제/복원 간 픽셀 대응 명시적 학습)로 미세 아티팩트 판별 강화</li>
    <li><b>객체 중심 평가</b>: LPIPS, CIEDE2000, PSNR, SSIM, Histogram Cosine Similarity, MSE를 <b>객체 마스크 영역 기준</b>으로 산출</li>
  </ul>

  <h2>무엇이 좋은가? (요약 성능)</h2>
  <div class="grid cols-2">
    <div>
      <h3>정량 지표 (객체 영역 기준)</h3>
      <table>
        <thead>
          <tr>
            <th>지표</th><th>값</th>
          </tr>
        </thead>
        <tbody>
          <tr><td>LPIPS ↓</td><td>0.078</td></tr>
          <tr><td>CIEDE2000 ↓</td><td>5.766</td></tr>
          <tr><td>SSIM ↑</td><td>0.903</td></tr>
          <tr><td>PSNR ↑</td><td>26.58 dB</td></tr>
          <tr><td>HistCosSim ↑</td><td>0.744</td></tr>
          <tr><td>MSE ↓</td><td>386.7</td></tr>
        </tbody>
      </table>
      <p class="muted">Autoencoder, SRCNN, U-Net, ResNet50, DnCNN 대비 전반적 우위</p>
    </div>
    <div>
      <h3>디텍션 전이 효과</h3>
      <table>
        <thead>
          <tr><th>방법</th><th>평균 정확도</th></tr>
        </thead>
        <tbody>
          <tr><td>Template Matching</td><td>97.2%</td></tr>
          <tr><td>YOLO</td><td>99.2%</td></tr>
        </tbody>
      </table>
      <p class="muted">원본 이미지 성능에 근접</p>
    </div>
  </div>

  <h2>범위와 한계</h2>
  <ul>
    <li><b>단색광(모노크롬)</b> 기반 설계 우선</li>
    <li><b>공간적으로 불균일한 배경 조명</b>, <b>고반사/고채도 재질</b>, <b>RGB 조명 분포 외삽</b> 구간에서 복원 편차 가능</li>
    <li>실전 SAR/AR 적용을 위해 <b>다색/동적 조명 데이터</b> 확장 및 <b>경량화·실시간화</b> 권장</li>
  </ul>

  <h2>이 프로젝트에 포함된 것</h2>
  <ul>
    <li><b>GAN 복원 파이프라인 핵심 구현</b> (생성기/판별기/손실/지표 모듈)</li>
    <li><b>디텍션 평가 루틴</b> (Template Matching, YOLO) 및 <b>표/그림 재현 스크립트</b></li>
    <li><b>재현성 강화 구조</b>:
      전체 데이터 공개가 어려울 경우 <b>샘플 + 생성 스크립트</b> 방식으로 연동</li>
  </ul>

  <h2>디렉토리 구조</h2>
  <details open>
    <summary><b>폴더 트리 보기</b></summary>
    <pre><code>├─ 1_GanModel/
│  ├─ GanModel.py                  # GAN 모델 정의(Generator/Discriminator, 손실 구성 포함)
│  ├─ GanModel_exe.py              # GAN 학습/복원 실행 로직(실험 스크립트 엔트리)
│  ├─ GanModel_outPut.py           # 복원 결과 저장/시각화 유틸
│  └─ GanModel_PT/                 # 사전학습/최종 가중치(.h5)
│      ├─ discriminator_epoch_50.h5
│      └─ generator_epoch_50.h5
├─ 2_ModelSet/
│  ├─ ModelSet.py                  # 비교 복원 베이스라인(AE/SRCNN/U-Net/ResNet/DnCNN) 모듈
│  ├─ Model_exe.py                 # 베이스라인 실행/추론 파이프라인
│  └─ Model_OutPut.py              # 베이스라인 출력 정리/저장
├─ 3_Value/
│  ├─ CropImage_Evaluation_EXE.py  # 객체 마스크/크롭 기준 정량 평가 실행
│  ├─ Image_Evaluation_EXE.py      # 전체 이미지 기준 정량 평가 실행
│  ├─ Image_Evaluation_Funtion.py  # 평가 지표(PSNR/SSIM/LPIPS/CIEDE2000/HistCosSim/MSE)
│  └─ Template/
│      ├─ FruitImage_Real/         # 템플릿 매칭용 클래스별 템플릿(실제 이미지 크롭)
│      └─ TestDetecting_Rotation_evaluate_2.py  # 템플릿 매칭 평가(회전 등 조건 실험)
│  └─ Yolo/
│      ├─ 08066best.pt             # YOLO 가중치(프로젝트 베스트)
│      ├─ TestDetecting_Yolo_evaluate_1.py      # YOLO 평가 스크립트
│      └─ YoloDetection.py         # YOLO 추론/후처리 유틸
├─ ImageData/
│  ├─ Original_100/                # 원본 이미지(샘플)
│  ├─ Yolo_Label_100/              # YOLO 라벨(박스/클래스)
│  └─ SampleI_mage/                # 데모/문서용 샘플 이미지(※ 이름 정리 권장: SampleImage)
├─ FeatureDataCreate.py            # RGB 조건 벡터 등 부가 피처 생성
</code></pre>
  </details>

  <h3>디렉토리 역할 요약</h3>
  <ul>
    <li><b>1_GanModel/</b>: GAN 복원기 학습/추론 핵심 코드 + 가중치</li>
    <li><b>2_ModelSet/</b>: 베이스라인 복원 모델 실행 모듈</li>
    <li><b>3_Value/</b>: 정량 지표 및 디텍션 성능 평가(Template/YOLO)</li>
    <li><b>ImageData/</b>: 원본/라벨/샘플 데이터(일부만 포함)</li>
    <li><b>FeatureDataCreate.py</b>: RGB <em>조건 벡터 생성</em> 스크립트</li>
  </ul>

  <hr />
  <p class="muted">
    참고: GitHub에서는 일반적으로 <code>README.md</code>(마크다운)를 사용합니다.
    필요 시 마크다운 문서 안에 본문의 HTML 섹션을 그대로 붙여도 렌더링됩니다.
  </p>
</main>
</body>
</html>
