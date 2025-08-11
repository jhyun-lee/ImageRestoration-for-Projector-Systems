

프로젝트 소개
이 연구는 프로젝터–카메라(Projection–Camera) 환경에서 투사광 때문에 색이 왜곡된 영상을 GAN으로 원본에 가깝게 복원하고, 그 결과가 객체 검출(Template Matching, YOLO) 정확도를 실제로 끌어올리는지를 검증하는 연구 코드입니다. 시스템은 복원 단계와 디텍션 단계가 모듈화되어 있어, 복원만 거치면 기존의 사전학습된 탐지 모델을 그대로 재사용할 수 있습니다.

왜 필요한가?
프로젝터가 물체를 비출 때 조명 색·세기·주변광이 섞이며 물체의 지각적 색 분포가 심하게 바뀝니다. 과일처럼 형태는 비슷하고 색으로 구분해야 하는 객체는 디텍터가 쉽게 오인식합니다. 본 프로젝트는 디텍터를 다시 학습시키지 않고, 입력 영상을 원복해 문제를 푸는 접근입니다.

핵심 아이디어
GAN 복원기->
Color Conditioning(프로젝터 조명 RGB 벡터 주입),
Attention + Residual 블록으로 세밀한 구조/색 복원,
WGAN-GP + Perceptual Loss로 지각 품질과 안정성 확보.

판별기 보조 신호: Similarity Map 분기(실제/복원 간 픽셀 대응을 명시적으로 학습)로 미세 아티팩트 판별 강화.

객체 중심 평가: LPIPS, CIEDE2000, PSNR, SSIM, Histogram Cosine Similarity, MSE를 객체 마스크 영역 기준으로 산출.

무엇이 좋은가? (요약 성능)
정량 지표(객체 영역 기준): LPIPS 0.078, CIEDE2000 5.766, SSIM 0.903, PSNR 26.58dB, HistCosSim 0.744, MSE 386.7로 대표 베이스라인(Autoencoder, SRCNN, U-Net, ResNet50, DnCNN) 대비 전반적 우위.

디텍션 전이 효과:

Template Matching 평균 97.2%,

YOLO 평균 99.2%로 원본 이미지 성능에 근접.

범위와 한계
단색광(모노크롬) 기반 설계를 우선적으로 다룹니다. 공간적으로 불균일한 배경 조명, 고반사/고채도 재질, RGB 조명 분포 외삽 구간에서는 복원 편차가 발생할 수 있습니다.

실전 SAR/AR 적용을 위해서는 다색/동적 조명 데이터 확장과 경량화·실시간화가 권장됩니다.

이 프로젝트에 포함된것
GAN 복원 파이프라인의 핵심 구현(생성기/판별기/손실/지표 모듈)

디텍션 평가 루틴(Template Matching, YOLO)과 표/그림 재현 스크립트

재현성 강화를 위한 샘플 데이터/가중치 연동 구조(전체 공개가 어려울 경우, 샘플 + 생성 스크립트 제공 방식)






디렉토리 

├─ 1_GanModel/
│  ├─ GanModel.py
│  ├─ GanModel_exe.py
│  ├─ GanModel_outPut.py
│  └─ GanModel_PT/
      ├─ discriminator_epoch_50.h5
      └─ generator_epoch_50.h5
├─ 2_ModelSet
│  ├─ ModelSet.py
│  ├─ Model_exe.py
│  └─ Model_OutPut.py
├─ 3_Value
│  ├─ CropImage_Evaluation_EXE.py
│  ├─ Image_Evaluation_EXE.py
│  ├─ Image_Evaluation_Funtion.py
│  └─ Template/
      ├─ FruitImage_Real/
      └─ TestDetecting_Rotation_evaluate _2.py
│  └─ Yolo/
      ├─ 08066best.pt
      ├─ TestDetecting_Yolo_evaluate_1.py
      └─ YoloDetection.py

├─ ImageData/
│  ├─ Original_100/
│  ├─ Yolo_Label_100/
│  └─ SampleI_mage/
├─ FeatureDataCreate.py


 