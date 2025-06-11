from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from load_object_model import load_object_model   # 함수가 있는 모듈
import argparse
import numpy as np
# ------------------------------------------------------------
parser = argparse.ArgumentParser()

parser.add_argument("-on","--object_name",dest="object_name",action="store")

args = parser.parse_args()
if args.object_name != None:
    object_name = args.object_name

# 1) 사전 학습 모델 불러오기
DMC_PRETRAIN_DIR = Path("")   # 필요하면 환경 변수 대신 직접 지정
model = load_object_model(
    model_name="model.pt",   # 서브폴더 이름 = 실험명
    object_name=object_name,         # 그래프에 저장된 key (예: "digit_6")
    #0_0, 1_5, 2_9, 3_3, 4_1 , 5_7, 6_6, 7_2, 8_8, and 9_4
    features=("rgba","pose_vectors_flat"),            # 색상까지 함께 읽기
    checkpoint=None,               # 마지막 model.pt
    lm_id=0,
)

# 2) 필요하면 위치·회전 보정
# model = model - [0, 1.5, 0]      # 평행 이동
# model = model.rotated([0, 0, 90], degrees=True)

# ───────── 색상 벡터 만들기 ─────────
# 밝기(0‥1) → 컬러맵(예: inferno)으로 변환
brightness = model.rgba[:, 0]        # R=G=B → 회색값
colors = plt.cm.viridis(brightness)  # inferno / viridis / plasma 등

# 3) 3-D 플롯
fig = plt.figure(figsize=(6, 6))
ax  = fig.add_subplot(111, projection="3d")
ax.scatter(model.x, model.y, model.z,
           c=colors, s=8, linewidth=0)
# 색상 정보가 있으면 사용, 없으면 단색
# colors = getattr(model, "rgba", None)
# if colors is not None:
#     ax.scatter(model.x, model.y, model.z, c=colors[:, :3])
# else:
#     ax.scatter(model.x, model.y, model.z, s=5, color="royalblue")

ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
ax.set_box_aspect([1, 0.1, 1])       # 큐브 비율
plt.show()
