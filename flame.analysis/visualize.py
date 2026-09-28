from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
FIGURE_DIR = PROJECT_DIR / "figures"
FIGURE_DIR.mkdir(exist_ok=True)

files = list(DATA_DIR.glob("*.npz"))

if not files:
    raise SystemExit(
        "data 폴더에 NPZ 파일이 없습니다. "
        "먼저 acquire.py를 실행하세요."
    )

# 가장 최근에 저장된 파일을 선택합니다.
input_path = max(files, key=lambda path: path.stat().st_mtime)

with np.load(input_path) as data:
    frames = data["frames"]
    timestamps_ms = data["esp32_ms"]
    frame_ids = data["frame_ids"]

if frames.ndim != 3 or frames.shape[1:] != (24, 32):
    raise SystemExit("열데이터 배열의 모양이 올바르지 않습니다.")

# 각 프레임의 최고·평균 온도를 계산합니다.
# NaN 및 무한대는 계산에서 제외합니다.
clean_frames = np.where(np.isfinite(frames), frames, np.nan)

max_temps = np.nanmax(clean_frames, axis=(1, 2))
mean_temps = np.nanmean(clean_frames, axis=(1, 2))

# 첫 프레임을 0초로 하는 시간축입니다.
time_s = (
    timestamps_ms.astype(np.float64)
    - float(timestamps_ms[0])
) / 1000.0

# 최고 온도가 가장 높았던 프레임을 선택합니다.
selected_index = int(np.nanargmax(max_temps))
selected_frame = clean_frames[selected_index]

print("읽은 파일:", input_path.name)
print("데이터 모양:", frames.shape)
print("선택한 프레임 ID:", frame_ids[selected_index])

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# 왼쪽: 열화상
image = axes[0].imshow(
    selected_frame,
    cmap="inferno",
    interpolation="nearest",
    origin="upper"
)

axes[0].set_title(
    f"Thermal Map - Frame {frame_ids[selected_index]}"
)
axes[0].set_xlabel("Pixel X")
axes[0].set_ylabel("Pixel Y")

fig.colorbar(
    image,
    ax=axes[0],
    label="Apparent Temperature (°C)"
)

# 오른쪽: 온도 변화 그래프
axes[1].plot(
    time_s,
    max_temps,
    label="Maximum temperature"
)

axes[1].plot(
    time_s,
    mean_temps,
    label="Mean temperature"
)

axes[1].set_title("Temperature Over Time")
axes[1].set_xlabel("Time (s)")
axes[1].set_ylabel("Apparent Temperature (°C)")
axes[1].grid(True, alpha=0.3)
axes[1].legend()

fig.tight_layout()

output_path = FIGURE_DIR / f"{input_path.stem}_preview.png"
fig.savefig(output_path, dpi=200)

print("그림 저장:", output_path)

plt.show()
