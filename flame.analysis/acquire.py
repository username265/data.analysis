from pathlib import Path
from datetime import datetime
import time

import serial
import numpy as np
import pandas as pd


# ── 사용자가 변경할 설정 ──

PORT = "/dev/cu.usbmodem1101"
BAUDRATE = 921600

DURATION_S = 15
LABEL = "test"

# ── 저장 위치 ──

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# 실행할 때마다 새로운 파일명을 만듭니다.
session_name = (
    LABEL + "_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
)

npz_path = DATA_DIR / f"{session_name}.npz"
csv_path = DATA_DIR / f"{session_name}.csv"

# ── 데이터를 담을 목록 ──

frames = []
frame_ids = []
timestamps_ms = []
sensor_temperatures = []
receive_times = []
summary_rows = []

invalid_lines = 0
missing_ids = 0
sensor_errors = 0
previous_id = None

print(f"연결할 포트: {PORT}")
print(f"수집 시간: {DURATION_S}초")
print("Arduino Serial Monitor가 닫혀 있는지 확인하세요.")


try:
    with serial.Serial(
        port=PORT,
        baudrate=BAUDRATE,
        timeout=1
    ) as ser:

        # 포트 연결 직후 보드가 재시작되는 경우를 기다립니다.
        time.sleep(2)

        # 연결 전에 쌓여 있던 데이터를 비웁니다.
        ser.reset_input_buffer()

        print("수집 시작! 센서 앞에서 손을 움직여 보세요.")

        start_time = time.monotonic()

        while time.monotonic() - start_time < DURATION_S:
            raw_line = ser.readline(16384)

            if not raw_line:
                continue

            # 끝까지 수신되지 않은 줄은 사용하지 않습니다.
            if not raw_line.endswith(b"\n"):
                invalid_lines += 1
                continue

            try:
                line = raw_line.decode("ascii").strip()

                # Arduino 오류 메시지는 화면에 표시합니다.
                if line.startswith("#ERROR"):
                    sensor_errors += 1
                    print("센서 메시지:", line)
                    continue

                # 설정 메시지 등은 건너뜁니다.
                if not line.startswith("F,"):
                    continue

                fields = line.split(",")

                if len(fields) != 772:
                    invalid_lines += 1
                    continue

                frame_id = int(fields[1])
                timestamp_ms = int(fields[2])
                sensor_ta = float(fields[3])

                frame = np.asarray(
                    fields[4:],
                    dtype=np.float32
                ).reshape(24, 32)

            except (ValueError, UnicodeDecodeError):
                invalid_lines += 1
                continue

            # 유효한 온도값만 요약 통계에 사용합니다.
            valid_values = frame[np.isfinite(frame)]

            if valid_values.size == 0:
                invalid_lines += 1
                continue

            if previous_id is not None:
                if frame_id <= previous_id:
                    print("프레임 번호가 되돌아갔습니다.")
                    print("보드 재시작 가능성이 있어 수집을 종료합니다.")
                    break

                missing_ids += frame_id - previous_id - 1

            previous_id = frame_id

            received_at = datetime.now().astimezone().isoformat(
                timespec="milliseconds"
            )

            frames.append(frame.copy())
            frame_ids.append(frame_id)
            timestamps_ms.append(timestamp_ms)
            sensor_temperatures.append(sensor_ta)
            receive_times.append(received_at)

            max_temp = float(valid_values.max())
            min_temp = float(valid_values.min())
            mean_temp = float(valid_values.mean())

            summary_rows.append({
                "condition": LABEL,
                "frame_id": frame_id,
                "esp32_ms": timestamp_ms,
                "pc_received_at": received_at,
                "sensor_ta": sensor_ta,
                "min_temp": min_temp,
                "max_temp": max_temp,
                "mean_temp": mean_temp,
                "valid_pixels": int(valid_values.size)
            })

            print(
                f"수신 {len(frames):3d}프레임 | "
                f"최고 {max_temp:.2f}°C | "
                f"평균 {mean_temp:.2f}°C | "
                f"유효 픽셀 {valid_values.size}"
            )


except KeyboardInterrupt:
    print("\n사용자가 중지했습니다. 받은 데이터는 저장합니다.")

except serial.SerialException as error:
    print("\n시리얼 연결 오류:", error)


# ── 수집한 데이터 저장 ──

if not frames:
    print("\n저장할 프레임이 없습니다.")
    print("다음을 확인하세요:")
    print("1. Arduino의 SEND_FULL_FRAME = true")
    print("2. 현재 포트 이름이 PORT와 같은지")
    print("3. Arduino Serial Monitor가 닫혀 있는지")
    print("4. ESP32가 USB로 연결되어 있는지")

else:
    np.savez_compressed(
        npz_path,
        frames=np.stack(frames),
        frame_ids=np.asarray(frame_ids, dtype=np.uint32),
        esp32_ms=np.asarray(timestamps_ms, dtype=np.uint32),
        sensor_ta=np.asarray(
            sensor_temperatures, dtype=np.float32
        ),
        pc_received_at=np.asarray(receive_times),
        condition=np.asarray(LABEL)
    )

    pd.DataFrame(summary_rows).to_csv(
        csv_path,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n저장 완료!")
    print("열데이터:", npz_path)
    print("요약 CSV:", csv_path)
    print("저장 프레임 수:", len(frames))
    print("잘못된 수신 줄:", invalid_lines)
    print("수신 프레임 번호의 누락:", missing_ids)
    print("센서 오류 메시지:", sensor_errors)

    if len(frames) >= 2:
        elapsed_s = (
            timestamps_ms[-1] - timestamps_ms[0]
        ) / 1000.0

        if elapsed_s > 0:
            actual_fps = (len(frames) - 1) / elapsed_s
            print(f"실제 수집 속도: {actual_fps:.2f} FPS")
