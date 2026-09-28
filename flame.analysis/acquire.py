"""기존 ESP32 F,id,ms,ta,pixel_0,...,pixel_767 형식의 실험 수집기."""

import argparse
from datetime import datetime
import json
from pathlib import Path
from threading import Event, Thread
import time

import numpy as np
import pandas as pd
import serial


def parse_frame(raw):
    if not raw.endswith(b"\n"):
        raise ValueError("incomplete line")
    line = raw.decode("ascii").strip()
    if not line.startswith("F,"):
        return None, line
    fields = line.split(",")
    if len(fields) != 772:
        raise ValueError("expected 772 fields")
    frame_id, timestamp = int(fields[1]), int(fields[2])
    ta = float(fields[3])
    frame = np.asarray(fields[4:], dtype=np.float32).reshape(24, 32)
    if frame_id < 0 or not 0 <= timestamp <= 0xFFFFFFFF:
        raise ValueError("invalid counter")
    return (frame_id, timestamp, ta, frame), line


def wait_for_enter(ser):
    """사용자가 준비하는 동안에도 USB 데이터를 읽어 과거 프레임의 적체를 줄입니다."""
    ready = Event()
    errors = []

    def confirm():
        try:
            input("준비 후 Enter: ")
        except EOFError as error:
            errors.append(error)
        finally:
            ready.set()

    Thread(target=confirm, daemon=True).start()
    ser.timeout = 0.1
    while not ready.is_set():
        ser.readline(16384)
    ser.timeout = 1
    if errors:
        raise errors[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="/dev/cu.usbmodem1101")
    parser.add_argument("--condition", choices=["test", "background", "flame", "static"], default="test")
    parser.add_argument("--distance", type=int, default=20, help="센서 앞면에서 열원까지의 거리(cm)")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--seconds", type=float, help="기본값: 배경 10초, 불꽃/정적 열원 3초, 시험 15초")
    parser.add_argument("--refresh", default="MLX90640_4_HZ", help="Arduino에 실제 설정한 값; 이 옵션이 센서 설정을 바꾸지는 않습니다.")
    parser.add_argument("--note", default="", help="관측 조건 또는 특이사항")
    args = parser.parse_args()
    if args.seconds is None:
        args.seconds = {"background": 10.0, "flame": 3.0, "static": 3.0, "test": 15.0}[args.condition]
    if args.seconds <= 0 or args.distance <= 0 or args.repeat <= 0:
        parser.error("시간, 거리, 반복 번호는 양수여야 합니다.")

    root = Path(__file__).resolve().parent
    data_dir = root / "data"
    data_dir.mkdir(exist_ok=True)
    label = f"{args.condition}_d{args.distance}_r{args.repeat:02d}"
    session = label + "_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    records, frames = [], []
    quality = {"invalid_lines": 0, "missing_ids": 0, "sensor_errors": 0}
    reason, previous_id, previous_ms = "not_started", None, None

    print(f"조건={args.condition}, 거리={args.distance}cm, 반복={args.repeat}, 시간={args.seconds:g}초")
    print(f"포트={args.port}, 기록할 센서 설정={args.refresh}")
    print("Arduino Serial Monitor를 닫아주세요.")
    try:
        with serial.Serial(args.port, 921600, timeout=1) as ser:
            time.sleep(2)
            ser.reset_input_buffer()
            # 실제 열화상 스트림을 먼저 확인합니다. 이때는 점화하지 않습니다.
            deadline = time.monotonic() + 8
            connected = False
            while time.monotonic() < deadline:
                raw = ser.readline(16384)
                if not raw:
                    continue
                try:
                    parsed, message = parse_frame(raw)
                except (ValueError, UnicodeDecodeError):
                    continue
                if message.startswith("#ERROR"):
                    print("센서:", message)
                if parsed is not None:
                    connected = True
                    break
            if not connected:
                raise serial.SerialException("F, 프레임을 받지 못했습니다. 포트와 SEND_FULL_FRAME=true를 확인하세요.")

            print("\n센서 연결 확인 완료. 아직 측정 파일에 저장하지 않습니다.")
            if args.condition == "flame":
                print("불꽃을 정해진 위치에 점화하고, 위치가 안정되면 Enter를 누르세요.")
                print("측정이 끝났다는 문구가 나오면 즉시 소화하세요.")
            elif args.condition == "background":
                print("같은 위치에 식은 라이터를 두고 불꽃이 없는 상태에서 Enter를 누르세요.")
            else:
                print("측정할 물체와 센서의 위치를 고정한 뒤 Enter를 누르세요.")
            wait_for_enter(ser)
            # Enter 전에 USB에 쌓인 프레임은 저장하지 않습니다.
            ser.reset_input_buffer()
            # 버퍼를 비우면서 잘린 첫 줄 하나를 버립니다. 다음 줄부터 프레임 경계를 맞춥니다.
            ser.readline(16384)
            print(f"측정 시작 — {args.seconds:g}초 동안 조건을 유지하세요.", flush=True)
            start = time.monotonic()
            reason = "completed"
            while True:
                remaining = args.seconds - (time.monotonic() - start)
                if remaining <= 0:
                    break
                ser.timeout = min(1.0, remaining)
                raw = ser.readline(16384)
                if not raw:
                    continue
                try:
                    parsed, message = parse_frame(raw)
                except (ValueError, UnicodeDecodeError):
                    quality["invalid_lines"] += 1
                    continue
                if message.startswith("#ERROR"):
                    quality["sensor_errors"] += 1
                    print("센서:", message)
                if parsed is None:
                    continue
                frame_id, timestamp, ta, frame = parsed
                if previous_id is not None:
                    if frame_id <= previous_id or timestamp <= previous_ms:
                        reason = "board_reset_or_counter_change"
                        print("프레임 번호 또는 시각이 되돌아갔습니다. 이번 측정을 중단합니다.")
                        break
                    quality["missing_ids"] += frame_id - previous_id - 1
                previous_id, previous_ms = frame_id, timestamp
                valid = frame[np.isfinite(frame)]
                frames.append(frame.copy())
                records.append({
                    "condition": args.condition, "distance_cm": args.distance, "repeat": args.repeat,
                    "frame_id": frame_id, "esp32_ms": timestamp,
                    "pc_received_at": datetime.now().astimezone().isoformat(timespec="milliseconds"),
                    "sensor_ta": ta, "min_temp": float(valid.min()) if valid.size else np.nan,
                    "max_temp": float(valid.max()) if valid.size else np.nan,
                    "mean_temp": float(valid.mean()) if valid.size else np.nan,
                    "valid_pixels": int(valid.size),
                })
            print("\n측정 종료! 불꽃 측정이면 지금 소화하세요.", flush=True)
    except KeyboardInterrupt:
        reason = "interrupted"
        print("\n중지했습니다. 불꽃이 있으면 소화하세요. 받은 데이터는 보관합니다.", flush=True)
    except (serial.SerialException, EOFError, OSError) as error:
        reason = "connection_error"
        print(f"\n수집 오류: {error}\n불꽃이 있으면 소화하세요.", flush=True)

    if not frames:
        print("저장할 프레임이 없습니다. 배선, 포트, 전체 프레임 전송을 확인하세요.")
        return 1
    table = pd.DataFrame(records)
    times = table["esp32_ms"].to_numpy(dtype=np.uint32)
    span = float(times[-1] - times[0]) / 1000 if len(times) > 1 else 0
    fps = (len(times) - 1) / span if span > 0 else None
    metadata = {
        "session": session, "condition": args.condition, "distance_cm": args.distance,
        "repeat": args.repeat, "requested_seconds": args.seconds, "refresh_setting": args.refresh,
        "note": args.note, "end_reason": reason, "frame_count": len(frames),
        "actual_fps": fps, **quality,
    }
    np.savez_compressed(
        data_dir / f"{session}.npz", frames=np.stack(frames),
        frame_ids=table["frame_id"].to_numpy(dtype=np.uint32), esp32_ms=times,
        sensor_ta=table["sensor_ta"].to_numpy(dtype=np.float32),
        pc_received_at=table["pc_received_at"].to_numpy(dtype=str),
        condition=np.asarray(args.condition), distance_cm=np.asarray(args.distance),
        repeat=np.asarray(args.repeat), refresh_setting=np.asarray(args.refresh),
        end_reason=np.asarray(reason),
    )
    table.to_csv(data_dir / f"{session}.csv", index=False, encoding="utf-8-sig")
    (data_dir / f"{session}.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print("저장 완료:", data_dir / f"{session}.npz")
    print(f"프레임={len(frames)}, 실제 FPS={fps:.2f}" if fps is not None else f"프레임={len(frames)}")
    print("수집 품질:", quality)
    if reason != "completed":
        print("미완료 세션으로 기록했습니다. 본 분석에서는 제외됩니다. 같은 번호로 재측정하세요.")
    return 0 if reason == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
