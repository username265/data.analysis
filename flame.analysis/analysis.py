from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")  # 창을 띄우지 않고 발표용 PNG/PDF를 저장합니다.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PATTERN = re.compile(r"^(background|flame)_d(\d+)_r(\d+)_(\d{8}_\d{6}(?:_\d+)?)\.npz$")


def load_session(path):
    with np.load(path, allow_pickle=False) as data:
        frames = data["frames"].astype(np.float64)
        times = data["esp32_ms"].astype(np.float64)
        reason = str(data["end_reason"].item()) if "end_reason" in data else "legacy_unknown"
        refresh = str(data["refresh_setting"].item()) if "refresh_setting" in data else "legacy_unknown"
    if frames.ndim != 3 or frames.shape[1:] != (24, 32) or len(frames) != len(times):
        raise ValueError("배열 모양 또는 프레임/시간 개수 불일치")
    if reason not in ("completed", "legacy_unknown"):
        raise ValueError(f"미완료 측정: {reason}")
    if len(times) < 5 or not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0):
        raise ValueError("프레임이 5개 미만이거나 시각이 비정상")
    valid = np.isfinite(frames).all(axis=(1, 2))
    removed = int((~valid).sum())
    frames, times = frames[valid], times[valid]
    if len(frames) < 5:
        raise ValueError("768픽셀이 모두 유효한 프레임이 5개 미만")
    return {"frames": frames, "times": (times - times[0]) / 1000.0,
            "fps": (len(times) - 1) * 1000 / (times[-1] - times[0]),
            "removed": removed, "refresh": refresh, "path": path}


def features(frames, baseline, threshold):
    delta = frames - baseline
    flat = delta.reshape(len(delta), -1)
    k = int(np.ceil(flat.shape[1] * 0.05))
    top = np.partition(flat, flat.shape[1] - k, axis=1)[:, -k:].mean(axis=1)
    hot = delta > threshold
    count = hot.sum(axis=(1, 2))
    yy, xx = np.indices((24, 32))
    cx = np.full(len(delta), np.nan)
    cy = np.full(len(delta), np.nan)
    np.divide((hot * xx).sum(axis=(1, 2)), count, out=cx, where=count > 0)
    np.divide((hot * yy).sum(axis=(1, 2)), count, out=cy, where=count > 0)
    return {"max_delta_c": flat.max(axis=1), "top5_delta_c": top,
            "hot_pixels": count, "centroid_x": cx, "centroid_y": cy}


def save_figure(fig, directory, name):
    fig.savefig(directory / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(directory / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--threshold", type=float, default=2.0, help="최소 온도 차이(°C)")
    parser.add_argument("--sigma", type=float, default=3.0, help="배경 표준편차 배수")
    parser.add_argument("--expected-repeats", type=int, default=5)
    args = parser.parse_args()
    if args.threshold <= 0 or args.sigma < 0 or args.expected_repeats < 1:
        parser.error("임계값/반복 횟수는 양수, sigma는 0 이상이어야 합니다.")
    root = Path(__file__).resolve().parent
    data_dir = (args.data_dir or root / "data").resolve()
    output_dir = (args.output_dir or root / "figures" / "analysis").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates, notes = {}, []
    for path in sorted(data_dir.glob("*.npz")):
        match = PATTERN.match(path.name)
        if not match:
            continue
        condition, distance, repeat, stamp = match.groups()
        key = (int(distance), int(repeat), condition)
        candidates.setdefault(key, []).append((stamp, path))
    if not candidates:
        raise SystemExit("배경/불꽃 실험 파일이 없습니다. background_d20_r01_날짜시간.npz 형식으로 수집하세요.")

    sessions = {}
    quality_rows = []
    for key, versions in sorted(candidates.items()):
        versions.sort()
        stamp, path = versions[-1]
        if len(versions) > 1:
            notes.append(f"중복 {key}: 가장 최근 파일 {path.name}만 선택. 이전 파일은 보존.")
        try:
            session = load_session(path)
        except (ValueError, KeyError, OSError, EOFError) as error:
            quality_rows.append({"file": path.name, "status": "excluded", "reason": str(error)})
            notes.append(f"제외 {path.name}: {error}. 같은 번호로 다시 수집하세요.")
            continue
        sessions[key] = session
        quality_rows.append({"file": path.name, "status": "loaded", "frames": len(session["frames"]),
                             "nonfinite_frames_removed": session["removed"], "actual_fps": session["fps"],
                             "refresh_setting": session["refresh"]})

    summaries, frame_tables, pairs = [], [], []
    pair_keys = sorted({(distance, repeat) for distance, repeat, _ in candidates})
    for distance, repeat in pair_keys:
        bg = sessions.get((distance, repeat, "background"))
        flame = sessions.get((distance, repeat, "flame"))
        if bg is None or flame is None:
            notes.append(f"d{distance} r{repeat:02d}: 유효한 배경/불꽃 쌍이 없어 제외.")
            continue
        if bg["refresh"] != flame["refresh"]:
            notes.append(f"d{distance} r{repeat:02d}: 센서 갱신 설정이 달라 제외.")
            continue
        if max(bg["fps"], flame["fps"]) / min(bg["fps"], flame["fps"]) > 1.3:
            notes.append(f"d{distance} r{repeat:02d}: 실제 FPS 차이가 30%를 넘어 제외. 설정과 누락 확인.")
            continue

        # 배경 앞부분으로 기준을 계산하고, 뒤쪽은 독립적인 배경 비교 구간으로 사용합니다.
        duration = flame["times"][-1]
        split = bg["times"][-1] - duration
        reference_mask = bg["times"] < split
        control_mask = ~reference_mask
        if reference_mask.sum() < 5 or control_mask.sum() < 5:
            notes.append(f"d{distance} r{repeat:02d}: 기준/비교 배경이 각각 5프레임 미만. 배경을 더 길게 수집하세요.")
            continue
        reference = bg["frames"][reference_mask]
        baseline = np.median(reference, axis=0)
        noise = np.std(reference, axis=0, ddof=1)
        threshold = np.maximum(args.threshold, args.sigma * noise)
        bg_control = bg["frames"][control_mask]
        bg_time = bg["times"][control_mask]
        bg_time = bg_time - bg_time[0]
        bg_features = features(bg_control, baseline, threshold)
        flame_features = features(flame["frames"], baseline, threshold)

        for condition, session, values, local_time in [
            ("background", bg, bg_features, bg_time),
            ("flame", flame, flame_features, flame["times"]),
        ]:
            row = {"distance_cm": distance, "repeat": repeat, "condition": condition,
                   "file": session["path"].name, "analyzed_frames": len(local_time),
                   "actual_fps": session["fps"], "mean_max_delta_c": float(np.mean(values["max_delta_c"])),
                   "mean_top5_delta_c": float(np.mean(values["top5_delta_c"])),
                   "mean_hot_pixels": float(np.mean(values["hot_pixels"])),
                   "temporal_std_max_delta_c": float(np.std(values["max_delta_c"], ddof=1)),
                   "temporal_std_hot_pixels": float(np.std(values["hot_pixels"], ddof=1))}
            center_valid = np.isfinite(values["centroid_x"]) & np.isfinite(values["centroid_y"])
            if center_valid.any():
                cx, cy = values["centroid_x"][center_valid], values["centroid_y"][center_valid]
                row["centroid_rms_pixels"] = float(np.sqrt(np.mean((cx - cx.mean()) ** 2 + (cy - cy.mean()) ** 2)))
            else:
                row["centroid_rms_pixels"] = np.nan
            summaries.append(row)
            table = pd.DataFrame(values)
            table.insert(0, "time_s", local_time)
            table.insert(0, "condition", condition)
            table.insert(0, "repeat", repeat)
            table.insert(0, "distance_cm", distance)
            frame_tables.append(table)
        pairs.append({"distance": distance, "repeat": repeat, "baseline": baseline,
                      "refresh": flame["refresh"],
                      "flame_median": np.median(flame["frames"], axis=0),
                      "bg_time": bg_time, "flame_time": flame["times"],
                      "bg_features": bg_features, "flame_features": flame_features})

    pd.DataFrame(quality_rows).to_csv(output_dir / "data_quality.csv", index=False, encoding="utf-8-sig")
    if not pairs:
        report = "\n".join(notes) or "분석 가능한 쌍이 없습니다."
        (output_dir / "analysis_report.txt").write_text(report, encoding="utf-8")
        raise SystemExit(report)

    if len({pair["refresh"] for pair in pairs}) > 1:
        report = "센서 갱신 설정이 다른 쌍들이 섞여 있습니다. 동일 설정으로 측정한 파일만 한 폴더에 모아 분석하세요.\n" + "\n".join(notes)
        (output_dir / "analysis_report.txt").write_text(report, encoding="utf-8")
        raise SystemExit(report)

    session_table = pd.DataFrame(summaries)
    session_table.to_csv(output_dir / "session_features.csv", index=False, encoding="utf-8-sig")
    pd.concat(frame_tables, ignore_index=True).to_csv(output_dir / "frame_features.csv", index=False, encoding="utf-8-sig")
    metrics = ["mean_max_delta_c", "mean_top5_delta_c", "mean_hot_pixels", "temporal_std_max_delta_c"]
    grouped = session_table.groupby(["distance_cm", "condition"])[metrics].agg(["mean", "std", "count"])
    grouped.columns = [f"{metric}_{stat}" for metric, stat in grouped.columns]
    grouped.reset_index().to_csv(output_dir / "distance_summary.csv", index=False, encoding="utf-8-sig")

    distances = sorted({pair["distance"] for pair in pairs})
    representatives = [next(pair for pair in pairs if pair["distance"] == distance) for distance in distances]
    raw_min = min(min(pair["baseline"].min(), pair["flame_median"].min()) for pair in pairs)
    raw_max = max(max(pair["baseline"].max(), pair["flame_median"].max()) for pair in pairs)
    if raw_max <= raw_min:
        raw_max = raw_min + 1
    delta_limit = max(1.0, max(np.abs(pair["flame_median"] - pair["baseline"]).max() for pair in pairs))
    fig, axes = plt.subplots(len(distances), 3, figsize=(12, 3.1 * len(distances)), squeeze=False, constrained_layout=True)
    for row, pair in enumerate(representatives):
        for col, (title, array) in enumerate([
            ("Background reference", pair["baseline"]), ("Flame median", pair["flame_median"]),
            ("Temperature difference", pair["flame_median"] - pair["baseline"]),
        ]):
            delta_plot = col == 2
            image = axes[row, col].imshow(array, interpolation="nearest", origin="upper",
                                          cmap="coolwarm" if delta_plot else "inferno",
                                          vmin=-delta_limit if delta_plot else raw_min,
                                          vmax=delta_limit if delta_plot else raw_max)
            axes[row, col].set_title(f"{pair['distance']} cm, r{pair['repeat']:02d}: {title}")
            axes[row, col].set_xlabel("Pixel X")
            axes[row, col].set_ylabel("Pixel Y")
            fig.colorbar(image, ax=axes[row, col], label="Delta T (°C)" if delta_plot else "Apparent temperature (°C)", shrink=0.8)
    save_figure(fig, output_dir, "01_thermal_maps")

    fig, axes = plt.subplots(len(distances), 2, figsize=(11, 3.1 * len(distances)), squeeze=False, constrained_layout=True)
    for row, pair in enumerate(representatives):
        for condition, color in [("background", "tab:blue"), ("flame", "tab:orange")]:
            values = pair["bg_features" if condition == "background" else "flame_features"]
            local_time = pair["bg_time" if condition == "background" else "flame_time"]
            axes[row, 0].plot(local_time, values["max_delta_c"], marker=".", color=color, label=condition.title())
            axes[row, 1].plot(local_time, values["hot_pixels"], marker=".", color=color, label=condition.title())
        for col in range(2):
            axes[row, col].set_xlabel("Time within each condition (s)")
            axes[row, col].grid(alpha=0.3)
            axes[row, col].legend()
        axes[row, 0].set_title(f"{pair['distance']} cm: Maximum Temperature Difference")
        axes[row, 0].set_ylabel("Maximum Delta T (°C)")
        axes[row, 1].set_title(f"{pair['distance']} cm: Hot Pixel Count")
        axes[row, 1].set_ylabel("Hot pixels")
    save_figure(fig, output_dir, "02_time_series")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for ax, metric, title, ylabel in zip(axes, metrics[:3],
        ["Distance vs Maximum Temperature Difference", "Distance vs Top 5% Temperature Difference", "Distance vs Hot Pixel Count"],
        ["Session mean of maximum Delta T (°C)", "Session mean of top 5% Delta T (°C)", "Session mean hot pixels"]):
        for condition, color, offset in [("background", "tab:blue", -0.35), ("flame", "tab:orange", 0.35)]:
            subset = session_table[session_table.condition == condition]
            stats = subset.groupby("distance_cm")[metric].agg(["mean", "std", "count"])
            x = stats.index.to_numpy(dtype=float) + offset
            ax.plot(x, stats["mean"], marker="o", color=color, label=condition.title())
            enough = stats["count"] >= 2
            if enough.any():
                ax.errorbar(x[enough], stats.loc[enough, "mean"], yerr=stats.loc[enough, "std"], fmt="none", capsize=4, color=color)
            for distance in distances:
                vals = subset.loc[subset.distance_cm == distance, metric].to_numpy()
                ax.scatter(np.full(len(vals), distance + offset), vals, s=20, alpha=0.4, color=color)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Distance (cm)")
        ax.set_ylabel(ylabel)
        ax.set_xticks(distances)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle("Points: individual sessions; error bars: SD across repeated sessions", fontsize=10)
    save_figure(fig, output_dir, "03_distance_comparison")

    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    positions = np.arange(len(distances))
    for condition, color, shift in [("background", "tab:blue", -0.18), ("flame", "tab:orange", 0.18)]:
        stats = session_table[session_table.condition == condition].groupby("distance_cm")[metrics[3]].agg(["mean", "std", "count"]).reindex(distances)
        ax.bar(positions + shift, stats["mean"], width=0.34, color=color, alpha=0.8, label=condition.title())
        enough = stats["count"] >= 2
        if enough.any():
            ax.errorbar((positions + shift)[enough], stats.loc[enough, "mean"], yerr=stats.loc[enough, "std"], fmt="none", capsize=4, color="black")
    ax.set_xticks(positions, [str(distance) for distance in distances])
    ax.set_xlabel("Distance (cm)")
    ax.set_ylabel("Within-session SD of maximum Delta T (°C)")
    ax.set_title("Temporal Variability of Measured Thermal Signals")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    save_figure(fig, output_dir, "04_temporal_variability")

    report = [
        f"분석한 배경/불꽃 쌍: {len(pairs)}개", f"분석한 거리: {distances}",
        f"고온 픽셀 기준: Delta T > max({args.threshold:g}°C, {args.sigma:g} × 픽셀별 배경 표준편차)",
        "배경 앞부분: 기준 중앙값·잡음 계산. 배경 뒷부분: 불꽃과 비슷한 길이의 비교 구간.",
        "원본 NPZ/CSV 파일은 수정하지 않습니다. 비유한 온도 픽셀이 있는 프레임은 이번 분석에서 제외.",
        "거리별 오차 막대: 세션 요약값의 반복 간 표준편차(SD). 프레임을 독립 반복으로 세지 않습니다.",
        "미완료 세션은 제외. 같은 조건/거리/반복 번호가 여러 개면 가장 최근 파일만 사용.",
        "대표 열화상·시간 그래프: 각 거리의 가장 작은 유효 반복 번호. 가장 강한 프레임을 고르지 않음.",
        "배경 대비 검출 영역은 물리적 불꽃 면적이 아니라 픽셀 수입니다.",
        "이 분석은 불꽃 실제 온도 또는 불꽃의 빠른 진동 주파수를 측정하지 않습니다.",
    ]
    for distance in distances:
        found = sorted(pair["repeat"] for pair in pairs if pair["distance"] == distance)
        missing = sorted(set(range(1, args.expected_repeats + 1)) - set(found))
        report.append(f"{distance}cm: 유효 반복 번호 {found}; 목표 반복 중 미완료 {missing}")
    report.extend(notes)
    text = "\n".join(report)
    (output_dir / "analysis_report.txt").write_text(text, encoding="utf-8")
    print(text)
    print("\n분석 완료. CSV, PNG, PDF 저장 위치:", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
