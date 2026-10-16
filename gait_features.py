"""Extract interpretable gait features from a walking video using 2D pose landmarks."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

if not hasattr(mp, "solutions"):
    raise RuntimeError(
        "This script requires MediaPipe's legacy Solutions API, which is missing "
        "from the installed version. Create and use the project .venv as "
        "described in the README Setup section."
    )

JOINTS = {
    "left_hip": mp.solutions.pose.PoseLandmark.LEFT_HIP.value,
    "right_hip": mp.solutions.pose.PoseLandmark.RIGHT_HIP.value,
    "left_knee": mp.solutions.pose.PoseLandmark.LEFT_KNEE.value,
    "right_knee": mp.solutions.pose.PoseLandmark.RIGHT_KNEE.value,
    "left_ankle": mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value,
    "right_ankle": mp.solutions.pose.PoseLandmark.RIGHT_ANKLE.value,
}


def angle_at_joint(point_a, point_b, point_c):
    """Return the angle ABC in degrees; return NaN for invalid coordinates."""
    a, b, c = (np.asarray(p, dtype=float) for p in (point_a, point_b, point_c))
    if not (np.isfinite(a).all() and np.isfinite(b).all() and np.isfinite(c).all()):
        return np.nan
    ba, bc = a - b, c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom < 1e-9:
        return np.nan
    cosine = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def extract_landmarks(video_path: Path, output_dir: Path, save_overlay: bool = True):
    """Extract six lower-body landmarks per frame and save them to CSV."""
    pose_api = mp.solutions.pose
    drawing = mp.solutions.drawing_utils
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        capture.release()
        raise ValueError("The video does not report a valid frame rate.")

    overlay = None
    if save_overlay:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        overlay = cv2.VideoWriter(
            str(output_dir / "skeleton_overlay.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )

    rows, frame_index = [], 0
    try:
        with pose_api.Pose(static_image_mode=False, model_complexity=1,
                           min_detection_confidence=0.5,
                           min_tracking_confidence=0.5) as pose:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                result = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                row = {"frame": frame_index, "time_s": frame_index / fps}
                for name in JOINTS:
                    row[f"{name}_x"] = np.nan
                    row[f"{name}_y"] = np.nan
                    row[f"{name}_visibility"] = np.nan

                if result.pose_landmarks:
                    landmarks = result.pose_landmarks.landmark
                    for name, landmark_index in JOINTS.items():
                        landmark = landmarks[landmark_index]
                        row[f"{name}_x"] = landmark.x
                        row[f"{name}_y"] = landmark.y
                        row[f"{name}_visibility"] = landmark.visibility
                    drawing.draw_landmarks(frame, result.pose_landmarks, pose_api.POSE_CONNECTIONS)

                rows.append(row)
                if overlay is not None:
                    overlay.write(frame)
                frame_index += 1
    finally:
        capture.release()
        if overlay is not None:
            overlay.release()

    if not rows:
        raise ValueError("No frames could be read from the video.")
    keypoints = pd.DataFrame(rows)
    keypoints.to_csv(output_dir / "keypoints.csv", index=False)
    return keypoints, fps


def compute_features(keypoints: pd.DataFrame, fps: float):
    """Calculate summary angles and rough step-timing features from landmarks."""
    df = keypoints.copy()
    # Ignore low-confidence landmarks rather than treating them as reliable coordinates.
    for joint in JOINTS:
        low_confidence = df[f"{joint}_visibility"] < 0.35
        df.loc[low_confidence, [f"{joint}_x", f"{joint}_y"]] = np.nan

    left_knee, right_knee, hip_angle = [], [], []
    for row in df.itertuples(index=False):
        values = row._asdict()
        def point(joint):
            return [values[f"{joint}_x"], values[f"{joint}_y"]]
        left_knee.append(angle_at_joint(point("left_hip"), point("left_knee"), point("left_ankle")))
        right_knee.append(angle_at_joint(point("right_hip"), point("right_knee"), point("right_ankle")))
        hip_angle.append(angle_at_joint(point("left_knee"), point("left_hip"), point("right_hip")))

    df["left_knee_angle_deg"] = left_knee
    df["right_knee_angle_deg"] = right_knee
    df["hip_angle_deg"] = hip_angle

    # Ankle y-coordinates are normalised image coordinates: lower y means higher in the image.
    ankle_y = df["left_ankle_y"].interpolate(limit_direction="both").to_numpy()
    if len(ankle_y) >= 3 and np.isfinite(ankle_y).any():
        # Smooth slightly to reduce single-frame detection noise.
        window = min(5, len(ankle_y) if len(ankle_y) % 2 else len(ankle_y) - 1)
        smooth_y = pd.Series(ankle_y).rolling(window=max(1, window), center=True, min_periods=1).mean().to_numpy()
        peaks, _ = find_peaks(-smooth_y, distance=max(1, int(0.25 * fps)))
        step_intervals = np.diff(peaks) / fps
    else:
        step_intervals = np.array([])

    avg_step_duration = float(np.mean(step_intervals)) if len(step_intervals) else np.nan
    cadence = float(60.0 / avg_step_duration) if np.isfinite(avg_step_duration) and avg_step_duration > 0 else np.nan
    ankle_displacement = float(np.nanmean(np.abs(np.diff(ankle_y)))) if len(ankle_y) > 1 else np.nan

    summary = {
        "left_knee_angle_mean_deg": float(np.nanmean(df["left_knee_angle_deg"])) if df["left_knee_angle_deg"].notna().any() else np.nan,
        "left_knee_angle_variance": float(np.nanvar(df["left_knee_angle_deg"])) if df["left_knee_angle_deg"].notna().any() else np.nan,
        "right_knee_angle_mean_deg": float(np.nanmean(df["right_knee_angle_deg"])) if df["right_knee_angle_deg"].notna().any() else np.nan,
        "right_knee_angle_variance": float(np.nanvar(df["right_knee_angle_deg"])) if df["right_knee_angle_deg"].notna().any() else np.nan,
        "hip_angle_mean_deg": float(np.nanmean(df["hip_angle_deg"])) if df["hip_angle_deg"].notna().any() else np.nan,
        "mean_step_interval_s": avg_step_duration,
        "estimated_cadence_steps_per_min": cadence,
        "mean_left_ankle_vertical_displacement_per_frame": ankle_displacement,
    }
    return df, summary


def save_angle_plot(df: pd.DataFrame, output_path: Path):
    """Save a knee-angle time series plot."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["time_s"], df["left_knee_angle_deg"], label="Left knee")
    ax.plot(df["time_s"], df["right_knee_angle_deg"], label="Right knee")
    ax.set(title="Knee joint angles during walking", xlabel="Time (seconds)", ylabel="Angle (degrees)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Extract interpretable gait features from a walking video.")
    parser.add_argument("video", type=Path, help="Path to an input walking video")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Directory for CSV, plot and feature summary")
    parser.add_argument("--no-overlay", action="store_true", help="Do not write a skeleton-overlay video")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    keypoints, fps = extract_landmarks(args.video, args.output_dir, save_overlay=not args.no_overlay)
    angle_data, summary = compute_features(keypoints, fps)
    angle_data.to_csv(args.output_dir / "frame_level_features.csv", index=False)
    pd.Series(summary, name="value").to_csv(args.output_dir / "gait_feature_summary.csv", header=True)
    save_angle_plot(angle_data, args.output_dir / "knee_angles.png")

    print("\nGait feature summary")
    for name, value in summary.items():
        print(f"- {name}: {value:.4f}" if np.isfinite(value) else f"- {name}: unavailable")
    print(f"\nSaved results to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
