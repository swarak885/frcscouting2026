import streamlit as st
import cv2
import pandas as pd
import numpy as np
import tempfile
import matplotlib.pyplot as plt
import os

st.title("FRC Match Scouting Tool - Robot Shot Events")

OUTPUT_FOLDER = "outputs"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

ZONES = {
    "hub": (100, 50, 400, 300),
    "climb": (500, 100, 700, 300)
}

ALLIANCE_SPLIT_X = 400  # left=Red, right=Blue

def is_in_zone(x, y, zone):
    x1, y1, x2, y2 = zone
    return x1 <= x <= x2 and y1 <= y <= y2

def get_alliance(x):
    return "Red" if x < ALLIANCE_SPLIT_X else "Blue"

uploaded_video = st.file_uploader("Upload match video", type=["mp4","mov"])

if uploaded_video:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())

    cap = cv2.VideoCapture(tfile.name)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    st.write(f"Total frames detected: {frame_count}")

    all_data = []
    first_frame = None
    skip_initial_frames = 5

    # Track previous in_hub state per robot for counting shots
    prev_in_hub = {}

    for frame_idx in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (21, 21), 0)

        if first_frame is None:
            if frame_idx < skip_initial_frames:
                continue
            first_frame = blur
            continue

        frame_delta = cv2.absdiff(first_frame, blur)
        thresh = cv2.threshold(frame_delta, 15, 255, cv2.THRESH_BINARY)[1]

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt_idx, cnt in enumerate(contours):
            x, y, w, h = cv2.boundingRect(cnt)
            if w*h < 50:
                continue
            cx, cy = x + w // 2, y + h // 2
            in_hub = is_in_zone(cx, cy, ZONES["hub"])
            in_climb = is_in_zone(cx, cy, ZONES["climb"])
            alliance = get_alliance(cx)

            robot_name = f"robot_{cnt_idx+1}"

            # Determine if this frame counts as a new shot
            if robot_name not in prev_in_hub:
                prev_in_hub[robot_name] = False

            shot_event = 0
            if in_hub and not prev_in_hub[robot_name]:
                shot_event = 1  # robot just entered hub = shot
            prev_in_hub[robot_name] = in_hub

            all_data.append({
                "robot": robot_name,
                "frame": frame_idx,
                "x": cx,
                "y": cy,
                "in_hub": in_hub,
                "in_climb": in_climb,
                "alliance": alliance,
                "shot": shot_event
            })

    cap.release()

    if not all_data:
        st.warning("No motion detected. Try a different video or adjust lighting.")
    else:
        df = pd.DataFrame(all_data)

        # Breakdown per robot
        breakdown = df.groupby("robot").agg(
            total_frames=("frame","count"),
            hub_entries=("shot","sum"),
            climb_frames=("in_climb","sum")
        )
        breakdown["climb_ratio"] = breakdown["climb_frames"] / breakdown["total_frames"]

        # Add alliance (most common position)
        alliance_series = df.groupby("robot")["alliance"].agg(lambda x: x.mode()[0])
        breakdown["alliance"] = alliance_series

        st.subheader("Robot Breakdown (Shots counted per hub entry)")
        st.dataframe(breakdown)

        # Download CSV
        csv_path = os.path.join(OUTPUT_FOLDER, "robot_shots_breakdown.csv")
        breakdown.to_csv(csv_path)
        st.download_button(
            "Download Robot Shots CSV",
            breakdown.to_csv(index=True),
            file_name="robot_shots_breakdown.csv"
        )

        # Heatmaps per robot
        st.subheader("Robot Heatmaps")
        robot_names = df["robot"].unique()
        for robot in robot_names:
            robot_data = df[df["robot"]==robot]
            plt.figure()
            plt.hist2d(robot_data["x"], robot_data["y"], bins=[50,30], cmap="Reds")
            plt.title(f"{robot} Heatmap")
            plt.xlabel("X position")
            plt.ylabel("Y position")
            plt.gca().invert_yaxis()
            st.pyplot()
