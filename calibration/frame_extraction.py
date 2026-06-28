import cv2
import os
import time
import pandas as pd

COUNTDOWN_SECONDS = 3
HIGHLIGHT_SECONDS = 5

READING_SEGMENT_SECONDS = 25

FUNNY_READING_TEXT = [
    "Imagine explaining to your future self that you ignored",
    "posture warnings from an AI trained entirely on videos",
    "of you slowly becoming a shrimp.",
    "The AI tried its best.",
    "It watched patiently as your neck moved closer and closer",
    "to the screen like a curious turtle searching for Wi-Fi.",
    "",
    "At first, it warned you gently.",
    "Then it started judging you silently.",
    "",
    "Eventually, after the 47th warning, it simply accepted",
    "your fate and began classifying you as:",
    '"aquatic creature."',
    "",
    "Use the classifier responsibly. Your spine is depending on it."
]


def beep():
    print("\a", end="", flush=True)


def clear_old_frames(frames_dir):
    os.makedirs(frames_dir, exist_ok=True)

    for file in os.listdir(frames_dir):
        file_path = os.path.join(frames_dir, file)
        if os.path.isfile(file_path):
            os.remove(file_path)


def save_frame(frame, output_folder, frame_index):
    filename = f"frame_{frame_index:04d}.jpg"
    filepath = os.path.join(output_folder, filename)
    cv2.imwrite(filepath, frame)
    return filename


def load_reference_images(reference_dir):
    if reference_dir is None or not os.path.isdir(reference_dir):
        return []

    images = []
    valid_extensions = (".jpg", ".jpeg", ".png")

    TARGET_HEIGHT = 220  # larger and clearer

    for filename in sorted(os.listdir(reference_dir)):
        if filename.lower().endswith(valid_extensions):

            path = os.path.join(reference_dir, filename)
            image = cv2.imread(path)

            if image is None:
                continue

            h, w = image.shape[:2]

            scale = TARGET_HEIGHT / h

            new_w = int(w * scale)
            new_h = TARGET_HEIGHT

            resized = cv2.resize(
                image,
                (new_w, new_h),
                interpolation=cv2.INTER_AREA
            )

            images.append((filename, resized))

    return images

def draw_image_group(frame, image_group, active_index, title):
    frame_h, frame_w = frame.shape[:2]
    panel_img_h = 220
    panel_img_w = max(img.shape[1] for _, img in image_group)

    panel_x = max(10, frame_w - panel_img_w - 30)
    start_y = 120
    gap = 20

    # white background panel
    panel_top = 60
    panel_bottom = min(
        frame_h - 10,
        start_y + len(image_group) * (panel_img_h + gap)
    )
    cv2.rectangle(
        frame,
        (panel_x - 15, panel_top),
        (panel_x + panel_img_w + 15, panel_bottom),
        (255, 255, 255),
        -1
    )

    cv2.putText(
        frame,
        title,
        (panel_x, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        2
    )

    for i, (filename, ref_img) in enumerate(image_group):

        img_h, img_w = ref_img.shape[:2]
        y = start_y + i * (img_h + gap)

        if y + img_h > frame_h - 10:
            break

        frame[y:y + img_h, panel_x:panel_x + img_w] = ref_img

        is_active = i == active_index
        border_color = (0, 0, 255) if is_active else (150, 150, 150)
        border_thickness = 5 if is_active else 1

        cv2.rectangle(
            frame,
            (panel_x, y),
            (panel_x + img_w, y + img_h),
            border_color,
            border_thickness
        )

        cv2.putText(
            frame,
            str(i + 1),
            (panel_x + 8, y + img_h - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            border_color,
            2
        )

    return frame

def show_countdown_on_camera(cap, image_group, label):
    for count in range(COUNTDOWN_SECONDS, 0, -1):
        start = time.time()

        while time.time() - start < 1:
            ret, frame = cap.read()

            frame = cv2.flip(frame, 1)

            if not ret:
                continue

            frame = draw_image_group(
                frame=frame,
                image_group=image_group,
                active_index=-1,
                title=f"{label} posture examples"
            )

            cv2.putText(
                frame,
                f"STARTING IN: {count}",
                (30, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 0, 255),  # red
                3
            )

            cv2.imshow("Personalized Calibration", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                return False

    beep()
    return True


def record_posture_segment_with_images(
    cap,
    label,
    output_folder,
    start_index,
    reference_images,
    target_fps=5
):
    saved_rows = []
    frame_index = start_index

    camera_fps = cap.get(cv2.CAP_PROP_FPS)
    if camera_fps <= 0:
        camera_fps = 30

    hop_interval = max(1, int(camera_fps / target_fps))

    print(f"\nRecording {label} posture guided by reference images...")

    total_images = len(reference_images)

    if total_images == 0:
        print(f"No reference images found for {label}.")
        return saved_rows, frame_index

    for pose_idx, (ref_filename, ref_image) in enumerate(reference_images):
        single_image = [(ref_filename, ref_image)]

        should_continue = wait_for_start_button(
            cap=cap,
            image_group=single_image,
            label=label
        )

        if not should_continue:
            break

        should_continue = show_countdown_on_camera(
            cap=cap,
            image_group=single_image,
            label=label
        )

        if not should_continue:
            break

        beep()

        segment_id = f"{label.lower()}_pose_{pose_idx + 1}"

        start_time = time.time()
        frame_count = 0

        while time.time() - start_time < HIGHLIGHT_SECONDS:
            ret, clean_frame = cap.read()
            clean_frame = cv2.flip(clean_frame, 1)

            if not ret:
                print("Could not read frame from webcam.")
                break

            display_frame = clean_frame.copy()

            elapsed = time.time() - start_time
            remaining = max(0, HIGHLIGHT_SECONDS - elapsed)

            display_frame = draw_image_group(
                frame=display_frame,
                image_group=single_image,
                active_index=0,
                title=f"{label} posture example"
            )

            cv2.putText(
                display_frame,
                f"Calibration: {label}",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2
            )

            cv2.putText(
                display_frame,
                f"Current posture: {pose_idx + 1}/{total_images}",
                (30, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            cv2.putText(
                display_frame,
                f"Time left: {remaining:.1f}s",
                (30, 125),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            cv2.imshow("Personalized Calibration", display_frame)

            if frame_count % hop_interval == 0:
                filename = save_frame(
                    frame=clean_frame,
                    output_folder=output_folder,
                    frame_index=frame_index
                )

                saved_rows.append({
                    "Image_ID": filename,
                    "Label": label,
                    "Segment_ID": segment_id
                })

                frame_index += 1

            frame_count += 1

            if cv2.waitKey(1) & 0xFF == ord("q"):
                return saved_rows, frame_index

    print(f"Finished recording {label} posture.")

    return saved_rows, frame_index

def draw_blue_button(frame, text="Ready", button_rect=None):
    if button_rect is None:
        button_rect = (220, 420, 420, 490)  # x1, y1, x2, y2

    x1, y1, x2, y2 = button_rect

    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 120, 0), -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

    cv2.putText(
        frame,
        text,
        (x1 + 55, y1 + 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2
    )

    return button_rect


def wait_for_reading_ready_screen(cap):
    clicked = {"ready": False}
    button_rect = (220, 420, 420, 490)

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            x1, y1, x2, y2 = button_rect
            if x1 <= x <= x2 and y1 <= y <= y2:
                clicked["ready"] = True

    cv2.namedWindow("Personalized Calibration")
    cv2.setMouseCallback("Personalized Calibration", mouse_callback)

    while not clicked["ready"]:
        ret, frame = cap.read()

        if not ret:
            continue

        frame = cv2.flip(frame, 1)

        overlay = frame.copy()
        cv2.rectangle(overlay, (40, 40), (frame.shape[1] - 40, frame.shape[0] - 40), (255, 255, 255), -1)
        frame = cv2.addWeighted(overlay, 0.85, frame, 0.15, 0)

        instructions = [
            "Next, a short text will appear on the screen.",
            "Please read it naturally, with normal facial expressions",
            "and natural hand gestures.",
            "",
            "Try to keep a straight, healthy sitting posture.",
            "",
            "Click the blue button when you are ready."
        ]

        y = 110
        for line in instructions:
            cv2.putText(
                frame,
                line,
                (80, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 0, 0),
                2
            )
            y += 45

        draw_blue_button(frame, text="Ready", button_rect=button_rect)

        cv2.imshow("Personalized Calibration", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            return False

    beep()
    return True

def record_good_reading_segment(
    cap,
    output_folder,
    start_index,
    target_fps=5
):
    saved_rows = []
    frame_index = start_index

    camera_fps = cap.get(cv2.CAP_PROP_FPS)
    if camera_fps <= 0:
        camera_fps = 30

    hop_interval = max(1, int(camera_fps / target_fps))

    print("\nRecording Good posture while reading natural text...")

    ready = wait_for_reading_ready_screen(cap)
    if not ready:
        return saved_rows, frame_index

    clicked = {"end": False}
    end_button_rect = (700, 560, 900, 640)  # x1, y1, x2, y2

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            x1, y1, x2, y2 = end_button_rect
            if x1 <= x <= x2 and y1 <= y <= y2:
                clicked["end"] = True

    cv2.namedWindow("Personalized Calibration")
    cv2.setMouseCallback("Personalized Calibration", mouse_callback)

    frame_count = 0

    READING_SUBSEGMENT_SECONDS = 5
    reading_start_time = time.time()

    while not clicked["end"]:
        ret, clean_frame = cap.read()

        if not ret:
            print("Could not read frame from webcam.")
            break

        clean_frame = cv2.flip(clean_frame, 1)
        display_frame = clean_frame.copy()

        cv2.rectangle(
            display_frame,
            (40, 40),
            (display_frame.shape[1] - 40, display_frame.shape[0] - 40),
            (255, 255, 255),
            -1
        )

        cv2.putText(
            display_frame,
            "Read naturally - Good posture calibration",
            (70, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (0, 0, 0),
            2
        )

        y = 140
        for line in FUNNY_READING_TEXT:
            cv2.putText(
                display_frame,
                line,
                (70, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (0, 0, 0),
                2
            )
            y += 35

        # End button
        cv2.rectangle(
            display_frame,
            (end_button_rect[0], end_button_rect[1]),
            (end_button_rect[2], end_button_rect[3]),
            (255, 120, 0),
            -1
        )

        cv2.rectangle(
            display_frame,
            (end_button_rect[0], end_button_rect[1]),
            (end_button_rect[2], end_button_rect[3]),
            (255, 255, 255),
            2
        )

        cv2.putText(
            display_frame,
            "End",
            (end_button_rect[0] + 45, end_button_rect[1] + 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2
        )

        cv2.imshow("Personalized Calibration", display_frame)

        if frame_count % hop_interval == 0:
            filename = save_frame(
                frame=clean_frame,
                output_folder=output_folder,
                frame_index=frame_index
            )

            elapsed_reading_time = time.time() - reading_start_time
            subsegment_index = int(
                elapsed_reading_time // READING_SUBSEGMENT_SECONDS) + 1
            segment_id = f"good_reading_{subsegment_index:03d}"

            saved_rows.append({
                "Image_ID": filename,
                "Label": "Good",
                "Segment_ID": segment_id
            })

            frame_index += 1

        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord("q"):
            return saved_rows, frame_index

    beep()
    print("Finished recording Good reading posture.")

    return saved_rows, frame_index

def wait_for_start_button(cap, image_group, label):
    clicked = {"start": False}
    button_rect = (100, 420, 340, 490)

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            x1, y1, x2, y2 = button_rect
            if x1 <= x <= x2 and y1 <= y <= y2:
                clicked["start"] = True

    cv2.namedWindow("Personalized Calibration")
    cv2.setMouseCallback("Personalized Calibration", mouse_callback)

    while not clicked["start"]:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)

        frame = draw_image_group(
            frame=frame,
            image_group=image_group,
            active_index=-1,
            title=f"{label} posture examples"
        )

        cv2.putText(
            frame,
            "Press when ready.",
            (60, 360),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2
        )

        draw_blue_button(
            frame,
            text="Start",
            button_rect=button_rect
        )

        cv2.imshow("Personalized Calibration", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            return False

    beep()
    return True


def run_calibration(
    frames_dir,
    labels_csv_path,
    target_fps=5,
    good_reference_dir="assets/calibration_images/good",
    bad_reference_dir="assets/calibration_images/bad"
):
    clear_old_frames(frames_dir)

    good_reference_images = load_reference_images(good_reference_dir)
    bad_reference_images = load_reference_images(bad_reference_dir)

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    all_rows = []
    next_frame_index = 0

    try:
        # Good posture examples
        rows, next_frame_index = record_posture_segment_with_images(
            cap=cap,
            label="Good",
            output_folder=frames_dir,
            start_index=next_frame_index,
            reference_images=good_reference_images,
            target_fps=target_fps
        )

        all_rows.extend(rows)

        # Good reading/natural expression segment
        rows, next_frame_index = record_good_reading_segment(
            cap=cap,
            output_folder=frames_dir,
            start_index=next_frame_index,
            target_fps=target_fps
        )

        all_rows.extend(rows)

        # Bad posture examples
        rows, next_frame_index = record_posture_segment_with_images(
            cap=cap,
            label="Bad",
            output_folder=frames_dir,
            start_index=next_frame_index,
            reference_images=bad_reference_images,
            target_fps=target_fps
        )

        all_rows.extend(rows)

    finally:
        cap.release()
        cv2.destroyAllWindows()

    labels_df = pd.DataFrame(all_rows)
    labels_df.to_csv(labels_csv_path, index=False)

    print("\nCalibration complete.")
    print(f"Saved frames to: {frames_dir}")
    print(f"Saved labels to: {labels_csv_path}")
