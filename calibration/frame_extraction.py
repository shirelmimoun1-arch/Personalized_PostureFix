# ============================================================
# AUTOMATIC PERSONALIZED CALIBRATION
# ============================================================
#
# Purpose:
#   Record automatically labeled posture frames for personalized
#   model calibration.
#
# Calibration procedure:
#   1. Record guided examples of correct ("Good") posture.
#   2. Record natural reading behavior while maintaining good posture.
#   3. Record guided examples of incorrect ("Bad") posture.
#
# Reference images:
#   - Good and Bad calibration poses are demonstrated using reference
#     images displayed next to the live webcam feed.
#   - Each reference pose is recorded as a separate posture segment.
#
# Output:
#   - frames_dir:
#       Saved calibration frames sampled from the webcam.
#
#   - labels.csv:
#       Automatically generated labels containing:
#       Image_ID, Label, Segment_ID
#
# Segment structure:
#   - Guided posture examples receive one Segment_ID per reference pose.
#   - The natural reading recording is divided into fixed-duration
#     subsegments so that temporally related frames can later be kept
#     together during train/test splitting.
#
# Notes:
#   - Frames are horizontally flipped to provide a mirrored webcam view.
#   - Only clean webcam frames are saved; visual instructions and
#     reference images are displayed only in the user interface.
#   - Frames are sampled at target_fps rather than saving every camera
#     frame.
#   - Pressing 'q' allows the user to stop the current calibration stage.
# ============================================================

import cv2
import os
import time
import pandas as pd
# ============================================================
# CALIBRATION CONFIGURATION
# ============================================================

# Duration of the countdown shown before recording each pose.
COUNTDOWN_SECONDS = 3

# Duration for which each guided reference posture is recorded.
HIGHLIGHT_SECONDS = 5

# Duration of each temporal reading segment.
READING_SUBSEGMENT_SECONDS = 5


# Text displayed during the natural reading calibration stage.
# Its purpose is to encourage natural head, facial, and hand movement
# while the user maintains an overall correct sitting posture.
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


# ============================================================
# GENERAL UTILITIES
# ============================================================

def clear_old_frames(frames_dir):
    """
    Remove previously saved calibration frames.

    Args:
        frames_dir (str):
            Directory in which calibration frames are stored.

    Notes:
        The directory is created if it does not already exist.
        Only files directly inside the directory are removed.
    """
    os.makedirs(frames_dir, exist_ok=True)

    for file in os.listdir(frames_dir):
        file_path = os.path.join(frames_dir, file)
        if os.path.isfile(file_path):
            os.remove(file_path)


def save_frame(frame, output_folder, frame_index):
    """
    Save a single calibration frame to disk.

    Args:
        frame:
            OpenCV image to save.

        output_folder (str):
            Directory in which the frame will be stored.

        frame_index (int):
            Sequential index used to generate a unique filename.

    Returns:
        str:
            Filename assigned to the saved frame.
    """
    filename = f"frame_{frame_index:04d}.jpg"
    filepath = os.path.join(output_folder, filename)
    success = cv2.imwrite(filepath, frame)

    if not success:
        raise IOError(f"Failed to save calibration frame: {filepath}")

    return filename

# ============================================================
# REFERENCE IMAGE HANDLING
# ============================================================

def load_reference_images(reference_dir):
    """
    Load posture reference images used during calibration.

    Args:
        reference_dir (str):
            Directory containing the posture reference images.

    Returns:
        list:
            List of (filename, image) tuples for all successfully
            loaded reference images.

    Notes:
        Images are loaded at their original resolution and resized
        dynamically during display according to the webcam frame size.
        If the directory does not exist, an empty list is returned.
    """
    if reference_dir is None or not os.path.isdir(reference_dir):
        return []

    images = []
    valid_extensions = (".jpg", ".jpeg", ".png")

    for filename in sorted(os.listdir(reference_dir)):
        if filename.lower().endswith(valid_extensions):

            path = os.path.join(reference_dir, filename)
            image = cv2.imread(path)

            if image is None:
                continue

            images.append((filename, image))

    return images

def draw_image_group(frame, image_group, active_index, title):
    """
    Draw posture reference images on the live calibration frame.

    Reference images, spacing, text, and panel dimensions are scaled
    dynamically according to the current webcam frame size.

    Args:
        frame:
            Current webcam frame.

        image_group (list):
            List of (filename, image) reference-image tuples.

        active_index (int):
            Index of the currently active reference image.
            Use -1 when no image should be highlighted.

        title (str):
            Text displayed above the reference-image panel.

    Returns:
        frame:
            Webcam frame with the reference-image panel drawn on it.
    """
    if not image_group:
        return frame

    frame_h, frame_w = frame.shape[:2]

    target_h = int(frame_h * 0.30)
    gap = int(frame_h * 0.025)

    start_y = int(frame_h * 0.17)
    panel_top = int(frame_h * 0.08)

    margin_x = int(frame_w * 0.025)
    panel_padding = int(frame_w * 0.012)

    resized_images = []

    for filename, ref_img in image_group:
        original_h, original_w = ref_img.shape[:2]

        scale = target_h / original_h
        target_w = int(original_w * scale)

        resized = cv2.resize(
            ref_img,
            (target_w, target_h),
            interpolation=cv2.INTER_AREA
        )

        resized_images.append((filename, resized))

    panel_img_w = max(
        img.shape[1] for _, img in resized_images
    )

    panel_x = max(
        panel_padding,
        frame_w - panel_img_w - margin_x
    )

    panel_bottom = min(
        frame_h - int(frame_h * 0.02),
        start_y + len(resized_images) * (target_h + gap)
    )

    cv2.rectangle(
        frame,
        (panel_x - panel_padding, panel_top),
        (
            min(frame_w - 1, panel_x + panel_img_w + panel_padding),
            panel_bottom
        ),
        (255, 255, 255),
        -1
    )

    put_relative_text(
        frame,
        title,
        position=(panel_x / frame_w, 0.125),
        base_scale=0.55,
        color=(0, 0, 0),
        base_thickness=2
    )

    for i, (_, ref_img) in enumerate(resized_images):

        img_h, img_w = ref_img.shape[:2]
        y = start_y + i * (img_h + gap)

        if y + img_h > frame_h - int(frame_h * 0.02):
            break

        frame[
            y:y + img_h,
            panel_x:panel_x + img_w
        ] = ref_img

        is_active = i == active_index

        border_color = (
            (0, 0, 255)
            if is_active
            else (150, 150, 150)
        )

        ui_scale = get_ui_scale(frame)

        border_thickness = (
            max(2, int(round(5 * ui_scale)))
            if is_active
            else max(1, int(round(ui_scale)))
        )

        cv2.rectangle(
            frame,
            (panel_x, y),
            (panel_x + img_w, y + img_h),
            border_color,
            border_thickness
        )

        put_relative_text(
            frame,
            str(i + 1),
            position=(
                (panel_x + int(img_w * 0.04)) / frame_w,
                (y + int(img_h * 0.92)) / frame_h
            ),
            base_scale=0.8,
            color=border_color,
            base_thickness=2
        )

    return frame

# ============================================================
# CALIBRATION USER INTERFACE
# ============================================================

def show_countdown_on_camera(cap, image_group, label):
    """
    Display a short countdown before recording a calibration posture.

    The countdown provides the user with time to prepare before
    the timed posture-recording interval begins.
    """
    for count in range(COUNTDOWN_SECONDS, 0, -1):
        start = time.time()

        while time.time() - start < 1:
            ret, frame = cap.read()

            if not ret:
                print("Could not read frame from webcam.")
                return False

            frame = cv2.flip(frame, 1)

            frame = draw_image_group(
                frame=frame,
                image_group=image_group,
                active_index=-1,
                title=f"{label} posture examples"
            )

            put_relative_text(
                frame,
                f"STARTING IN: {count}",
                position=(0.04, 0.07),
                base_scale=1.2,
                color=(0, 0, 255),
                base_thickness=3
            )

            cv2.imshow("Personalized Calibration", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                return False

    # Show zero
    ret, frame = cap.read()

    if not ret:
        print("Could not read frame from webcam.")
        return False

    frame = cv2.flip(frame, 1)

    frame = draw_image_group(
        frame=frame,
        image_group=image_group,
        active_index=-1,
        title=f"{label} posture examples"
    )

    put_relative_text(
        frame,
        "STARTING IN: 0",
        position=(0.04, 0.07),
        base_scale=1.2,
        color=(0, 0, 255),
        base_thickness=3
    )

    cv2.imshow("Personalized Calibration", frame)

    # Force OpenCV to update the displayed frame.
    cv2.waitKey(1)

    return True

# ============================================================
# GUIDED POSTURE RECORDING
# ============================================================

def record_posture_segment_with_images(
    cap,
    label,
    output_folder,
    start_index,
    reference_images,
    target_fps=5
):
    """
    Record labeled posture segments using reference images.

    Each reference image represents one guided posture example.
    The user first confirms readiness, then follows the displayed
    posture while webcam frames are sampled for a fixed duration.

    Args:
        cap:
            OpenCV webcam capture object.

        label (str):
            Class assigned to the recorded frames, typically
            "Good" or "Bad".

        output_folder (str):
            Directory in which sampled frames are saved.

        start_index (int):
            Frame index from which filename numbering should continue.

        reference_images (list):
            List of reference posture images used to guide the user.

        target_fps (int):
            Approximate frame sampling rate for dataset creation.

    Returns:
        tuple:
            saved_rows:
                List of dictionaries containing Image_ID, Label,
                and Segment_ID for each saved frame.

            frame_index:
                Next available frame index after recording.

    Notes:
        All frames recorded for the same reference posture receive
        the same Segment_ID. This allows them to remain grouped during
        later segment-based train/test splitting.

        Frame sampling is time-based rather than relying on the
        webcam-reported FPS, improving consistency across cameras
        and operating systems.
    """

    if target_fps <= 0:
        raise ValueError("target_fps must be greater than zero.")

    saved_rows = []
    frame_index = start_index

    sample_interval = 1.0 / target_fps

    print(
        f"\nRecording {label} posture guided by reference images..."
    )

    total_images = len(reference_images)

    if total_images == 0:
        print(f"No reference images found for {label}.")
        return saved_rows, frame_index

    for pose_idx, (ref_filename, ref_image) in enumerate(reference_images):

        single_image = [(ref_filename, ref_image)]

        # --------------------------------------------------------
        # Wait until the user is ready
        # --------------------------------------------------------

        should_continue = wait_for_start_button(
            cap=cap,
            image_group=single_image,
            label=label
        )

        if not should_continue:
            break

        # --------------------------------------------------------
        # Countdown before recording
        # --------------------------------------------------------

        should_continue = show_countdown_on_camera(
            cap=cap,
            image_group=single_image,
            label=label
        )

        if not should_continue:
            break

        # --------------------------------------------------------
        # Record current posture segment
        # --------------------------------------------------------

        segment_id = f"{label.lower()}_pose_{pose_idx + 1}"

        start_time = time.time()
        last_saved_time = None

        while time.time() - start_time < HIGHLIGHT_SECONDS:

            ret, clean_frame = cap.read()

            if not ret:
                print("Could not read frame from webcam.")
                break

            # Mirror the webcam image.
            clean_frame = cv2.flip(clean_frame, 1)

            # UI elements are drawn only on this copy.
            display_frame = clean_frame.copy()

            current_time = time.time()
            elapsed = current_time - start_time
            remaining = max(0.0, HIGHLIGHT_SECONDS - elapsed)

            # ----------------------------------------------------
            # Draw calibration UI
            # ----------------------------------------------------

            display_frame = draw_image_group(
                frame=display_frame,
                image_group=single_image,
                active_index=0,
                title=f"{label} posture example"
            )

            put_relative_text(
                display_frame,
                f"Calibration: {label}",
                position=(0.03, 0.07),
                base_scale=1.0,
                color=(255, 255, 255),
                base_thickness=2
            )

            put_relative_text(
                display_frame,
                f"Current posture: {pose_idx + 1}/{total_images}",
                position=(0.03, 0.125),
                base_scale=0.8,
                color=(255, 255, 255),
                base_thickness=2
            )

            put_relative_text(
                display_frame,
                f"Time left: {remaining:.1f}s",
                position=(0.03, 0.175),
                base_scale=0.8,
                color=(255, 255, 255),
                base_thickness=2
            )

            cv2.imshow(
                "Personalized Calibration",
                display_frame
            )

            # ----------------------------------------------------
            # Time-based frame sampling
            # ----------------------------------------------------

            if (
                last_saved_time is None
                or current_time - last_saved_time >= sample_interval
            ):
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
                last_saved_time = current_time

            # ----------------------------------------------------
            # Allow user to stop calibration
            # ----------------------------------------------------

            if cv2.waitKey(1) & 0xFF == ord("q"):
                return saved_rows, frame_index

    print(f"Finished recording {label} posture.")

    return saved_rows, frame_index

# ============================================================
# INTERACTIVE CALIBRATION CONTROLS
# ============================================================

def get_ui_scale(frame):
    """
    Calculate a UI scale factor relative to the current frame size.

    The scale is based on a 1280x720 reference resolution and is used
    for text size, line thickness, and spacing.
    """
    frame_h, frame_w = frame.shape[:2]

    scale_x = frame_w / 1280.0
    scale_y = frame_h / 720.0

    return min(scale_x, scale_y)


def relative_point(frame, x_ratio, y_ratio):
    """
    Convert normalized frame coordinates into pixel coordinates.

    Args:
        frame:
            Current OpenCV frame.

        x_ratio (float):
            Horizontal position in the range [0, 1].

        y_ratio (float):
            Vertical position in the range [0, 1].

    Returns:
        tuple:
            Pixel coordinates (x, y).
    """
    frame_h, frame_w = frame.shape[:2]

    x = int(frame_w * x_ratio)
    y = int(frame_h * y_ratio)

    return x, y


def put_relative_text(
    frame,
    text,
    position,
    base_scale=1.0,
    color=(255, 255, 255),
    base_thickness=2
):
    """
    Draw text whose position and size scale with the frame dimensions.

    Args:
        frame:
            Current OpenCV frame.

        text (str):
            Text to display.

        position (tuple):
            Relative (x, y) position in the range [0, 1].

        base_scale (float):
            Font scale at the reference resolution.

        color (tuple):
            OpenCV BGR text color.

        base_thickness (int):
            Text thickness at the reference resolution.
    """
    ui_scale = get_ui_scale(frame)

    font_scale = max(0.45, base_scale * ui_scale)
    thickness = max(1, int(round(base_thickness * ui_scale)))

    x, y = relative_point(
        frame,
        position[0],
        position[1]
    )

    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness
    )

def get_button_rect(frame, position="bottom_center"):
    """
    Calculate a button rectangle relative to the current frame size.

    Args:
        frame:
            Current OpenCV frame.

        position (str):
            Desired button location.

    Returns:
        tuple:
            Button rectangle in the form (x1, y1, x2, y2).
    """
    frame_h, frame_w = frame.shape[:2]

    button_w = int(frame_w * 0.18)
    button_h = int(frame_h * 0.09)

    margin_x = int(frame_w * 0.05)
    margin_y = int(frame_h * 0.05)

    if position == "bottom_center":
        x1 = (frame_w - button_w) // 2
        y1 = frame_h - button_h - margin_y

    elif position == "bottom_right":
        x1 = frame_w - button_w - margin_x
        y1 = frame_h - button_h - margin_y

    elif position == "bottom_left":
        x1 = margin_x
        y1 = frame_h - button_h - margin_y

    else:
        raise ValueError(f"Unsupported button position: {position}")

    return (
        x1,
        y1,
        x1 + button_w,
        y1 + button_h
    )

def draw_blue_button(frame, text="Ready", button_rect=None):
    """
    Draw an interactive button that scales with the frame dimensions.
    """
    if button_rect is None:
        button_rect = get_button_rect(
            frame,
            position="bottom_center"
        )

    x1, y1, x2, y2 = button_rect

    ui_scale = get_ui_scale(frame)

    border_thickness = max(1, int(round(2 * ui_scale)))
    font_scale = max(0.45, 1.0 * ui_scale)
    text_thickness = max(1, int(round(2 * ui_scale)))

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (255, 120, 0),
        -1
    )

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (255, 255, 255),
        border_thickness
    )

    (text_w, text_h), _ = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        text_thickness
    )

    text_x = x1 + (x2 - x1 - text_w) // 2
    text_y = y1 + (y2 - y1 + text_h) // 2

    cv2.putText(
        frame,
        text,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        text_thickness
    )

    return button_rect


def wait_for_reading_ready_screen(cap):
    """
    Display instructions before the natural reading calibration stage.

    The screen explains the reading task and waits until the user
    clicks the Ready button.

    Args:
        cap:
            OpenCV webcam capture object.

    Returns:
        bool:
            True when the user clicks Ready.
            False if the user cancels by pressing 'q'.
    """
    clicked = {"ready": False}
    button_rect = None

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and button_rect is not None:
            x1, y1, x2, y2 = button_rect

            if x1 <= x <= x2 and y1 <= y <= y2:
                clicked["ready"] = True

    cv2.namedWindow("Personalized Calibration")
    cv2.setMouseCallback("Personalized Calibration", mouse_callback)

    while not clicked["ready"]:
        ret, frame = cap.read()

        if not ret:
            print("Could not read frame from webcam.")
            break

        frame = cv2.flip(frame, 1)

        button_rect = get_button_rect(
            frame,
            position="bottom_center"
        )

        overlay = frame.copy()
        frame_h, frame_w = frame.shape[:2]

        margin_x = int(frame_w * 0.04)
        margin_y = int(frame_h * 0.05)

        cv2.rectangle(
            overlay,
            (margin_x, margin_y),
            (frame_w - margin_x, frame_h - margin_y),
            (255, 255, 255),
            -1
        )
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

        start_y = 0.16
        line_spacing = 0.065

        for i, line in enumerate(instructions):

            if line == "":
                continue

            put_relative_text(
                frame,
                line,
                position=(0.08, start_y + i * line_spacing),
                base_scale=0.75,
                color=(0, 0, 0),
                base_thickness=2
            )

        draw_blue_button(frame, text="Ready", button_rect=button_rect)

        cv2.imshow("Personalized Calibration", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            return False

    return True

# ============================================================
# NATURAL READING CALIBRATION
# ============================================================

def record_good_reading_segment(
    cap,
    output_folder,
    start_index,
    target_fps=5
):
    """
    Record good posture while the user performs a natural reading task.

    The reading stage introduces natural head, facial, and hand movement
    while the user attempts to maintain correct sitting posture.
    Sampled frames are automatically labeled as "Good".

    Args:
        cap:
            OpenCV webcam capture object.

        output_folder (str):
            Directory in which sampled frames are saved.

        start_index (int):
            Frame index from which filename numbering should continue.

        target_fps (int):
            Approximate frame sampling rate for dataset creation.

    Returns:
        tuple:
            saved_rows:
                List of dictionaries containing Image_ID, Label,
                and Segment_ID for each saved frame.

            frame_index:
                Next available frame index after recording.

    Notes:
        The recording is divided into fixed-duration temporal
        subsegments. Frames within the same time interval receive the
        same Segment_ID so they can later be grouped during dataset
        splitting.

        Frame sampling is time-based rather than relying on the
        webcam-reported FPS, improving consistency across cameras
        and operating systems.
    """

    if target_fps <= 0:
        raise ValueError("target_fps must be greater than zero.")

    saved_rows = []
    frame_index = start_index

    sample_interval = 1.0 / target_fps
    last_saved_time = None

    print("\nRecording Good posture while reading natural text...")

    ready = wait_for_reading_ready_screen(cap)

    if not ready:
        return saved_rows, frame_index

    clicked = {"end": False}
    end_button_rect = None

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and end_button_rect is not None:
            x1, y1, x2, y2 = end_button_rect

            if x1 <= x <= x2 and y1 <= y <= y2:
                clicked["end"] = True

    cv2.namedWindow("Personalized Calibration")
    cv2.setMouseCallback(
        "Personalized Calibration",
        mouse_callback
    )

    reading_start_time = time.time()

    while not clicked["end"]:

        ret, clean_frame = cap.read()

        if not ret:
            print("Could not read frame from webcam.")
            break

        clean_frame = cv2.flip(clean_frame, 1)
        display_frame = clean_frame.copy()

        end_button_rect = get_button_rect(
            display_frame,
            position="bottom_right"
        )

        frame_h, frame_w = display_frame.shape[:2]

        margin_x = int(frame_w * 0.04)
        margin_y = int(frame_h * 0.05)

        cv2.rectangle(
            display_frame,
            (margin_x, margin_y),
            (frame_w - margin_x, frame_h - margin_y),
            (255, 255, 255),
            -1
        )

        put_relative_text(
            display_frame,
            "Read naturally - Good posture calibration",
            position=(0.07, 0.12),
            base_scale=0.85,
            color=(0, 0, 0),
            base_thickness=2
        )

        start_y = 0.20
        line_spacing = 0.048

        for i, line in enumerate(FUNNY_READING_TEXT):

            if line == "":
                continue

            put_relative_text(
                display_frame,
                line,
                position=(0.07, start_y + i * line_spacing),
                base_scale=0.62,
                color=(0, 0, 0),
                base_thickness=2
            )

        draw_blue_button(
            display_frame,
            text="End",
            button_rect=end_button_rect
        )

        cv2.imshow(
            "Personalized Calibration",
            display_frame
        )

        current_time = time.time()

        if (
            last_saved_time is None
            or current_time - last_saved_time >= sample_interval
        ):
            filename = save_frame(
                frame=clean_frame,
                output_folder=output_folder,
                frame_index=frame_index
            )

            elapsed_reading_time = (
                current_time - reading_start_time
            )

            subsegment_index = int(
                elapsed_reading_time // READING_SUBSEGMENT_SECONDS
            ) + 1

            segment_id = (
                f"good_reading_{subsegment_index:03d}"
            )

            saved_rows.append({
                "Image_ID": filename,
                "Label": "Good",
                "Segment_ID": segment_id
            })

            frame_index += 1
            last_saved_time = current_time

        if cv2.waitKey(1) & 0xFF == ord("q"):
            return saved_rows, frame_index

    print("Finished recording Good reading posture.")

    return saved_rows, frame_index

def wait_for_start_button(cap, image_group, label):
    """
    Wait for the user to confirm readiness before recording a pose.

    Args:
        cap:
            OpenCV webcam capture object.

        image_group (list):
            Reference images displayed to the user.

        label (str):
            Posture class currently being calibrated.

    Returns:
        bool:
            True when the Start button is clicked.
            False if the user cancels by pressing 'q'.
    """
    clicked = {"start": False}
    button_rect = None

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and button_rect is not None:
            x1, y1, x2, y2 = button_rect

            if x1 <= x <= x2 and y1 <= y <= y2:
                clicked["start"] = True

    cv2.namedWindow("Personalized Calibration")
    cv2.setMouseCallback("Personalized Calibration", mouse_callback)

    while not clicked["start"]:
        ret, frame = cap.read()

        if not ret:
            print("Could not read frame from webcam.")
            break

        frame = cv2.flip(frame, 1)
        button_rect = get_button_rect(
            frame,
            position="bottom_left"
        )

        frame = draw_image_group(
            frame=frame,
            image_group=image_group,
            active_index=-1,
            title=f"{label} posture examples"
        )

        put_relative_text(
            frame,
            "Press when ready.",
            position=(0.05, 0.72),
            base_scale=1.0,
            color=(255, 255, 255),
            base_thickness=2
        )

        draw_blue_button(
            frame,
            text="Start",
            button_rect=button_rect
        )

        cv2.imshow("Personalized Calibration", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            return False

    return True

# ============================================================
# CALIBRATION PIPELINE
# ============================================================

def run_calibration(
    frames_dir,
    labels_csv_path,
    target_fps=5,
    good_reference_dir="assets/calibration_images/good",
    bad_reference_dir="assets/calibration_images/bad"
):
    """
    Run the complete personalized calibration procedure.

    The calibration consists of three stages:
        1. Guided Good posture recording.
        2. Natural reading while maintaining Good posture.
        3. Guided Bad posture recording.

    Args:
        frames_dir (str):
            Directory in which calibration frames are saved.

        labels_csv_path (str):
            Path of the automatically generated labels CSV file.

        target_fps (int):
            Approximate sampling rate used when saving frames.

        good_reference_dir (str):
            Directory containing Good-posture reference images.

        bad_reference_dir (str):
            Directory containing Bad-posture reference images.

    Output:
        Saves:
            - Sampled webcam frames to frames_dir.
            - labels.csv containing Image_ID, Label, and Segment_ID.

    Raises:
        RuntimeError:
            If the webcam cannot be opened.

    Notes:
        Existing calibration frames are removed before a new calibration
        session begins. Webcam resources and OpenCV windows are released
        when recording finishes or is interrupted.
    """
    clear_old_frames(frames_dir)

    good_reference_images = load_reference_images(good_reference_dir)
    bad_reference_images = load_reference_images(bad_reference_dir)

    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

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
