import cv2
import time
from src.camera.camera import CameraManager
from src.inference.actions import build_trigger_from_config
from src.inference.gesture_translator import ISLGestureTranslator, smooth_predictions  # noqa: F401 (re-exported)
from src.inference.dynamic_inference import DynamicInferenceEngine
from src.inference.oneshot_dynamic import LiveDynamicMatcher, OneShotDynamicRecognizer
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger("RealTimePipeline")

def run_realtime_application(mode: str = "STATIC", config_path: str = "config.yaml"):
    """
    Unified Real-Time ISL Recognition & Action Trigger System.
    - Tracks hand landmarks using MediaPipe.
    - Classifies ISL signs (A-Z, Thumbs Up/Down, Peace/V, W, etc.).
    - Triggers Assistive Desktop Actions (config.yaml "actions"; defaults):
        * Hold Sign 'W' -> WhatsApp, 'Y' -> YouTube, 'L' -> Google
    STATIC mode classifies each frame (MediaPipe geometry + static CNN).
    DYNAMIC mode classifies the last few seconds of video with the recognizer selected by
    config dynamic.live_method: "oneshot" (60 signs, one reference clip each, frozen
    MobileNetV2 + DTW similarity) or "trained" (MobileNetV2 + Bi-GRU trained on the classes
    listed in models/dynamic/class_mapping.json, softmax probability). It falls back to
    stringing per-frame static labels together only if neither is available.
    """
    cfg = load_config(config_path)
    cam_cfg = cfg.get("camera", {})
    inf_cfg = cfg.get("inference", {})
    dyn_cfg = cfg.get("dynamic", {})

    conf_threshold = inf_cfg.get("confidence_threshold", 0.40)
    mirror_mode = True

    # Initialize Camera Hardware
    cam = CameraManager(
        camera_index=cam_cfg.get("index", 0),
        width=cam_cfg.get("width", 640),
        height=cam_cfg.get("height", 480),
        target_fps=cam_cfg.get("fps", 30)
    )

    if not cam.start():
        logger.error("Failed to start camera hardware. Exiting real-time application.")
        return

    # Initialize Gesture & ISL Translator Engine
    translator = ISLGestureTranslator()

    frame_counter = 0
    prev_time = time.time()
    fps = 0.0

    # Phrase history for dynamic sentence formation
    phrase_tokens = []
    last_added_token = ""
    last_added_time = 0

    # Sign-triggered app launches (W -> WhatsApp, Y -> YouTube, L -> Google by default)
    action_trigger = build_trigger_from_config(cfg)
    action_message = ""
    action_message_time = 0
    shortcut_text = "  |  ".join(f"'{a.key}' -> {a.name}" for a in action_trigger.actions.values())         if action_trigger else "disabled"

    # One-shot dynamic matcher (DYNAMIC mode only)
    dyn_matcher = None
    dyn_score_name = ""
    if mode.upper() == "DYNAMIC":
        live_method = str(dyn_cfg.get("live_method", "trained")).lower()
        if live_method == "trained":
            recognizer = DynamicInferenceEngine()
            dyn_score_name = "prob"
        else:
            recognizer = OneShotDynamicRecognizer(dyn_cfg.get("oneshot_gallery", "models/dynamic_oneshot/gallery.npz"))
            dyn_score_name = "DTW sim, one-shot"
        logger.info(f"DYNAMIC recognizer: {live_method}")
        if recognizer.is_loaded:
            dyn_matcher = LiveDynamicMatcher(
                recognizer,
                window_seconds=float(dyn_cfg.get("live_window_seconds", 4.0)),
                match_every_seconds=float(dyn_cfg.get("live_match_every_seconds", 1.0)),
            )
        else:
            logger.warning(f"Dynamic recognizer unavailable ({recognizer.load_error}); DYNAMIC mode falls back to "
                           f"per-frame static predictions, which is not temporal recognition.")
    last_dyn_token_time = 0.0

    logger.info(f"Starting Unified ISL Translator | Mode: {mode.upper()} | Mirror: {mirror_mode}")
    if not translator.static_cnn.is_loaded:
        logger.warning(f"Static CNN not loaded ({translator.static_cnn.load_error}); only MediaPipe geometric rules will run.")

    try:
        while True:
            ret, raw_frame = cam.read_frame()
            if not ret or raw_frame is None:
                logger.error("Empty frame received from camera feed.")
                break

            # Mirror mode (flip horizontally so movements match user perspective)
            if mirror_mode:
                frame = cv2.flip(raw_frame, 1)
            else:
                frame = raw_frame.copy()

            frame_counter += 1
            curr_time = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / (curr_time - prev_time + 1e-6))
            prev_time = curr_time

            # Run MediaPipe hand tracking & sign translation
            out_frame, current_label, current_confidence, top3_candidates, hand_bbox, distance_hint = translator.process_frame(frame)

            h, w, _ = out_frame.shape

            # -------------------------------------------------------------
            # ASSISTIVE ACTION TRIGGER: HOLD A MAPPED SIGN -> OPEN APP
            # -------------------------------------------------------------
            held_action, hold_progress, fired_action = (None, 0.0, None)
            if action_trigger is not None:
                held_action, hold_progress, fired_action = action_trigger.update(current_label, current_confidence, curr_time)
            if fired_action is not None:
                action_message = f"ACTION EXECUTED: OPENING {fired_action.name.upper()}!"
                action_message_time = curr_time

            # Dynamic mode: one-shot temporal matching on the un-mirrored frames
            if dyn_matcher is not None:
                dyn_matcher.push(cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB), curr_time, hand_bbox is not None)
                if dyn_matcher.poll(curr_time):
                    dyn_label = dyn_matcher.latest_result[0]
                    if dyn_label != last_added_token or (curr_time - last_dyn_token_time) > 3.0:
                        phrase_tokens.append(dyn_label)
                        last_added_token = dyn_label
                        last_dyn_token_time = curr_time
                        if len(phrase_tokens) > 6:
                            phrase_tokens.pop(0)
                if dyn_matcher.latest_result is not None and curr_time - dyn_matcher.latest_result[3] < 3.0:
                    d_label, d_sim, _, _ = dyn_matcher.latest_result
                    cv2.putText(out_frame, f"Dynamic match: {d_label}  ({dyn_score_name} {d_sim:.2f})",
                                (15, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 215, 0), 2)
            # Fallback token accumulation (no gallery): per-frame static labels
            elif mode.upper() == "DYNAMIC":
                if current_confidence >= 0.70 and current_label not in ("Show Hand to Camera", "Analyzing Hand..."):
                    if current_label != last_added_token and (curr_time - last_added_time) > 1.2:
                        token_short = current_label.replace("Sign '", "").replace("'", "").split(" ")[0]
                        phrase_tokens.append(token_short)
                        last_added_token = current_label
                        last_added_time = curr_time
                        if len(phrase_tokens) > 6:
                            phrase_tokens.pop(0)

            # Determine status color
            if current_confidence >= 0.70:
                conf_color = (0, 255, 0)      # Bright Green
            elif current_confidence >= conf_threshold:
                conf_color = (0, 230, 255)    # Amber/Yellow
            else:
                conf_color = (120, 180, 255)  # Light Blue / Gray

            # -------------------------------------------------------------
            # Top Dashboard Header Bar (Height = 115 px)
            # -------------------------------------------------------------
            overlay = out_frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, 115), (20, 20, 25), -1)
            cv2.addWeighted(overlay, 0.85, out_frame, 0.15, 0, out_frame)

            # Line 1: System Info & Distance Coaching
            if distance_hint == "TOO_CLOSE":
                dist_str = "TOO CLOSE (Move hand back slightly)"
                dist_color = (0, 140, 255)
            elif distance_hint == "TOO_FAR":
                dist_str = "TOO FAR (Move hand closer 1-2 ft)"
                dist_color = (0, 200, 255)
            elif distance_hint == "OPTIMAL":
                dist_str = "OPTIMAL (3D Skeleton Tracked)"
                dist_color = (0, 255, 100)
            else:
                dist_str = "SEARCHING (Keep hand 1-2 ft away)"
                dist_color = (180, 180, 180)

            cv2.putText(out_frame, f"Mode: {mode.upper()}  |  FPS: {fps:.1f}  |  Position: {dist_str}", 
                        (15, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.48, dist_color, 1)

            # Line 2: Prominent Recognized Sign Banner
            cv2.putText(out_frame, "Recognized Sign :", (15, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2)
            cv2.putText(out_frame, f"[ {current_label} ]", (210, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 255, 255), 2)

            # Line 2 (Right): Confidence Percentage
            conf_percent = current_confidence * 100.0
            cv2.putText(out_frame, f"{conf_percent:.1f}%", (w - 110, 62),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, conf_color, 2)

            # Draw Confidence Bar Underneath
            bar_x = w - 180
            bar_y = 72
            bar_w = 160
            bar_h = 7
            cv2.rectangle(out_frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
            fill_w = int(bar_w * min(1.0, current_confidence))
            cv2.rectangle(out_frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), conf_color, -1)

            # Line 3: Dynamic Sentence or Top 3 Candidates
            if mode.upper() == "DYNAMIC" and phrase_tokens:
                sentence_str = "Sign Sequence:  " + "  ->  ".join(phrase_tokens)
                cv2.putText(out_frame, sentence_str, (15, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 215, 0), 2)
            elif top3_candidates:
                top3_str = "CNN Top Candidates:  " + "   |   ".join(
                    [f"#{i+1}: [{c[0]}] ({c[1]*100:.0f}%)" for i, c in enumerate(top3_candidates[:3])]
                )
                cv2.putText(out_frame, top3_str, (15, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 220, 255), 1)
            else:
                cv2.putText(out_frame, f"Hold to open: {shortcut_text}", 
                            (15, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (170, 170, 170), 1)

            # -------------------------------------------------------------
            # ACTION TRIGGER OVERLAYS (WHATSAPP HOLD PROGRESS & TRIGGER BANNER)
            # -------------------------------------------------------------
            if held_action is not None and hold_progress > 0.0:
                # Draw animated progress bar in center of screen
                prog_w = 340
                prog_h = 42
                px = int((w - prog_w) / 2)
                py = 135

                cv2.rectangle(out_frame, (px - 5, py - 5), (px + prog_w + 5, py + prog_h + 5), (20, 20, 20), -1)
                cv2.rectangle(out_frame, (px - 5, py - 5), (px + prog_w + 5, py + prog_h + 5), (0, 255, 100), 2)

                # Progress fill
                fill_p = int(prog_w * hold_progress)
                cv2.rectangle(out_frame, (px, py + 22), (px + fill_p, py + prog_h - 4), (0, 255, 0), -1)

                pct = int(hold_progress * 100)
                cv2.putText(out_frame, f"OPENING {held_action.name.upper()}... HOLD SIGN ({pct}%)", (px + 12, py + 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 255), 2)

            # If action was recently triggered, flash success banner for 3.5 seconds
            if (curr_time - action_message_time) < 3.5:
                banner_w = 440
                banner_h = 48
                bx = int((w - banner_w) / 2)
                by = 130
                cv2.rectangle(out_frame, (bx - 5, by - 5), (bx + banner_w + 5, by + banner_h + 5), (0, 180, 0), -1)
                cv2.rectangle(out_frame, (bx - 5, by - 5), (bx + banner_w + 5, by + banner_h + 5), (255, 255, 255), 2)
                cv2.putText(out_frame, action_message, (bx + 15, by + 32),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

            # -------------------------------------------------------------
            # Bottom-Right Picture-in-Picture (PiP) Inset (Cropped Hand)
            # -------------------------------------------------------------
            if hand_bbox is not None:
                xmin, ymin, xmax, ymax = hand_bbox
                cropped_hand = frame[ymin:ymax, xmin:xmax]
                if cropped_hand is not None and cropped_hand.size > 0:
                    pip_size = 110
                    pip_img = cv2.resize(cropped_hand, (pip_size, pip_size))
                    px1 = w - pip_size - 15
                    py1 = h - pip_size - 38
                    px2 = px1 + pip_size
                    py2 = py1 + pip_size

                    cv2.rectangle(out_frame, (px1 - 2, py1 - 16), (px2 + 2, py2 + 2), (30, 30, 30), -1)
                    cv2.rectangle(out_frame, (px1 - 2, py1 - 16), (px2 + 2, py2 + 2), (0, 255, 0), 1)
                    cv2.putText(out_frame, "Hand Crop (AI)", (px1, py1 - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
                    out_frame[py1:py2, px1:px2] = pip_img

            # -------------------------------------------------------------
            # Footer Instructions Bar (Height = 28 px)
            # -------------------------------------------------------------
            cv2.rectangle(out_frame, (0, h - 28), (w, h), (15, 15, 15), -1)
            cv2.putText(out_frame, f"Hold sign: {shortcut_text}  |  [N] Numbers {'ON' if translator.numbers_mode else 'OFF'}  [Q] Exit  [M] Mirror  [C] Clear", 
                        (15, h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (200, 200, 200), 1)

            cv2.imshow("ISL Real-Time Recognition System", out_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                logger.info("User requested exit ('q' key pressed).")
                break
            elif key == ord('n'):
                translator.numbers_mode = not translator.numbers_mode
                logger.info(f"Numbers mode (finger count 0-5): {translator.numbers_mode}")
            elif key == ord('m'):
                mirror_mode = not mirror_mode
                logger.info(f"Mirror mode toggled: {mirror_mode}")
            elif key == ord('c'):
                phrase_tokens.clear()
                if dyn_matcher is not None:
                    dyn_matcher.clear()
                translator.history.clear()

    finally:
        cam.release()
        translator.close()
        if dyn_matcher is not None:
            dyn_matcher.close()
        logger.info("Real-time application terminated safely.")
