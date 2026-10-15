import cv2
import time
import math
import collections
import numpy as np
import os

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from src.inference.static_inference import StaticInferenceEngine
from src.utils.config import resolve_path
from src.utils.logger import get_logger

logger = get_logger("GestureTranslator")

# Hand landmark indices
WRIST = 0
THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP = 3
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_PIP = 6
INDEX_DIP = 7
INDEX_TIP = 8
MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_DIP = 11
MIDDLE_TIP = 12
RING_MCP = 13
RING_PIP = 14
RING_DIP = 15
RING_TIP = 16
PINKY_MCP = 17
PINKY_PIP = 18
PINKY_DIP = 19
PINKY_TIP = 20

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (5, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring
    (13, 17), (17, 18), (18, 19), (19, 20),# Pinky
    (0, 17)                                # Palm base
]


def smooth_predictions(history) -> str:
    """Majority vote over recent labels; ties go to the label seen first in the window."""
    if not history:
        return ""
    return collections.Counter(history).most_common(1)[0][0]


class ISLGestureTranslator:
    """
    Real-time ISL Gesture Recognition and Hand Tracking Engine.
    Combines MediaPipe 21-landmark geometric classification with the trained CNN model
    and skin-contour fallback when the hand is close or palm is partially occluded.
    """

    def __init__(self, model_task_path: str = "gesture_recognizer.task"):
        self.model_task_path = str(resolve_path(model_task_path))
        self.recognizer = None
        self.static_cnn = StaticInferenceEngine()
        self.history = collections.deque(maxlen=3)
        self.last_bbox = None
        self.last_bbox_age = 0
        self.numbers_mode = False  # True: one-hand finger counting 0-5 instead of letters
        self._init_recognizer()

    def _init_recognizer(self):
        if not os.path.exists(self.model_task_path):
            logger.warning(f"Task model {self.model_task_path} not found. Will use fallback detection.")
            return

        try:
            base_options = python.BaseOptions(model_asset_path=self.model_task_path)
            options = vision.GestureRecognizerOptions(
                base_options=base_options,
                num_hands=2,  # ISL fingerspelling mostly uses two hands
                min_hand_detection_confidence=0.15,
                min_hand_presence_confidence=0.15,
                min_tracking_confidence=0.15
            )
            self.recognizer = vision.GestureRecognizer.create_from_options(options)
            logger.info("MediaPipe Gesture Recognizer initialized (confidence=0.15, ultra-responsive) successfully!")
        except Exception as e:
            logger.error(f"Failed to initialize MediaPipe recognizer: {e}")
            self.recognizer = None

    def close(self):
        """Release the MediaPipe recognizer explicitly (its destructor fails during interpreter shutdown)."""
        if self.recognizer is not None:
            self.recognizer.close()
            self.recognizer = None

    def count_raised_fingers(self, lm) -> int:
        """
        Number of extended fingers (0-5) on one hand, scale-invariant: a finger is raised when
        its tip is clearly farther from the wrist than its PIP joint; the thumb is raised when
        its tip is far from the index knuckle relative to palm size.
        """
        palm = self.dist(lm[WRIST], lm[MIDDLE_MCP]) + 1e-6
        count = 0
        for tip, pip in ((INDEX_TIP, INDEX_PIP), (MIDDLE_TIP, MIDDLE_PIP), (RING_TIP, RING_PIP), (PINKY_TIP, PINKY_PIP)):
            if self.dist(lm[tip], lm[WRIST]) > 1.16 * self.dist(lm[pip], lm[WRIST]):
                count += 1
        if self.dist(lm[THUMB_TIP], lm[INDEX_MCP]) / palm > 0.55 and                 self.dist(lm[THUMB_TIP], lm[PINKY_MCP]) > self.dist(lm[THUMB_IP], lm[PINKY_MCP]):
            count += 1
        return count

    def dist(self, p1, p2):
        return math.hypot(p1.x - p2.x, p1.y - p2.y)

    def classify_geometric_landmarks(self, lm, handedness="Right"):
        """
        Classifies hand sign based on 21 3D hand landmarks.
        Uses rotation-invariant finger-to-palm ratios for instant, tilt-independent recognition.
        Recognizes: Thumbs Up, Thumbs Down, V (Peace), W (WhatsApp), B (Open Palm),
        A (Fist), D (Point/One), L, Y (Call/Phone), I, F (OK), I Love You, C.
        """
        d_wrist_index_tip = self.dist(lm[INDEX_TIP], lm[WRIST])
        d_wrist_index_pip = self.dist(lm[INDEX_PIP], lm[WRIST])
        index_ratio = d_wrist_index_tip / (d_wrist_index_pip + 1e-6)
        index_extended = (index_ratio > 1.16) or (lm[INDEX_TIP].y < lm[INDEX_PIP].y and d_wrist_index_tip > d_wrist_index_pip)

        d_wrist_mid_tip = self.dist(lm[MIDDLE_TIP], lm[WRIST])
        d_wrist_mid_pip = self.dist(lm[MIDDLE_PIP], lm[WRIST])
        mid_ratio = d_wrist_mid_tip / (d_wrist_mid_pip + 1e-6)
        middle_extended = (mid_ratio > 1.16) or (lm[MIDDLE_TIP].y < lm[MIDDLE_PIP].y and d_wrist_mid_tip > d_wrist_mid_pip)

        d_wrist_ring_tip = self.dist(lm[RING_TIP], lm[WRIST])
        d_wrist_ring_pip = self.dist(lm[RING_PIP], lm[WRIST])
        ring_ratio = d_wrist_ring_tip / (d_wrist_ring_pip + 1e-6)
        ring_extended = (ring_ratio > 1.16) or (lm[RING_TIP].y < lm[RING_PIP].y and d_wrist_ring_tip > d_wrist_ring_pip)

        d_wrist_pinky_tip = self.dist(lm[PINKY_TIP], lm[WRIST])
        d_wrist_pinky_pip = self.dist(lm[PINKY_PIP], lm[WRIST])
        pinky_ratio = d_wrist_pinky_tip / (d_wrist_pinky_pip + 1e-6)
        pinky_extended = (pinky_ratio > 1.16) or (lm[PINKY_TIP].y < lm[PINKY_PIP].y and d_wrist_pinky_tip > d_wrist_pinky_pip)

        # Thumb extension
        thumb_tip = lm[THUMB_TIP]
        thumb_ip = lm[THUMB_IP]
        thumb_mcp = lm[THUMB_MCP]
        wrist = lm[WRIST]

        thumb_up = (thumb_tip.y < thumb_mcp.y) and (thumb_tip.y < lm[INDEX_MCP].y)
        thumb_down = (thumb_tip.y > thumb_mcp.y) and (thumb_tip.y > wrist.y)
        thumb_out = self.dist(thumb_tip, lm[INDEX_MCP]) > 0.11

        d_index_middle = self.dist(lm[INDEX_TIP], lm[MIDDLE_TIP])
        d_thumb_index = self.dist(thumb_tip, lm[INDEX_TIP])

        all_four_curled = (not index_extended) and (not middle_extended) and (not ring_extended) and (not pinky_extended)
        all_four_up = index_extended and middle_extended and ring_extended and pinky_extended

        # 1. THUMBS UP
        if thumb_up and all_four_curled:
            return "Thumbs Up", 0.98

        # 2. THUMBS DOWN
        if thumb_down and all_four_curled:
            return "Thumbs Down", 0.98

        # 3. SIGN V (VICTORY / PEACE / TWO)
        if index_extended and middle_extended and (not ring_extended) and (not pinky_extended):
            if d_index_middle > 0.035:
                return "Sign 'V' (Peace / Two)", 0.97
            else:
                return "Sign 'U'", 0.92

        # 4. SIGN W (WHATSAPP / THREE FINGERS)
        if index_extended and middle_extended and ring_extended and (not pinky_extended):
            return "Sign 'W' (WhatsApp)", 0.98

        # 5. SIGN D (POINTING UP / ONE)
        if index_extended and (not middle_extended) and (not ring_extended) and (not pinky_extended):
            if thumb_out and abs(thumb_tip.x - lm[INDEX_TIP].x) > 0.10:
                return "Sign 'L'", 0.95
            return "Sign 'D' (One / Point)", 0.95

        # 6. SIGN L
        if index_extended and thumb_out and (not middle_extended) and (not ring_extended) and (not pinky_extended):
            return "Sign 'L'", 0.96

        # 7. SIGN Y (CALL / PHONE / SHAKA)
        if thumb_out and pinky_extended and (not index_extended) and (not middle_extended) and (not ring_extended):
            return "Sign 'Y' (Call / Phone)", 0.96

        # 8. I LOVE YOU / ROCK
        if thumb_out and index_extended and pinky_extended and (not middle_extended) and (not ring_extended):
            return "Sign 'I Love You'", 0.97
        if index_extended and pinky_extended and (not middle_extended) and (not ring_extended):
            return "Rock / Love", 0.94

        # 9. SIGN I (PINKY ONLY)
        if pinky_extended and (not index_extended) and (not middle_extended) and (not ring_extended) and (not thumb_out):
            return "Sign 'I'", 0.93

        # 10. SIGN F / OK SIGN
        if d_thumb_index < 0.055 and middle_extended and ring_extended and pinky_extended:
            return "Sign 'F' (OK Sign)", 0.96

        # 11. SIGN B (OPEN PALM / HELLO)
        if all_four_up:
            return "Sign 'B' (Open Hand / Hello)", 0.96

        # 12. SIGN A (CLOSED FIST)
        if all_four_curled and not thumb_up and not thumb_down:
            return "Sign 'A' (Fist)", 0.92

        # 13. SIGN C (Curved fingers forming 'C' arc)
        if (not all_four_curled) and (not all_four_up):
            if 0.07 < d_thumb_index < 0.22 and (lm[INDEX_TIP].y > lm[INDEX_PIP].y or lm[INDEX_DIP].y > lm[INDEX_PIP].y):
                return "Sign 'C'", 0.88

        return None, 0.0

    def detect_skin_hand_roi(self, frame_bgr):
        """
        Fast skin-color contour detection as a robust fallback.
        Suppresses upper-center face region to avoid mistaking the user's face for a hand.
        """
        H, W, _ = frame_bgr.shape
        ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
        lower_skin = np.array([0, 133, 77], dtype=np.uint8)
        upper_skin = np.array([255, 173, 127], dtype=np.uint8)
        mask = cv2.inRange(ycrcb, lower_skin, upper_skin)

        # Morphological clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        large_contours = [c for c in contours if cv2.contourArea(c) > 4000]

        if not large_contours:
            return None

        # Prioritize contours outside the top-center face zone
        hand_candidates = []
        for c in large_contours:
            M = cv2.moments(c)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                is_face = (cy < 0.38 * H) and (0.28 * W < cx < 0.72 * W)
                if not is_face:
                    hand_candidates.append(c)

        selected_contour = max(hand_candidates, key=cv2.contourArea) if hand_candidates else max(large_contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(selected_contour)

        # Expand padding
        pad = 20
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(W, x + w + pad)
        y2 = min(H, y + h + pad)

        return (x1, y1, x2, y2)

    def process_frame(self, frame_bgr):
        """
        Process a single live webcam frame:
        1. Runs MediaPipe hand detection (high-sensitivity 0.15).
        2. If MediaPipe succeeds -> 3D landmarks + geometric recognition.
        3. If MediaPipe misses -> skin contour + CNN fallback.
        Returns:
            processed_frame, prediction_label, confidence, top3_list, hand_bbox, distance_hint
        """
        h, w, _ = frame_bgr.shape
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        # Unannotated copy for CNN crops: the overlays below are drawn onto frame_bgr in place
        clean_bgr = frame_bgr.copy()

        pred_label = "Position hand 1-2 ft away"
        pred_conf = 0.0
        top3_list = []
        hand_bbox = None

        media_pipe_found = False

        if self.recognizer is not None:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            result = self.recognizer.recognize(mp_image)

            if result.hand_landmarks:
                media_pipe_found = True
                hands = result.hand_landmarks
                landmarks = hands[0]

                # Bounding box around ALL detected hands (two-handed ISL letters need both in the crop)
                xs = [lm.x * w for hand in hands for lm in hand]
                ys = [lm.y * h for hand in hands for lm in hand]
                xmin = max(0, int(min(xs)) - 25)
                xmax = min(w, int(max(xs)) + 25)
                ymin = max(0, int(min(ys)) - 25)
                ymax = min(h, int(max(ys)) + 25)
                hand_bbox = (xmin, ymin, xmax, ymax)
                self.last_bbox = hand_bbox
                self.last_bbox_age = 0

                # 1. Draw Skeleton Bones & Landmarks for every hand
                for hand in hands:
                    for start_idx, end_idx in HAND_CONNECTIONS:
                        pt1 = (int(hand[start_idx].x * w), int(hand[start_idx].y * h))
                        pt2 = (int(hand[end_idx].x * w), int(hand[end_idx].y * h))
                        cv2.line(frame_bgr, pt1, pt2, (0, 255, 128), 2)
                    for idx, lm in enumerate(hand):
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        pt_color = (255, 255, 0) if idx in (4, 8, 12, 16, 20) else (0, 220, 255)
                        cv2.circle(frame_bgr, (cx, cy), 5, pt_color, -1)
                        cv2.circle(frame_bgr, (cx, cy), 6, (0, 0, 0), 1)

                # 2. Draw Dynamic Hand Tracking Box
                cv2.rectangle(frame_bgr, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                cv2.putText(frame_bgr, f"3D SKELETON TRACKED ({len(hands)} hand{'s' if len(hands) > 1 else ''})",
                            (xmin, max(25, ymin - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 0), 2)

                # 3. Geometric Classification: the one-hand rules only apply to a single hand;
                #    two visible hands mean a two-handed ISL letter, left to the CNN
                if len(hands) == 1:
                    geom_label, geom_conf = self.classify_geometric_landmarks(landmarks)
                else:
                    geom_label, geom_conf = None, 0.0

                if self.numbers_mode:
                    if len(hands) == 1:
                        n = self.count_raised_fingers(landmarks)
                        pred_label, pred_conf = f"Number '{n}'", 0.95
                    else:
                        pred_label, pred_conf = "Use ONE hand for numbers", 0.0
                    return self._finish(frame_bgr, pred_label, pred_conf, [], hand_bbox, h)

                # 4. Crop hand for CNN classification
                cropped_hand = clean_bgr[ymin:ymax, xmin:xmax]
                cnn_label = None
                cnn_conf = 0.0

                if cropped_hand.size > 0 and self.static_cnn.is_loaded:
                    try:
                        c_lbl, c_conf, c_top3 = self.static_cnn.predict(cropped_hand)
                        cnn_label = c_lbl
                        cnn_conf = c_conf
                        top3_list = c_top3
                    except (RuntimeError, ValueError, cv2.error) as e:
                        logger.error(f"CNN prediction error: {e}")

                # 5. Smart Fusion
                if geom_label is not None and geom_conf >= 0.85:
                    pred_label = geom_label
                    pred_conf = geom_conf
                    if not top3_list and cnn_label:
                        top3_list = [(cnn_label, cnn_conf)]
                elif cnn_label is not None and cnn_conf >= 0.35:
                    pred_label = f"Sign '{cnn_label}'"
                    pred_conf = cnn_conf
                elif geom_label is not None:
                    pred_label = geom_label
                    pred_conf = geom_conf
                else:
                    pred_label = f"Sign '{cnn_label}'" if cnn_label else "Analyzing Hand..."
                    pred_conf = cnn_conf

        # -------------------------------------------------------------
        # FALLBACK: If MediaPipe missed (hand too close / palm clipped)
        # -------------------------------------------------------------
        if not media_pipe_found:
            fallback_box = self.detect_skin_hand_roi(frame_bgr)
            if fallback_box is None and self.last_bbox is not None and self.last_bbox_age < 5:
                # Use cached bbox
                fallback_box = self.last_bbox
                self.last_bbox_age += 1

            if fallback_box is not None:
                xmin, ymin, xmax, ymax = fallback_box
                hand_bbox = fallback_box

                cv2.rectangle(frame_bgr, (xmin, ymin), (xmax, ymax), (0, 220, 255), 2)
                cv2.putText(frame_bgr, "HAND TRACKED (Move back for 3D skeleton)", (xmin, max(25, ymin - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1)

                cropped_hand = clean_bgr[ymin:ymax, xmin:xmax]
                if cropped_hand.size > 0 and self.static_cnn.is_loaded:
                    try:
                        c_lbl, c_conf, c_top3 = self.static_cnn.predict(cropped_hand)
                        top3_list = c_top3
                        # Report the CNN's own softmax confidence; the WhatsApp trigger applies its own threshold
                        if c_lbl == 'W' and c_conf >= 0.30:
                            pred_label = "Sign 'W' (WhatsApp)"
                            pred_conf = c_conf
                        elif c_conf >= 0.30:
                            pred_label = f"Sign '{c_lbl}'"
                            pred_conf = c_conf
                        else:
                            pred_label = "Analyzing Sign..."
                            pred_conf = c_conf
                    except (RuntimeError, ValueError, cv2.error) as e:
                        logger.error(f"Fallback CNN error: {e}")
            else:
                pred_label = "Keep palm & fingers in view"
                pred_conf = 0.0

        return self._finish(frame_bgr, pred_label, pred_conf, top3_list, hand_bbox, h, media_pipe_found)

    def _finish(self, frame_bgr, pred_label, pred_conf, top3_list, hand_bbox, h, media_pipe_found=True):
        """Distance coaching + temporal smoothing shared by letter and number modes."""
        # Distance coaching calculation
        distance_hint = "SEARCHING"
        if hand_bbox is not None:
            _, y1, _, y2 = hand_bbox
            box_h = y2 - y1
            if box_h > 0.68 * h:
                distance_hint = "TOO_CLOSE"
            elif box_h < 0.18 * h:
                distance_hint = "TOO_FAR"
            else:
                distance_hint = "OPTIMAL"

        # Ultra-Fast Zero-Lag Smoothing
        if pred_conf >= 0.75:
            # Immediate response without latency!
            self.history.append(pred_label)
        elif media_pipe_found or hand_bbox is not None:
            self.history.append(pred_label)
            if len(self.history) >= 2:
                pred_label = smooth_predictions(self.history)
        else:
            # When hand is absent, clear history so the next sign reacts instantly!
            self.history.clear()

        return frame_bgr, pred_label, pred_conf, top3_list, hand_bbox, distance_hint
