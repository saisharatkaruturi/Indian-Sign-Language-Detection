import os
import cv2
import time
from typing import Generator, Tuple, Optional
from src.utils.logger import get_logger
from src.utils.config import load_config

logger = get_logger("CameraManager")

class CameraManager:
    """Handles real-time OpenCV VideoCapture camera interactions."""

    def __init__(self, camera_index: int = 0, width: int = 640, height: int = 480, target_fps: int = 30):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.target_fps = target_fps
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_running = False

    def is_available(self) -> bool:
        """Check if camera hardware is accessible."""
        test_cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY)
        if not test_cap.isOpened():
            test_cap.release()
            return False
        ret, frame = test_cap.read()
        test_cap.release()
        return ret and frame is not None

    def start(self) -> bool:
        """Initialize and open OpenCV VideoCapture hardware."""
        logger.info(f"Opening camera hardware at index {self.camera_index}...")
        # Use CAP_DSHOW on Windows for fast backend initialization
        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY)
        if not self.cap.isOpened():
            logger.error(f"Failed to open camera at index {self.camera_index}. Hardware unavailable or in use.")
            return False

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)

        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)

        logger.info(f"Camera opened successfully! Resolution: {actual_w}x{actual_h}, FPS: {actual_fps}")
        self.is_running = True
        return True

    def read_frame(self) -> Tuple[bool, Optional[cv2.Mat]]:
        """Capture single frame from camera instance."""
        if not self.cap or not self.cap.isOpened():
            return False, None
        ret, frame = self.cap.read()
        return ret, frame

    def release(self):
        """Release camera hardware safely."""
        if self.cap and self.cap.isOpened():
            self.cap.release()
            logger.info("Camera hardware released.")
        self.is_running = False
        cv2.destroyAllWindows()

def run_camera_test():
    """Standalone test verifying camera hardware execution."""
    cfg = load_config()
    cam_cfg = cfg.get("camera", {})
    cam = CameraManager(
        camera_index=cam_cfg.get("index", 0),
        width=cam_cfg.get("width", 640),
        height=cam_cfg.get("height", 480),
        target_fps=cam_cfg.get("fps", 30)
    )

    print("\n==============================================")
    print("       TESTING LAPTOP CAMERA HARDWARE         ")
    print("==============================================")
    print("Press 'q' in the OpenCV window to exit.\n")

    if not cam.start():
        print("[ERROR] Could not open camera hardware! Check USB webcam / permissions.")
        return

    prev_time = time.time()
    fps = 0.0

    try:
        while True:
            ret, frame = cam.read_frame()
            if not ret or frame is None:
                print("[ERROR] Failed to grab frame from camera.")
                break

            curr_time = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / (curr_time - prev_time + 1e-6))
            prev_time = curr_time

            # Status Overlay
            cv2.putText(frame, f"Camera: Connected | FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "Press 'Q' to quit", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            cv2.imshow("ISL Real-Time Camera Test", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("Quit key 'q' pressed by user.")
                break
    finally:
        cam.release()
        print("Camera hardware test complete.")

if __name__ == "__main__":
    run_camera_test()
