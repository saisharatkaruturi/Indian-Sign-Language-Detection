import unittest
from src.camera.camera import CameraManager

class TestCameraManager(unittest.TestCase):

    def test_camera_initialization(self):
        cam = CameraManager(camera_index=0, width=640, height=480, target_fps=30)
        self.assertEqual(cam.camera_index, 0)
        self.assertEqual(cam.width, 640)
        self.assertEqual(cam.height, 480)
        self.assertEqual(cam.target_fps, 30)
        self.assertFalse(cam.is_running)

    def test_camera_release_safety(self):
        cam = CameraManager(camera_index=999) # Non-existent camera index
        cam.release()
        self.assertFalse(cam.is_running)

if __name__ == "__main__":
    unittest.main()
