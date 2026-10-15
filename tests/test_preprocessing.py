import unittest
import numpy as np
from src.preprocessing.image_preprocessing import preprocess_static_image, validate_image
from src.preprocessing.video_preprocessing import preprocess_video_sequence, sample_or_pad_sequence
from src.preprocessing.sequence_builder import RollingSequenceBuffer

class TestPreprocessingContracts(unittest.TestCase):

    def setUp(self):
        self.synthetic_frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        self.synthetic_frames = [
            np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
            for _ in range(15)
        ]

    def test_static_preprocessing_shape(self):
        tensor = preprocess_static_image(self.synthetic_frame, target_size=(224, 224), normalize=True)
        self.assertEqual(tensor.shape, (224, 224, 3))
        self.assertEqual(tensor.dtype, np.float32)
        self.assertGreaterEqual(tensor.min(), 0.0)
        self.assertLessEqual(tensor.max(), 1.0)

    def test_sequence_sampling_and_padding(self):
        sampled = sample_or_pad_sequence(self.synthetic_frames, target_len=30)
        self.assertEqual(len(sampled), 30)

    def test_video_sequence_tensor(self):
        seq_tensor = preprocess_video_sequence(self.synthetic_frames, target_len=30, target_size=(224, 224), normalize=True)
        self.assertEqual(seq_tensor.shape, (30, 224, 224, 3))
        self.assertEqual(seq_tensor.dtype, np.float32)

    def test_rolling_sequence_buffer(self):
        buf = RollingSequenceBuffer(sequence_length=30, target_size=(224, 224), normalize=True)
        self.assertFalse(buf.is_full())
        for f in self.synthetic_frames:
            buf.append(f)
        self.assertEqual(buf.count(), 15)
        seq_tensor = buf.get_sequence_tensor()
        self.assertEqual(seq_tensor.shape, (30, 224, 224, 3))

if __name__ == "__main__":
    unittest.main()
