import unittest
import tensorflow as tf
from src.models.cnn_model import build_cnn_feature_extractor
from src.models.static_model import build_static_model
from src.models.dynamic_model import build_dynamic_model

class TestModelArchitectures(unittest.TestCase):

    def test_cnn_feature_extractor_shape(self):
        extractor = build_cnn_feature_extractor(input_shape=(224, 224, 3), backbone="Custom", embedding_dim=256)
        dummy_input = tf.zeros((2, 224, 224, 3))
        output = extractor(dummy_input)
        self.assertEqual(output.shape, (2, 256))

    def test_static_model_shape(self):
        static_model = build_static_model(input_shape=(224, 224, 3), num_classes=26, backbone="Custom")
        dummy_input = tf.zeros((2, 224, 224, 3))
        output = static_model(dummy_input)
        self.assertEqual(output.shape, (2, 26))

    def test_dynamic_gru_model_shape(self):
        dynamic_gru = build_dynamic_model(temporal_model_type="GRU", sequence_length=10, input_shape=(224, 224, 3), num_classes=15, backbone="Custom")
        dummy_input = tf.zeros((2, 10, 224, 224, 3))
        output = dynamic_gru(dummy_input)
        self.assertEqual(output.shape, (2, 15))

    def test_dynamic_lstm_model_shape(self):
        dynamic_lstm = build_dynamic_model(temporal_model_type="LSTM", sequence_length=10, input_shape=(224, 224, 3), num_classes=15, backbone="Custom")
        dummy_input = tf.zeros((2, 10, 224, 224, 3))
        output = dynamic_lstm(dummy_input)
        self.assertEqual(output.shape, (2, 15))

if __name__ == "__main__":
    unittest.main()
