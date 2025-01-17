import unittest
import numpy as np
from starter.ml.model import train_model, compute_model_metrics, inference


class TestModel(unittest.TestCase):
    def setUp(self):
        self.X_train = np.array([[1, 2], [3, 4], [5, 6]])
        self.y_train = np.array([0, 1, 0])
        self.model = train_model(self.X_train, self.y_train)
        self.X_test = np.array([[1, 2], [3, 4]])
        self.y_test = np.array([0, 1])

    def test_train_model(self):
        self.assertIsNotNone(self.model)

    def test_compute_model_metrics(self):
        preds = inference(self.model, self.X_test)
        precision, recall, fbeta = compute_model_metrics(self.y_test, preds)
        self.assertGreaterEqual(precision, 0)
        self.assertGreaterEqual(recall, 0)
        self.assertGreaterEqual(fbeta, 0)

    def test_inference(self):
        preds = inference(self.model, self.X_test)
        self.assertEqual(len(preds), len(self.X_test))


if __name__ == "__main__":
    unittest.main()
