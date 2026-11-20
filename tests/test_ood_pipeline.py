import os
import sys
import tempfile
import unittest
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.datasets import generate_synthetic_cache, load_feature_cache
from src.methods import DeepKNNDetector, EnergyDetector, MSPDetector
from src.utils.metrics import compute_ood_metrics


class TestOODPipeline(unittest.TestCase):
    def test_detectors_and_metrics_on_synthetic_cache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            generate_synthetic_cache(cache_dir=tmpdir, seeds=(0,), feature_dim=64, num_classes=10)
            id_train = load_feature_cache(tmpdir, 0, "cifar10_train")
            id_test = load_feature_cache(tmpdir, 0, "cifar10_test")
            ood_svhn = load_feature_cache(tmpdir, 0, "svhn")

            msp = MSPDetector().fit(id_train)
            energy = EnergyDetector(temperature=1.0).fit(id_train)
            knn = DeepKNNDetector(k=20, normalize_l2=True, sampling_ratio=1.0).fit(id_train)

            for det in (msp, energy, knn):
                id_scores = det.score(id_test)
                ood_scores = det.score(ood_svhn)
                self.assertEqual(id_scores.shape, (1000,))
                self.assertEqual(ood_scores.shape, (1000,))

                metrics = compute_ood_metrics(id_scores, ood_scores)
                self.assertGreaterEqual(metrics["auroc"], 0.0)
                self.assertLessEqual(metrics["auroc"], 100.0)
                self.assertGreaterEqual(metrics["fpr95"], 0.0)
                self.assertLessEqual(metrics["fpr95"], 100.0)

            # Kiểm tra Deep kNN với L2-norm phải đạt AUROC rất cao (>90%) trên tập Far-OOD giả lập
            knn_metrics = compute_ood_metrics(knn.score(id_test), knn.score(ood_svhn))
            self.assertGreater(knn_metrics["auroc"], 90.0)


if __name__ == "__main__":
    unittest.main()
