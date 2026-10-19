"""Synthetic invariants and CLI checks; no private recordings are required."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import bearing_fault_cnn as cnn


class PipelineTests(unittest.TestCase):
    def test_strings_header_and_ignored_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'signal.csv'
            path.write_text('Accelerometer,Other\n' + '\n'.join(
                f'"{i}",not-a-number' for i in range(420000)) + '\n')
            signal, count = cnn.read_signal(path, 420000)
            self.assertEqual(count, 420000)
            self.assertEqual(len(signal) // cnn.SIGNAL_LENGTH, 410)
            self.assertEqual(len(signal) % cnn.SIGNAL_LENGTH, 160)
            self.assertEqual(signal[-1], 419999)
            path.write_text('1,ignored\n' * 1024)
            signal, count = cnn.read_signal(path, 1024)
            self.assertEqual(count, 1024)
            self.assertEqual(signal[0], 1)
            for bad in ('', 'oops', 'nan', 'inf'):
                path.write_text('Signal,Other\n1,x\n' + bad + ',x\n')
                with self.subTest(bad=bad), self.assertRaises(ValueError):
                    cnn.read_signal(path, 1024)

    def test_architecture_and_learning(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        model = cnn.BearingCNN()
        self.assertEqual(sum(isinstance(x, torch.nn.Conv1d) for x in model.modules()), 2)
        signal = torch.randn(32, 1, 1024)
        before = model.network[0].weight.detach().clone()
        logits = model(signal)
        self.assertEqual(tuple(logits.shape), (32, 3))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss = torch.nn.CrossEntropyLoss()(logits, torch.arange(32) % 3)
        loss.backward()
        self.assertTrue(torch.isfinite(model.network[0].weight.grad).all())
        optimizer.step()
        self.assertFalse(torch.equal(before, model.network[0].weight))

    def test_metrics_and_recording_separation(self):
        files = [name for group in cnn.FILE_SPLITS.values() for name in group]
        self.assertEqual(len(files), len(set(files)))
        matrix = cnn.confusion_matrix(np.array([0, 0, 1, 2]), np.array([0, 1, 1, 0]))
        self.assertEqual(matrix.tolist(), [[1, 1, 0], [0, 1, 0], [1, 0, 0]])
        self.assertEqual(cnn.binary_matrix(matrix, 0).tolist(), [[1, 1], [1, 1]])

    def test_end_to_end_synthetic(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for split, names in cnn.FILE_SPLITS.items():
                for label, name in enumerate(names):
                    values = np.sin(np.arange(2048) * (label + 1) * 0.03) + label
                    (root / name).write_text('Signal,Unused\n' + '\n'.join(
                        f'{value},ignored' for value in values) + '\n')
            subprocess.run([sys.executable, str(Path(cnn.__file__)), '--data-dir', str(root),
                            '--output-dir', str(root / 'output'), '--epochs', '1',
                            '--max-samples', '2048', '--threads', '2'],
                           check=True, capture_output=True, text=True, timeout=90)
            result = json.loads((root / 'output/results.json').read_text())
            self.assertEqual(sum(map(sum, result['confusion_matrix'])), 6)
            self.assertEqual(result['best_epoch'], 1)
            for artifact in ('bearing_cnn.pt', 'test_predictions.csv', 'confusion_matrices.png'):
                self.assertGreater((root / 'output' / artifact).stat().st_size, 0)


if __name__ == '__main__':
    unittest.main()
