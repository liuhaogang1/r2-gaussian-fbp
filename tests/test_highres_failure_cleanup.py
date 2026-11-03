import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock
import numpy as np
import torch
import export_highres as highres


class ExportFailureTests(unittest.TestCase):
    def test_incomplete_export_removed_and_existing_result_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'volume.npy'
            calls = [dict(vol=torch.ones((8,8,8))), RuntimeError('intentional query failure')]
            with mock.patch.object(highres, 'query', side_effect=calls):
                with self.assertRaisesRegex(RuntimeError, 'intentional'):
                    highres.export_volume(None, SimpleNamespace(), [0,0,0], [2,2,2],
                                          (16,8,8), (8,8,8), str(path), 'float32')
            self.assertFalse(path.exists())
            np.save(path, np.array([12.],np.float32))
            with self.assertRaises(FileExistsError):
                highres.export_volume(None, SimpleNamespace(), [0,0,0], [2,2,2],
                                      (16,8,8), (8,8,8), str(path), 'float32')
            np.testing.assert_array_equal(np.load(path), [12.])
