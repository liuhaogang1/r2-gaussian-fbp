"""Check the production evaluation body independently of CLI dependencies."""
import ast
import contextlib
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import yaml
try:
    import torch
    import SimpleITK as sitk
except ImportError:
    torch = sitk = None

from r2_gaussian.utils.volume_export_utils import save_nifti_xyz, volume_geometry


@unittest.skipIf(torch is None or sitk is None, "Torch and SimpleITK required")
class EvaluationExportTests(unittest.TestCase):
    def test_evaluation_exports_original_lengths_and_xyz_preview_without_mutating_query(self):
        # Execute the actual function body, leaving the user's CLI and Python
        # compatibility choices untouched. Annotations are irrelevant here.
        path = Path(__file__).resolve().parents[1] / "test.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == "evaluate_volume")
        for argument in function.args.args:
            argument.annotation = None
        function.returns = None
        module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
        shape = (7, 9, 11)
        values = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) / np.prod(shape)
        tensor = torch.from_numpy(values)
        prediction = tensor * .8
        scene_scale = .25
        scanner = {"sVoxel": [1.4, 2.7, 4.4], "offOrigin": [.4, -.6, .9]}
        normalized = {key: (np.asarray(value) * scene_scale).tolist() for key, value in scanner.items()}
        normalized["nVoxel"] = list(shape)
        query = mock.Mock(return_value={"vol": prediction})
        write_images = mock.Mock()
        namespace = {"os": os, "osp": os.path, "np": np, "torch": torch, "yaml": yaml,
                     "query": query, "t2a": lambda value: value.detach().cpu().numpy(),
                     "metric_vol": lambda a, b, metric: (1., [1., 1., 1.]),
                     "multithread_write": write_images, "save_nifti_xyz": save_nifti_xyz}
        exec(compile(module, str(path), "exec"), namespace)
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            namespace["evaluate_volume"](folder, "reconstruction", normalized, None, None,
                                           tensor, scene_scale=scene_scale)
            self.assertEqual(query.call_args.args[1:4],
                             (normalized["offOrigin"], list(shape), normalized["sVoxel"]))
            spacing, origin = volume_geometry(shape, scanner)
            for suffix, expected in (("gt", values), ("pred", prediction.numpy())):
                image = sitk.ReadImage(str(Path(folder) / f"vol_{suffix}.nii.gz"))
                self.assertEqual(image.GetSize(), shape)
                np.testing.assert_allclose(image.GetSpacing(), spacing)
                np.testing.assert_allclose(image.GetOrigin(), origin, atol=1e-6)
                np.testing.assert_allclose(image.GetDirection(), np.eye(3).ravel())
                np.testing.assert_array_equal(sitk.GetArrayFromImage(image).transpose(2, 1, 0), expected)
                np.testing.assert_array_equal(np.load(Path(folder) / f"vol_{suffix}.npy"), expected)
            for call, expected in zip(write_images.call_args_list, (values, prediction.numpy())):
                for index, plane in enumerate(call.args[0]):
                    np.testing.assert_array_equal(plane.numpy()[0], expected[:, :, index])
                self.assertTrue(call.kwargs["display_xy"])
        np.testing.assert_array_equal(tensor.numpy(), values)


if __name__ == "__main__":
    unittest.main()
