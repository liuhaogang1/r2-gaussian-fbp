"""Server regressions for TIGRE API variants and full training checkpoints."""
import contextlib
import io
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch

from r2_gaussian.utils.artifact_utils import load_training_checkpoint
from scripts import probe_tigre_parallel_offsets as probe


class ServerCompatibilityTests(unittest.TestCase):
    def test_gaussian_and_pose_checkpoints_restore_numpy_and_optimizer_state(self):
        # Match the NumPy scalar/array and optimizer state in production capture.
        parameter=torch.nn.Parameter(torch.tensor([1.,2.]))
        optimizer=torch.optim.Adam([parameter],lr=.01)
        parameter.square().sum().backward()
        optimizer.step()
        model_state=(parameter.detach().clone(),np.float64(.25),np.array([.001,.5]),optimizer.state_dict())
        binding=dict(binding_version=np.asarray(1,np.int32),model_iteration=np.asarray(6,np.int64),
                     image_ids=np.array(['0000','0001']),base_angles=np.array([0.,.5]),
                     dataset_identity=np.asarray('local-fixture'))
        pose_state=dict(raw_angle=torch.tensor([.01,-.01]))
        for checkpoint in ((model_state,6),(model_state,6,pose_state,optimizer.state_dict(),binding)):
            with self.subTest(entries=len(checkpoint)):
                stream=io.BytesIO()
                torch.save(checkpoint,stream)
                stream.seek(0)
                restored=load_training_checkpoint(stream)
                self.assertEqual(len(restored),len(checkpoint))
                self.assertEqual(restored[1],6)
                torch.testing.assert_close(restored[0][0],model_state[0])
                self.assertIsInstance(restored[0][1],np.float64)
                np.testing.assert_array_equal(restored[0][2],model_state[2])
                fresh_parameter=torch.nn.Parameter(restored[0][0].clone())
                fresh_optimizer=torch.optim.Adam([fresh_parameter],lr=.9)
                fresh_optimizer.load_state_dict(restored[0][3])
                self.assertEqual(fresh_optimizer.param_groups[0]['lr'],.01)
                original=optimizer.state[parameter]
                for key in ('step','exp_avg','exp_avg_sq'):
                    torch.testing.assert_close(fresh_optimizer.state[fresh_parameter][key],original[key])
                if len(checkpoint)==5:
                    np.testing.assert_array_equal(restored[4]['image_ids'],binding['image_ids'])
                    np.testing.assert_array_equal(restored[4]['base_angles'],binding['base_angles'])
                    torch.testing.assert_close(restored[2]['raw_angle'],pose_state['raw_angle'])

    def run_probe(self,projector):
        geometry=SimpleNamespace(offDetector=np.zeros(2))
        parser=SimpleNamespace(parse_args=lambda args:args)
        prepare=SimpleNamespace(build_parser=lambda:parser,build_geometry=lambda args,shape:geometry)
        ax_module=SimpleNamespace(Ax=projector)
        with mock.patch.dict('sys.modules',{'prepare_fbp_tiff':prepare,'tigre.utilities.Ax':ax_module}), \
             contextlib.redirect_stdout(io.StringIO()) as output:
            probe.main()
        return output.getvalue()

    def test_legacy_ax_runs_default_without_unsupported_keyword(self):
        methods=[]
        def legacy(img,geo,angles,krylov='ray-voxel'):
            methods.append(krylov)
            return np.ones((1,96,112),np.float32)
        output=self.run_probe(legacy)
        self.assertEqual(methods,['ray-voxel']*15)
        self.assertIn('selector comparison is unavailable',output)
        self.assertEqual(output.count('method=installed_default'),15)

    def test_modern_ax_runs_both_declared_projection_methods(self):
        methods=[]
        def modern(img,geo,angles,projection_type='Siddon'):
            methods.append(projection_type)
            return np.ones((1,96,112),np.float32)
        output=self.run_probe(modern)
        self.assertEqual(methods,['Siddon','interpolated']*15)
        self.assertEqual(output.count('centroid_uv='),30)

    def test_projector_errors_are_not_swallowed_as_api_fallback(self):
        def broken(img,geo,angles,projection_type='Siddon'):
            raise TypeError('CUDA projection failed internally')
        with self.assertRaisesRegex(TypeError,'failed internally'):
            self.run_probe(broken)


if __name__=='__main__':
    unittest.main()
