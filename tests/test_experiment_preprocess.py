"""Experiment-wrapper preprocessing and content cache regressions."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import numpy as np
import tifffile
from scripts import prepare_sio2_experiments as experiment
from scripts.fbp_preprocess import process_projection


class ExperimentPreprocessTests(unittest.TestCase):
    def options(self, root):
        return SimpleNamespace(input_dir=root, output_dir=root/'out', config=root/'scan.txt',
            projection_glob='Tomo_*.tiff',flat_glob='Flat_*.tiff',dark_value=10.,flat_eps=1e-6,
            log_eps=1e-6,shift_v=0,pixel_subsample=3,resize_order=1,projection_scale=.5,reprocess=False)

    def test_corrected_sampling_matches_shared_physical_resampling(self):
        y,x=np.indices((51,74))
        integral=(.1+.002*y+.001*x).astype(np.float32)
        image=10+90*np.exp(-integral)
        args=self.options(Path('.'))
        actual=experiment.process_corrected(image,np.full_like(image,100),args)
        settings=SimpleNamespace(**vars(args))
        settings.input_type,settings.clip_percentile,settings.flat_field='line_integral',100,False
        expected=process_projection(integral,settings)
        np.testing.assert_allclose(actual,expected,rtol=2e-6,atol=2e-7)

    def test_cache_reuses_only_same_inputs_parameters_and_output_content(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root=Path(directory)
            args=self.options(root)
            args.config.write_text('NumberImages=3\nAngleInterval=90\n')
            tifffile.imwrite(root/'Flat_0.tiff',np.full((30,42),100,np.float32))
            for i in range(3):
                tifffile.imwrite(root/f'Tomo_{i}.tiff',np.full((30,42),60+i,np.float32))
            original=experiment.process_corrected
            with mock.patch.object(experiment,'process_corrected',wraps=original) as process:
                first,angles=experiment.load_or_make_corrected(args)
                self.assertEqual((len(first),len(angles)),(3,2))
                second,_=experiment.load_or_make_corrected(args)
                np.testing.assert_array_equal(first,second)
                self.assertEqual(process.call_count,3)
                tifffile.imwrite(root/'Tomo_1.tiff',np.full((30,42),50,np.float32))
                changed,_=experiment.load_or_make_corrected(args)
                self.assertEqual(process.call_count,6)
                self.assertFalse(np.array_equal(first,changed))
                np.save(args.output_dir/'corrected_all.npy',changed+1)
                restored,_=experiment.load_or_make_corrected(args)
                np.testing.assert_array_equal(restored,changed)
                self.assertEqual(process.call_count,9)
                args.projection_scale=1
                scaled,_=experiment.load_or_make_corrected(args)
                np.testing.assert_allclose(scaled,2*changed)
                self.assertEqual(process.call_count,12)


if __name__=='__main__':
    unittest.main()
