import unittest
from pathlib import Path
import tempfile
from unittest.mock import patch

import nibabel as nib
import numpy as np

from utils.vertebrae_refinement import postprocessing_vertebrae_refinement
from utils.vertebrae_refinement_core import LABEL_NAMES, process_array


class VertebraeRefinementTests(unittest.TestCase):
    def setUp(self):
        self.source = np.zeros((12, 12, 36), dtype=np.uint8)
        self.source[3:8, 3:8, 3:8] = 1
        self.source[3:8, 3:8, 14:19] = 2
        self.affine = np.diag([-1.2, 1.4, 2.0, 1.0])
        self.image = nib.Nifti1Image(self.source, self.affine)
        self.masks = {
            name: (self.source == label).astype(np.uint8)
            for label, name in LABEL_NAMES.items()
        }

    def test_matches_core_on_partial_scan(self):
        expected, _ = process_array(self.source, self.affine)
        result = postprocessing_vertebrae_refinement("partial", self.masks, self.image)
        actual = np.zeros_like(self.source)
        for label, name in LABEL_NAMES.items():
            actual[result[name] > 0] = label
        np.testing.assert_array_equal(actual, expected)

    def test_mapping_affine_and_other_organs(self):
        organ = np.ones_like(self.source)
        self.masks["liver"] = organ
        report = {"changed_voxels": 0, "input_foreground_voxels": 250,
                  "output_foreground_voxels": 250}
        with patch("utils.vertebrae_refinement.process_array",
                   return_value=(self.source, report)) as call:
            result = postprocessing_vertebrae_refinement("case", self.masks, self.image)
        np.testing.assert_array_equal(call.call_args.args[0], self.source)
        np.testing.assert_array_equal(call.call_args.args[1], self.affine)
        self.assertIs(result["liver"], organ)
        self.assertEqual(set(result), set(self.masks))

    def test_empty_input(self):
        masks = {"vertebrae_L5": np.zeros_like(self.source)}
        self.assertIs(postprocessing_vertebrae_refinement("empty", masks, self.image), masks)

    def test_shape_mismatch(self):
        with self.assertRaises(ValueError):
            postprocessing_vertebrae_refinement(
                "bad", {"vertebrae_L5": np.zeros((2, 2, 2))}, self.image)

    def test_overlapping_masks(self):
        self.masks["vertebrae_L4"] = self.masks["vertebrae_L5"].copy()
        with self.assertRaises(ValueError):
            postprocessing_vertebrae_refinement("overlap", self.masks, self.image)

    def test_nonfinite_mask(self):
        self.masks["vertebrae_L5"] = np.full(self.source.shape, np.nan)
        with self.assertRaises(ValueError):
            postprocessing_vertebrae_refinement("invalid", self.masks, self.image)

    def test_empty_output_replaces_copied_input(self):
        from utils.utils import save_and_combine_segmentations
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / "segmentations"
            folder.mkdir()
            nib.save(self.image, folder / "vertebrae_L5.nii.gz")
            save_and_combine_segmentations(
                {"vertebrae_L5": np.zeros_like(self.source)},
                {26: "vertebrae_L5"}, self.image, directory, True)
            mask = np.asarray(nib.load(folder / "vertebrae_L5.nii.gz").dataobj)
            combined = np.asarray(nib.load(Path(directory) / "combined_labels.nii.gz").dataobj)
            self.assertFalse(mask.any())
            self.assertFalse(combined.any())


if __name__ == "__main__":
    unittest.main()
