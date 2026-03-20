"""Tests for the imagehandler.py module"""

import sys
import pytest
import numpy as np
from pathlib import Path
from astropy.io import fits

# Local import
sys.path.append(str(Path.cwd().parent))
from cleany import imagehandler


def test_DataEnsemble_empty():
    D = imagehandler.DataEnsemble()
    assert isinstance(D, imagehandler.DataEnsemble)
    assert D.extno is None
    assert D.data is None
    assert D.WCS is None
    assert D.header is None


def test_DataEnsemble_with_input(test_data):
    D = imagehandler.DataEnsemble(filename=[test_data / "test0.fits", test_data / "test1.fits"],
                                  EXPTIME="EXPOSURE", EXPUNIT="d",
                                  MJD_START="BJDREFI+TSTART+-2400000.5", FILTER="-Tess",
                                  xycuts=[0, 100, 0, 100])
    assert isinstance(D, imagehandler.DataEnsemble)
    assert D.extno == 0
    assert hasattr(D, "WCS")
    assert isinstance(D.data, np.ndarray)
    assert np.shape(D.data) == (2, 100, 100)


def test_save_fits_one(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    D = imagehandler.DataEnsemble()
    D.data = np.random.random([5, 5])
    D.one_header = fits.PrimaryHDU().header
    D.one_header['OBSERVER'] = "TEST"
    imagehandler.save_fits(DataEnsembleObject=D, filename="output")
    assert Path("output.fits").exists()
    assert Path("output.fits").stat().st_size > 0


@pytest.mark.parametrize(("n_files", "expected_out_names"),
                         [(3, ["output_0.fits", "output_1.fits", "output_2.fits"]),
                          (15, ["output_00.fits", "output_09.fits", "output_14.fits"])
                          ])
def test_save_fits_multiple(n_files, expected_out_names, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    D = imagehandler.DataEnsemble()
    D.data = np.random.random([n_files, 5, 5])
    one_header = fits.PrimaryHDU().header
    one_header['OBSERVER'] = "TEST"
    D.header = [one_header] * n_files
    imagehandler.save_fits(DataEnsembleObject=D, filename="output")
    for f in expected_out_names:
        assert Path(f).exists()
    assert Path(expected_out_names[-1]).stat().st_size > 0


def test_save_fits_fail(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    D = np.random.random(5)
    with pytest.raises(ValueError, match="The input is not a valid DataEnsemble object."):
        imagehandler.save_fits(DataEnsembleObject=D, filename="output")


def test_all(test_data, tmp_path, monkeypatch):
    """Read files in, write them back out, compare..."""
    monkeypatch.chdir(tmp_path)
    D = imagehandler.DataEnsemble(filename=[test_data / "test0.fits", test_data / "test1.fits"],
                                  EXPTIME="EXPOSURE", EXPUNIT="d",
                                  MJD_START="BJDREFI+TSTART+-2400000.5", FILTER="-Tess")
    imagehandler.save_fits(DataEnsembleObject=D, filename="output")
    # Imagehandler actually changes the header a little, so we can't just use filecmp to compare the files.
    # Instead, read output back in and compare.
    O = imagehandler.DataEnsemble(filename=["output_0.fits", "output_1.fits"],
                                  EXPTIME="EXPOSURE", EXPUNIT="d",
                                  MJD_START="BJDREFI+TSTART+-2400000.5", FILTER="-Tess")
    assert np.all(D.data == O.data)


