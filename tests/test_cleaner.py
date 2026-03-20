"""Tests for the cleaner.py module"""

import sys
import copy
import pytest
import numpy as np
from pathlib import Path
from astropy.io import fits

# Local import
sys.path.append(str(Path.cwd().parent))
from cleany import cleaner

test_filenames = ["test0.fits", "test1.fits", "test2.fits", "test3.fits", "test4.fits"]

def test_DataCleaner_empty():
    C = cleaner.DataCleaner()
    assert C.filename is None
    assert C.extno == 0
    assert C.aligned is False
    assert C.reprojected is False
    assert C.bglevel_subtracted is False
    assert C.template_subtracted is False
    assert C.cleaned_data.data is None


def test_DataCleaner_with_input(test_data):
    test_files = [test_data / f for f in test_filenames]
    C = cleaner.DataCleaner(filename=test_files,
                            extno=0, EXPTIME="EXPOSURE", EXPUNIT="d",
                            MJD_START="BJDREFI+TSTART+-2400000.5", FILTER="-Tess")
    assert C.filename is test_files
    assert C.extno == 0
    assert C.aligned is False
    assert C.reprojected is False
    assert C.bglevel_subtracted is False
    assert C.template_subtracted is False
    assert isinstance(C.cleaned_data.data, np.ndarray)


def test_rough_align(test_data):
    test_files = [test_data / f for f in test_filenames]
    C = cleaner.DataCleaner(filename=test_files,
                            extno=0, EXPTIME="EXPOSURE", EXPUNIT="d",
                            MJD_START="BJDREFI+TSTART+-2400000.5", FILTER="-Tess")
    shape_before = C.cleaned_data.data.shape
    assert shape_before == (5, 140, 140)
    C.rough_align(target=0)
    assert C.aligned is True
    assert isinstance(C.cleaned_data.data, np.ndarray)
    shape_after = C.cleaned_data.data.shape
    # I've deliberately offset the WCS of "test4.fits" to be 1 pixel off, to show the padding that happens here
    assert shape_before != shape_after
    assert shape_after == (5, 141, 141)
    assert np.isnan(C.cleaned_data.data[0][0, 0])
    assert np.isnan(C.cleaned_data.data[0][0, -1])
    assert np.isnan(C.cleaned_data.data[0][-1, 0])
    assert not np.isnan(C.cleaned_data.data[0][-1, -1])


def test_reproject_data(test_data):
    test_files = [test_data / f for f in test_filenames]
    C = cleaner.DataCleaner(filename=test_files,
                            extno=0, EXPTIME="EXPOSURE", EXPUNIT="d",
                            MJD_START="BJDREFI+TSTART+-2400000.5", FILTER="-Tess")
    C.reproject_data(target=0)
    C.reprojected is True
    # I think I need better test data to truly test this well with assertive statements.


def test_subtract_background_level(test_data):
    test_files = [test_data / f for f in test_filenames]
    C = cleaner.DataCleaner(filename=test_files,
                            extno=0, EXPTIME="EXPOSURE", EXPUNIT="d",
                            MJD_START="BJDREFI+TSTART+-2400000.5", FILTER="-Tess")
    median_value_before = np.array([np.nanmedian(frame) for frame in C.cleaned_data.data])
    C.subtract_background_level(mode='median')
    C.bglevel_subtracted is True
    median_value_after = np.array([np.nanmedian(frame) for frame in C.cleaned_data.data])
    # Before, the median pixel value is aroung 150
    assert np.all(median_value_before > 100)
    # After, it is close to 0 (not exact, due to limitations of subtracting median value from 19600 float32 values).
    assert np.all(np.abs(median_value_after) < 0.5)


def test_mask_bright_sources(test_data):
    test_files = [test_data / f for f in test_filenames]
    C = cleaner.DataCleaner(filename=test_files,
                            extno=0, EXPTIME="EXPOSURE", EXPUNIT="d",
                            MJD_START="BJDREFI+TSTART+-2400000.5", FILTER="-Tess")
    max_value_before = np.array([np.nanmax(frame) for frame in C.cleaned_data.data])
    assert np.all(max_value_before > 1000)
    nan_values_before = np.array([np.sum(np.isnan(frame)) for frame in C.cleaned_data.data])
    assert np.all(nan_values_before == 0)
    C.mask_bright_sources(threshold=1000, radius=3)
    assert C.masked is True
    max_value_after = np.array([np.nanmax(frame) for frame in C.cleaned_data.data])
    assert np.all(max_value_after <= 1000)
    nan_values_after = np.array([np.sum(np.isnan(frame)) for frame in C.cleaned_data.data])
    # Each frame (19600 pixels) should have 600-700 masked pixels after.
    assert np.all(nan_values_after > 600) & np.all(nan_values_after < 700)

def test_templace_subtract_donut(test_data):
    test_files = [test_data / f for f in test_filenames]
    C = cleaner.DataCleaner(filename=test_files,
                            extno=0, EXPTIME="EXPOSURE", EXPUNIT="d",
                            MJD_START="BJDREFI+TSTART+-2400000.5", FILTER="-Tess")
    before = copy.deepcopy(C.cleaned_data.data)
    # Let's use h=2, d=1 so that the image itself and its immediate neighbors are not used in the template.
    C.template_subtract(template_type='donut', ninner=2, nouter=3)
    # With just 5 images, we're really only getting true donut'ing on the center image, so test that one.
    # So the middle image after should equal the middle image before minus the median of the first and last image)
    assert np.all(C.cleaned_data.data[2] == before[2] - np.nanmedian([before[0], before[4]], 0))


def test_templace_subtract_median(test_data):
    test_files = [test_data / f for f in test_filenames]
    C = cleaner.DataCleaner(filename=test_files,
                            extno=0, EXPTIME="EXPOSURE", EXPUNIT="d",
                            MJD_START="BJDREFI+TSTART+-2400000.5", FILTER="-Tess")
    before = copy.deepcopy(C.cleaned_data.data)
    # I'll modify the data, to ensure it'll be obvious the donut worked correctly
    C.template_subtract(template_type='median')
    # With just 5 images, we're really only getting true donut'ing on the center image, so test that one.
    assert np.all(C.cleaned_data.data[2] == before[2] - np.nanmedian(before, 0))
