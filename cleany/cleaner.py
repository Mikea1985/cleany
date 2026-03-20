# -*- coding: utf-8 -*-
# cleany/cleany/cleaner.py

'''
   Classes / methods for reading raw image fits files and cleaning them
   Provides methods to
   - Read in images (using imagehandler's DataEnsemble)
   - align the images via reprojection (slow!)
   - background normalisation
   - template subtraction
   - masking
'''

# -----------------------------------------------------------------------------
# Third party imports
# -----------------------------------------------------------------------------
import os
import sys
import copy
import numpy as np
from typing import List
from datetime import datetime

# from astropy import wcs
from astropy.nddata import CCDData
from ccdproc import wcs_project  # , Combiner


# -----------------------------------------------------------------------------
# Any local imports
# -----------------------------------------------------------------------------
sys.path.append(os.path.dirname(os.path.dirname(
                os.path.realpath(__file__))))
from cleany.imagehandler import DataEnsemble, save_fits


# -----------------------------------------------------------------------------
# Various class definitions for *data import * in cleany
# -----------------------------------------------------------------------------

class DataCleaner():
    '''
    (1) Loads a list of fits-files, via DataEnsemble
    (2) Cleans data in various ways
    (3) Returns a cleaned DataEnsemble object and/or saves to fits files.

    methods:
    --------
    reproject_data
    subtract_background_level
    _subtract_average_background_level
    _subtract_sep_background_level
    template_subtract
    _subtract_provided_template
    _subtract_overall
    _subtract_local_average
    _subtract_donut

    main public method:
    -------------------
    reproject_data
    subtract_background_level
    template_subtract

    '''
    # Turning off some stupid syntax-checker warnings:
    # pylint: disable=too-many-instance-attributes
    # Why on earth should an object only have 7 attributes?
    # That seems dumb. Turning off this warning.
    # pylint: disable=too-few-public-methods
    # I get why an object should have at least two public methods in order to
    # not be pointless, but it's an annoying warning during dev. Turning off.
    # pylint: disable=attribute-defined-outside-init
    # Why should a method not be allowed to redefine an attribute?
    # It doesn't make any sense. Then what's the point of methods?
    # pylint: disable=too-many-locals
    # OK, this one I understand (methods should not be too long),
    # but it's currently just annoying.

    def __init__(self,
                 filename: List[str] = None,
                 extno: int = 0,
                 verbose: bool = False,
                 **kwargs) -> None:
        '''
        inputs:
        -------
        filename       - list of str  - list of filepaths to valid fits-files
        extno          - int OR       - Extension number to use for all images
                         list of int  - list of extension to use for each image
        verbose        - bool         - Print extra stuff if True
        EXPTIME        - str OR float - Exposure time in seconds
                                      - (keyword or value)
        EXPUNIT        - str OR float - Units on exposure time
                                      - ('s','m','h','d' or float seconds/unit)
        MJD_START      - str OR float - MJD at start of exposure
                                      - (keyword or value)
        FILTER         - str          - Filter name (keyword or '-name')
        NAXIS1         - str OR int   - Number of pixels along axis 1
        NAXIS2         - str OR int   - Number of pixels along axis 2
        INSTRUMENT     - str          - Instrument name (keyword)

        A float/integer value can be defined for most keywords,
        rather than a keyword name; this will use that value
        rather than searching for the keyword in the headers.
        INSTRUMENT and FILTER obviously can't be floats/integers,
        so use a leading '-' to specify a value
        rather than a keyword name to use.
        Not yet supported: list of a keyword values/names for each image, which
        hopefully should only ever be useful if attempting to stack images
        from different instruments; so low priority.
        '''
        # Set some values
        self.filename = filename
        self.extno = extno
        self.verbose = verbose
        if filename is not None:
            self.cleaned_data = DataEnsemble(filename, extno, verbose, **kwargs)
        else:
            self.cleaned_data = DataEnsemble()
        self.aligned = False
        self.reprojected = False
        self.bglevel_subtracted = False
        self.masked = False
        self.template_subtracted = False
        # Do some awesome stuff!!!

    def rough_align(self,
                    target: int = 0,
                    padmean: bool = False) -> None:
        '''
        Aligns images to integer pixel accuracy.

        inputs:
        -------
        target        - int  - Index of the target image to align relative to
        padmean       - bool - Whether to pad with the mean value (True) or
                               NaN values (False, default).

        outputs:
        --------
        self.cleaned_data, having been modified directly
        (should take less memory than making a copy to modify)
        '''
        # Offsets
        offsetsl = []
        for i, w in enumerate(self.cleaned_data.WCS):
            offsetsl.append(self.cleaned_data.WCS[target].all_world2pix(*w.all_pix2world(0, 0, 0), 0))
        offsets = np.array(offsetsl).round().astype(int)[:, ::-1]
        offsets -= offsets.min(0)
        xymax = np.max(offsets, 0)

        padded = []
        for i in np.arange(len(self.cleaned_data.data)):
            print(f"Aligning and padding image {i}", end='\r')
            pad_value = (np.nanmean(self.cleaned_data.data[i]) if padmean
                         else np.nan)  # Mean or NaN
            # Size of pad on 4 sides
            pad_size = ((offsets[i, 0], xymax[0] - offsets[i, 0]),
                        (offsets[i, 1], xymax[1] - offsets[i, 1]))
            # Align the array by adding a padding around the edge
            paddedi = np.pad(self.cleaned_data.data[i], pad_size,
                             constant_values=pad_value)
            padded.append(paddedi)
            # Update WCS to account for padding, both in .wcs and .header
            self.cleaned_data.WCS[i].wcs.crpix += (offsets[i, 1], offsets[i, 0])
            # Update header
            del self.cleaned_data.header[i]['CD?_?']  # required for update to work
            # Required to display correctly in ds9:
            self.cleaned_data.header[i].remove('DATASEC', ignore_missing=True)
            # Make new header:
            new_wcs_header = self.cleaned_data.WCS[i].to_header(relax=True)
            # Don't change things that aren't WCP related.
            for key in ['TIMESYS', 'TIMEUNIT', 'MJDREF', 'DATE-OBS',
                        'MJD-OBS', 'TSTART', 'DATE-END', 'MJD-END',
                        'TSTOP', 'TELAPSE', 'TIMEDEL', 'TIMEPIXR']:
                new_wcs_header.remove(key, ignore_missing=True)
            del new_wcs_header['???MJD???']
            del new_wcs_header['???TIME???']
            del new_wcs_header['???DATE???']
            self.cleaned_data.header[i].update(new_wcs_header)
            # Add a comment to the header about the reprojection
            now = str(datetime.today())[:19]
            self.cleaned_data.header[i]['COMMENT'] = (f'Data was alligned to '
                                                      f'integer pixel at {now}')
        del self.cleaned_data.data
        self.cleaned_data.data = np.array(padded)
        del padded
        self.aligned = True
        print("\nDone")

    def reproject_data(self,
                       target: int = 0,
                       padmean: bool = False) -> None:
        '''
        Reprojects each layer of self.cleaned_data.data to be aligned,
        using their WCS.
        By default aligns everything else with the first image,
        but this can be changed by setting target.

        inputs:
        -------
        target        - int  - Index of the target image to align relative to
        padmean       - bool - Whether to pad with the mean value (True) or
                               NaN values (False, default).

        outputs:
        --------
        self.cleaned_data, having been modified directly
        (should take less memory than making a copy to modify)
        '''
        # Do rough pixel alignment and padding.
        self.rough_align(target, padmean)

        for i, wcsi in enumerate(self.cleaned_data.WCS):
            print(f"Reprojecting image {i}", end='\r')
            # Do the reprojection
            self.cleaned_data.data[i] = wcs_project(CCDData(self.cleaned_data.data[i],
                                                            wcs=wcsi, unit='adu'),
                                                    self.cleaned_data.WCS[target]).data
            # Update WCS, both in self.cleaned_data.wcs and self.cleaned_data.header
            self.cleaned_data.WCS[i] = self.cleaned_data.WCS[target]
            del self.cleaned_data.header[i]['CD?_?']  # required for update to work
            new_wcs_header = self.cleaned_data.WCS[i].to_header(relax=True)
            for key in ['TIMESYS', 'TIMEUNIT', 'MJDREF', 'DATE-OBS',
                        'MJD-OBS', 'TSTART', 'DATE-END', 'MJD-END',
                        'TSTOP', 'TELAPSE', 'TIMEDEL', 'TIMEPIXR']:
                new_wcs_header.remove(key, ignore_missing=True)
            del new_wcs_header['???MJD???']
            del new_wcs_header['???TIME???']
            del new_wcs_header['???DATE???']
            self.cleaned_data.header[i].update(new_wcs_header)
            # Add a comment to the header about the reprojection
            now = str(datetime.today())[:19]
            self.cleaned_data.header[i]['COMMENT'] = (f'Data was reprojected to WCS '
                                                      f'of file {target} at {now}')
        self.reprojected = True
        print("\nDone")

    def subtract_background_level(self,
                                  mode: str = None) -> None:
        '''
        Subtract the background level (effectively making the background 0).

        inputs:
        -------
        mode - str - 'mean', 'median', 'sep' or 'source_extractor'
                      'sep' and 'source_extractor' are synonymous.

        outputs:
        --------
        self.cleaned_data, having been modified directly
        (should take less memory than making a copy to modify)

        mode = 'source_extractor' uses the 'sep' package to get the background.
        This allows a variation of the background accross the field, rather than just
        subtracting a single value.
        '''
        if mode == 'mean':
            self._subtract_average_background_level(usemean=True)
        elif mode == 'median':
            self._subtract_average_background_level(usemean=False)
        elif mode in ['sep', 'source_extractor']:
            self._subtract_sep_background_level()
        else:
            raise ValueError("subtract_background_level takes mode='mean', 'median'"
                             " 'sep' or 'source_extractor'")
        self.bglevel_subtracted = True

    def _subtract_sep_background_level(self,
                                       usemean: bool = False) -> None:
        '''
        Subtract the background level (effectively making the background 0)
        This method uses the 'sep' backage, which measures a varying
        background accross the image.

        inputs:
        -------
        usemean - Bool - use mean (True) instead of median (False)
                         Mean is faster, median is better.
                         Default is False

        outputs:
        --------
        self.cleaned_data, having been modified directly
        (should take less memory than making a copy to modify)
        '''
        import sep
        for i, dat in enumerate(self.cleaned_data.data):
            print(f"Subtracting background level in image {i}", end='\r')
            try:
                background_value = sep.Background(dat)
            except ValueError:
                background_value = sep.Background(dat.byteswap().newbyteorder())
            self.cleaned_data.data[i] -= background_value
            # Add a comment to the header about the subtraction
            now = str(datetime.today())[:19]
            self.cleaned_data.header[i]['COMMENT'] = (f'Background level '
                                                      f'subtracted at {now}')
        print("\nDone")

    def _subtract_average_background_level(self,
                                           usemean: bool = False) -> None:
        '''
        Subtract the background level (effectively making the background 0)

        inputs:
        -------
        usemean - Bool - use mean (True) instead of median (False)
                         Mean is faster, median is better.
                         Default is False

        outputs:
        --------
        self.cleaned_data, having been modified directly
        (should take less memory than making a copy to modify)
        '''
        for i, dat in enumerate(self.cleaned_data.data):
            print(f"Subtracting background level in image {i}", end='\r')
            background_value = (np.nanmean(dat) if usemean
                                else np.nanmedian(dat))
            std_value = np.nanstd(dat)
            idx_old = (dat > -np.inf)
            idx = np.abs(dat - background_value) < 5 * std_value
            j = 0
            while (not np.all(idx_old == idx)) & (j < 25):
                background_value = (np.nanmean(dat[idx]) if usemean
                                    else np.nanmedian(dat[idx]))
                std_value = np.nanstd(dat[idx])
                idx_old = idx
                idx = np.abs(dat - background_value) < 5 * std_value
                j += 1
            self.cleaned_data.data[i] -= background_value
            # Add a comment to the header about the subtraction
            now = str(datetime.today())[:19]
            self.cleaned_data.header[i]['COMMENT'] = (f'Background level '
                                                      f'subtracted at {now}')
        print("\nDone")

    def mask_bright_sources(self,
                            threshold: float = 20000.,
                            radius: float = 1.0) -> None:
        '''
        Mask pixels brighter than <threshold> as well as pixels within <radius>.

        inputs:
        -------
        threshold - int or float - Threshold above which to mask
        radius    - int or float - Radius around masked pixels to also mask
        '''
        mean_stack = np.nanmean(self.cleaned_data.data, 0)
        high_pixels_index = np.argwhere(mean_stack > threshold)
        neg_lim = int(np.floor(-radius))
        pos_lim = int(np.ceil(radius + 1))
        for hp_idx in high_pixels_index:
            for yi in np.arange(neg_lim, pos_lim):
                for xi in np.arange(neg_lim, pos_lim):
                    if (yi**2 + xi**2) <= radius**2:
                        yii = hp_idx[0] + yi
                        xii = hp_idx[1] + xi
                        if ((yii < len(self.cleaned_data.data[0, :, 0]))
                            & (xii < len(self.cleaned_data.data[0, 0, :]))
                            & (yii >= 0) & (xii >= 0)):
                            self.cleaned_data.data[:, yii, xii] = np.nan
        self.masked = True


    def template_subtract(self,
                          template_type: str = None,
                          usemean: bool = False,
                          nobs: int = 100,
                          ninner: int = 25,
                          nouter: int = 100,
                          template_array: np.ndarray = None) -> None:
        '''
        Subtract a template from each image.

        inputs:
        -------
        template_type - str - 'mean', 'median', 'local', 'donut', 'array'

        The following keywords are only used for certain template_types:
        if 'local':
        nobs    - int  - number of observations on each side to use
        usemean - bool - Use mean (True) or median (False)

        if 'donut':
        ninner  - int  - inner radius of 1D annulus.
                         number of observations on each side to _not_ use
                         (size of hole)
        nouter  - int  - outer radius of 1D annulus.
                         number of observations on each side
                         out to which used area extends.
        usemean - bool - Use mean (True) or median (False)

        if 'array':
        template_array - array - 2D array of same shape as an individual frame,
                                 or 3D array shape of whole data cube.
        '''
        valid_types = ['mean', 'median', 'local', 'donut', 'doughnut', 'array']
        if template_type not in valid_types:
            raise ValueError(f'template_type must be in {valid_types}')
        elif template_type == 'mean':
            self._subtract_overall(usemean=True)
        elif template_type == 'median':
            self._subtract_overall(usemean=False)
        elif template_type == 'local':
            self._subtract_local_average(usemean=usemean, nobs=nobs)
        elif template_type in ['donut', 'doughnut']:
            self._subtract_donut(usemean=usemean, ninner=ninner, nouter=nouter)
        elif (template_type == 'template_array') & (template_array is not None):
            self._subtract_user_defined_array(template_array=template_array)
        else:
            raise NotImplementedError('Code should never reach this point.'
                                      'You found a bug! Please report.')
        self.template_subtracted = True

    def _subtract_overall(self,
                          usemean: bool = False) -> None:
        '''
        Create a template from all of the observations
        and subtract it from every image.
        This is the fastest, but least good, template subtraction.

        inputs:
        -------
        usemean       - bool - Whether to use mean (True)
                               or median (False, default)

        outputs:
        --------
        self.cleaned_data, having been modified directly
        (should take less memory than making a copy to modify)
        '''
        print("Creating a universal template from all images.")
        universal_template = (np.nanmean(self.cleaned_data.data, 0) if usemean
                              else np.nanmedian(self.cleaned_data.data, 0))
        print("Subtracting template")
        self.cleaned_data.data -= universal_template
        for i in np.arange(len(self.cleaned_data.data)):
            # Add a comment to the header about the subtraction
            now = str(datetime.today())[:19]
            self.cleaned_data.header[i]['COMMENT'] = (f'Template '
                                                      f'subtracted at {now}')
        print("\nDone")

    def _subtract_local_average(self,
                                usemean: bool = False,
                                nobs: int = 100):
        '''
        Create a 'local' (temporally) template from the closest nobs
        observations on each side of each image, and subtract it.

        inputs:
        -------
        usemean       - bool - Whether to use mean (True)
                               or median (False, default)
        nobs          - int  - number of observations before and after

        outputs:
        --------
        self.cleaned_data, having been modified directly
        (should take less memory than making a copy to modify)
        '''
        original_data = copy.deepcopy(self.cleaned_data.data)
        for i in np.arange(len(self.cleaned_data.data)):
            print(f"Subtracting template from image {i}", end='\r')
            i_min = np.max([0, i - nobs])
            i_max = np.min([i + nobs, len(original_data)])
            self.cleaned_data.data[i] -= (np.nanmean(original_data[i_min:i_max], 0)
                                          if usemean
                                          else np.nanmedian(original_data[i_min:i_max], 0))
            # Add a comment to the header about the subtraction
            now = str(datetime.today())[:19]
            self.cleaned_data.header[i]['COMMENT'] = (f'Template '
                                                      f'subtracted at {now}')
        print("\nDone")

    def _subtract_donut(self,
                        usemean: bool = False,
                        nouter: int = 100,
                        ninner: int = 25):
        '''
        Create a 'local' (temporally) template from the closest nouter
        observations on each side of each image, with an ninner hole on each
        side, and subtract it.
        This should prevent subtracting the signal from a moving source.

        inputs:
        -------
        usemean       - bool - Whether to use mean (True)
                               or median (False, default)
        nouter          - int  - number of observations before and after
        ninner          - int  - number of observations before and after

        outputs:
        --------
        self.cleaned_data, having been modified directly
        (should take less memory than making a copy to modify)
        '''
        original_data = copy.deepcopy(self.cleaned_data.data)
        nobs = len(original_data)
        for i in np.arange(len(self.cleaned_data.data)):
            print(f"Subtracting template from image {i}", end='\r')
            i_min = np.max([0, i - nouter + 1])
            i_mid1 = np.max([0, i - ninner + 1])
            i_mid2 = np.min([i + ninner, nobs])
            i_max = np.min([i + nouter, nobs])
            use_data = (list(original_data[i_min:i_mid1])
                        + list(original_data[i_mid2:i_max]))
            donut_template = (np.mean(use_data, 0) if usemean
                              else np.median(use_data, 0))
            self.cleaned_data.data[i] -= donut_template
            # Add a comment to the header about the subtraction
            now = str(datetime.today())[:19]
            self.cleaned_data.header[i]['COMMENT'] = (f'Template '
                                                      f'subtracted at {now}')
        print("\nDone")

    def _subtract_user_defined_array(self, template_array: np.ndarray) -> None:
        '''
        Create a 'local' (temporally) template from the closest nouter
        observations on each side of each image, with an ninner hole on each
        side, and subtract it.
        This should prevent subtracting the signal from a moving source.

        inputs:
        -------
        template_array - array - 2D array of same shape as an individual frame,
                                 or 3D array shape of whole data cube.
        outputs:
        --------
        self.cleaned_data, having been modified directly
        (should take less memory than making a copy to modify)
        '''
        self.cleaned_data.data -= template_array

    def save_cleaned(self,
                     filename: str = 'clean') -> None:
        '''Save the shifted images to fits files.'''
        save_fits(self.cleaned_data, filename, self.verbose)


# END
