# -*- coding: utf-8 -*-
# cleany/cleany/imagehandler.py

'''
   Classes / methods for reading image fits files.
'''

# -----------------------------------------------------------------------------
# Third party imports
# -----------------------------------------------------------------------------
import numpy as np
from astropy import wcs
from astropy.io import fits
from datetime import datetime
from typing import Any, Dict, List

# -----------------------------------------------------------------------------
# Various class definitions for *data import * in cleany
# -----------------------------------------------------------------------------


class OneImage():
    '''
    (1)Loads a fits-file,
       including data array, WCS, header and select header keywords
    (2)Calculate pixel coordinates? (Seems unneccessary at this stage?)

    methods:
    --------
    loadImageAndHeader()
    others?

    main public method:
    -------------------
    loadImageAndHeader()


    '''
    # Turning off some stupid syntax-checker warnings:
    # pylint: disable=too-many-instance-attributes
    # Why on earth should an object only have 7 attributes?
    # That seems dumb. Turning off this warning.
    # pylint: disable=too-few-public-methods
    # I get why an object should have at least two public methods in order to
    # not be pointless, but it's an annoying warning during dev. Turning off.

    def __init__(self,
                 filename: str = None,
                 extno: int = 0,
                 verbose: bool = False,
                 **kwargs) -> None:
        '''
        inputs:
        -------
        filename       - str          - filepath to one valid fits-file
        extno          - int          - Extension number of image data
                                      - (0 if single-extension)
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
        '''
        # Make readOneImageAndHeader a method even though it doesn't need to be
        self.readOneImageAndHeader = readOneImageAndHeader
        # Initialize some attributes that will get filled later
        self.key_values = {}       # filled out by loadImageAndHeader
        self.header_keywords = {}  # filled out by loadImageAndHeader
        self.WCS = None            # filled out by loadImageAndHeader
        self.header = None         # filled out by loadImageAndHeader
        self.header0 = None        # filled out by loadImageAndHeader
        self.data = None           # filled out by loadImageAndHeader
        self.filename = filename
        self.extno = extno
        if filename:
            (self.data, self.header, self.header0, self.WCS,
             self.header_keywords, self.key_values
             ) = self.readOneImageAndHeader(filename, extno,
                                            verbose=verbose, **kwargs)

    # -------------------------------------------------------------------------
    # Public Methods
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # The methods below are for the loading of *general* fits-format data files
    # -------------------------------------------------------------------------


class DataEnsemble():
    '''
    (1)Loads a list of fits-files
    (2)Stores image data in a cube array, WCS in an array, headers in an array.
    (3)Can reproject the data to a common WCS if neccesarry.

    methods:
    --------
    reproject_data()

    main public method:
    -------------------
    reproject_data()


    '''
    # Turning off some stupid syntax-checker warnings:
    # pylint: disable=too-many-instance-attributes
    # Why on earth should an object only have 7 attributes?
    # That seems dumb. Turning off this warning.
    # pylint: disable=too-few-public-methods
    # I get why an object should have at least two public methods in order to
    # not be pointless, but it's an annoying warning during dev. Turning off.

    def __init__(self,
                 filename: List[str] = None,
                 extno: int = 0,
                 verbose: bool = False,
                 **kwargs):
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
        self.filename = filename
        if filename is not None:
            # Set some values
            self.extno = extno
            self.reprojected = False
            self.read_fits_files(verbose=verbose, **kwargs)
        else:
            self.extno = None
            self.reprojected = None
            self.data = None
            self.WCS = None
            self.header = None

    def read_fits_files(self,
                        verbose: bool = False,
                        **kwargs) -> None:
        '''
        Read in the files defined in self.filename
        from extension[s] in self.extno.

        input:
        ------
        self
        verbose        - bool         - Print extra stuff if True
        kwargs passed along into OneImage
        '''
        # Do some awesome stuff!!!
        datacube = []
        wcscube = []
        headercube = []
        for i, filei in enumerate(self.filename):
            print(f"Reading image {i}: {filei}", end='\r')
            exti = self.extno if isinstance(self.extno, int) else self.extno[i]
            OneIm = OneImage(filei, exti, verbose, **kwargs)
            datacube.append(OneIm.data)
            wcscube.append(OneIm.WCS)
            headercube.append(OneIm.header)
        print("")
        print(f"Read {len(self.filename)} files!")
        self.data = np.array(datacube)
        self.WCS = np.array(wcscube)
        self.header = headercube
        print("Done!")


# -------------------------------------------------------------------------
# These functions really don't need to be methods, and therefore aren't.
# No need to over-complicate things.
# -------------------------------------------------------------------------

def save_fits(DataEnsembleObject: DataEnsemble,
              filename: str='data.fits',
              verbose: bool=True):
    '''
    Save some data to [a] fits file[s].

    input:
    ------
    DataEnsembleObject - DataEnsemble Object - contains data and header.
    filename           - str                 - name or shared start of filename

    output:
    -------
    one or more fits file.
    If multiple, they are numbered *_000.fits, *_001.fits, etc.
    '''
    if len(DataEnsembleObject.data.shape) == 2:
        filename = filename if '.fits' in filename else filename + '.fits'
        hdu = fits.PrimaryHDU(data=DataEnsembleObject.data,
                              header=DataEnsembleObject.header)
        if verbose:
            print(f'Saving to file {filename}')
        hdu.writeto(filename, overwrite=True)
    elif len(DataEnsembleObject.data.shape) == 3:
        # Calculate how many digits number need to have
        # (ie. if there are 1108 files, filenames run from _0000 to _1107
        # but if there are only 18 files, filenames run from _00 to _17)
        nfiles = len(str(len(DataEnsembleObject.data)))
        # Save each file
        for i, data in enumerate(DataEnsembleObject.data):
            filenamei = filename.replace('.fits', '')
            filenamei += f'_{i:0{nfiles}.0f}.fits'
            hdu = fits.PrimaryHDU(data=data,
                                  header=DataEnsembleObject.header[i])
            if verbose:
                print(f'Saving to file {filenamei}', end='\r')
            hdu.writeto(filenamei, overwrite=True)
    else:
        raise ValueError('The input is not a valid DataEnsemble object.')
    print('\nDone!')


def readOneImageAndHeader(filename:str = None,
                          extno: int = 0,
                          verbose: bool = False,
                          **kwargs):
    '''
    Reads in a fits file (or a given extension of one).
    Returns the image data, the header, and a few useful keyword values.

    input:
    ------
    filename       - str          - valid filepath to one valid fits-file
    extno          - int          - Extension number of image data
                                  - (0 if single-extension)
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

    output:
    -------
    data            - np.array   - the float array of pixel data
                                 - shape == (NAXIS2, NAXIS1)
    header          - fits.header.Header - Sort of like a dictionary
    header0         - fits.header.Header - Sort of like a dictionary
    WCS             - wcs.wcs.WCS - World Coordinate System plate solution
    header_keywords - dictionary - A bunch of header keyword names.
                                 - Needed later at all??? Not sure
    key_values      - dictionary - A bunch of important values

    Content of key_values:
    EXPTIME         - float      - Exposure time in seconds
    MJD_START       - float      - MJD at start of exposure
    MJD_MID         - float      - MJD at centre of exposure
    FILTER          - str        - Filter name
    NAXIS1          - int        - Number of pixels along axis 1
    NAXIS2          - int        - Number of pixels along axis 2
    '''

    # Check whether a filename is supplied.
    if filename is None:
        raise TypeError('filename must be supplied!')

    # Define default keyword names
    header_keywords = {'EXPTIME': 'EXPTIME',      # Exposure time [s]
                       'MJD_START': 'MJD-STR',    # MJD at start
                       'FILTER': 'FILTER',        # Filter name
                       'NAXIS1': 'NAXIS1',        # Pixels along axis1
                       'NAXIS2': 'NAXIS2',        # Pixels along axis2
                       'INSTRUMENT': 'INSTRUME',  # Instrument name
                       }
    header_comments = {'EXPTIME': 'Exposure time [s]',
                       'MJD_START': 'MJD at start of exposure',
                       'FILTER': 'Filter letter',
                       'NAXIS1': 'Pixels along axis1',
                       'NAXIS2': 'Pixels along axis2',
                       'INSTRUMENT': 'Instrument name',
                       }
    key_values = {}
    EXPUNIT = 's'
    xycuts = None

    # Do a loop over the kwargs and see if any header keywords need updating
    # (because they were supplied)
    for key, non_default_name in kwargs.items():
        if key in header_keywords:
            header_keywords[key] = non_default_name
        if key=='EXPUNIT':
            EXPUNIT = non_default_name
        if key=='xycuts':
            xycuts = non_default_name

    # Read the file. Do inside a "with ... as ..." to auto close file after
    with fits.open(filename) as han:
        data = han[extno].data
        header = han[extno].header  # Header for the extension
        # Overall header for whole mosaic, etx0:
        header0 = han[0].header  # pylint: disable=E1101 # Pylint stupid errors

    if xycuts is not None:
        xycuts[1] = (xycuts[1] - 1 + header['NAXIS1']) % header['NAXIS1'] + 1
        xycuts[3] = (xycuts[3] - 1 + header['NAXIS2']) % header['NAXIS2'] + 1
        if verbose:
            print(xycuts)
        data = data[xycuts[2]:xycuts[3], xycuts[0]:xycuts[1]]
        # CRPIX1
        header['OLD_CRPIX1'] = (header['CRPIX1'], header.comments['CRPIX1'])
        header['COMMENT'] = (f'OLD_CRPIX1 added by CLEANY'
                             f' at {str(datetime.today())[:19]}')
        header['COMMENT'] = (f'OLD_CRPIX1 contains old value of CRPIX1')
        header['CRPIX1'] -= xycuts[0]
        # CRPIX2
        header['OLD_CRPIX2'] = (header['CRPIX2'], header.comments['CRPIX2'])
        header['COMMENT'] = (f'OLD_CRPIX2 added by CLEANY'
                             f' at {str(datetime.today())[:19]}')
        header['COMMENT'] = (f'OLD_CRPIX2 contains old value of CRPIX2')
        header['CRPIX2'] -= xycuts[2]
        # NAXIS1
        header['OLD_NAXIS1'] = (header['NAXIS1'],
                                header.comments['NAXIS1'])
        header['COMMENT'] = (f'OLD_NAXIS1 added by CLEANY'
                             f' at {str(datetime.today())[:19]}')
        header['COMMENT'] = (f'OLD_NAXIS1 contains old value of NAXIS1')
        header['NAXIS1'] = xycuts[1] - xycuts[0]
        header['COMMENT'] = (f'NAXIS1 updated by CLEANY'
                             f' at {str(datetime.today())[:19]}')
        header['COMMENT'] = (f'NAXIS1 adjusted for xycut')
        # NAXIS2
        header['OLD_NAXIS2'] = (header['NAXIS2'],
                                header.comments['NAXIS2'])
        header['COMMENT'] = (f'OLD_NAXIS2 added by CLEANY'
                             f' at {str(datetime.today())[:19]}')
        header['COMMENT'] = (f'OLD_NAXIS2 contains old value of NAXIS2')
        header['NAXIS2'] = xycuts[3] - xycuts[2]
        header['COMMENT'] = (f'NAXIS2 updated by CLEANY'
                             f' at {str(datetime.today())[:19]}')
        header['COMMENT'] = (f'NAXIS2 adjusted for xycut')

    # Use the defined keywords to save the values into key_values.
    # Search both headers if neccessary (using keyValue)
    for key, use in header_keywords.items():
        if verbose:
            print(key, use)
        if key in ['NAXIS1', 'NAXIS2']:
            # Some keywords can have an integer value defined
            # instead of keyword name
            key_values[key] = (use if isinstance(use, int) else
                               int(_find_key_value(header, header0, use)))
        elif key == 'INSTRUMENT':
            # INSTRUMENT and FILTER obviously can't be floats,
            # so use a leading '-' to specify a value to use
            # instead of a keyword name.
            key_values[key] = (use[1:] if use[0] == '-'
                               else _find_key_value(header, header0, use))
        elif key == 'FILTER':
            # Filter only wants the first character of the supplied,
            # not all the junk (telescopes usually put numbers after)
            key_values[key] = (use[1] if use[0] == '-' else
                               _find_key_value(header, header0, use)[0])
        elif key == 'MJD_START':
            # Sometimes, like for Tess, the MJD of the start isn't given
            # but can be calculated from the sum of a reference MJD and
            # the time since that reference. 
            # A + in the keyword is used to indicate the sum of two keywords.
            if isinstance(use, float):
                key_values[key] = use
            elif isinstance(use, str):
                uses = use.split('+')
                key_values[key] = 0.
                for usei in uses:
                    if '*' in usei:
                        multiplier = eval(''.join(usei.split('*')[1:]))
                        usei = usei.split('*')[0]
                    else:
                        multiplier = 1
                    try:  # see whether the value is just a number:
                        key_values[key] += float(usei) * multiplier
                    except(ValueError):
                        key_values[key] += (float(_find_key_value(header,
                                                                  header0,
                                                                  usei))
                                            * multiplier)
            else:
                raise TypeError('MJD_START must be float or string')
        elif key == 'EXPTIME':
            # Some stupid telescopes record exposure time in unites 
            # other than seconds... allow for this.
            if isinstance(EXPUNIT, str) or isinstance(EXPUNIT, float):
                timeunit = (86400.0 if EXPUNIT=='d' else
                            1440.0 if EXPUNIT=='h' else
                            24.0 if EXPUNIT=='m' else
                            1.0 if EXPUNIT=='s' else
                            EXPUNIT)
            else:
                raise TypeError('EXPUNIT must be float or string')
            key_values[key] = timeunit * (use if isinstance(use, float) else
                                  float(_find_key_value(header, header0, use)))
        else:
            # Most keywords can just have a float value defined
            # instead of keyword name, that's what the if type is about.
            key_values[key] = (use if isinstance(use, float) else
                               float(_find_key_value(header, header0, use)))
        if verbose:
            print(key_values.keys())
        header[f'CLEANY_{key}'] = (key_values[key], header_comments[key])
        header['COMMENT'] = (f'CLEANY_{key} added by CLEANY'
                             f' at {str(datetime.today())[:19]}')
        header['COMMENT'] = f'CLEANY_{key} derived from {use}'

    # Also define the middle of the exposure:
    key_values['MJD_MID'] = (key_values['MJD_START'] +
                             key_values['EXPTIME'] / (86400 * 2))
    header[f'CLEANY_MJD_MID'] = (key_values['MJD_MID'],
                                 'MJD at middle of exposure')
    header['COMMENT'] = (f'CLEANY_MJD_MID added by CLEANY'
                         f' at {str(datetime.today())[:19]}')
    header['COMMENT'] = (f'CLEANY_MJD_MID derived from '
                         f'{header_keywords["MJD_START"]} '
                         f'and {header_keywords["EXPTIME"]}')

    print('{}\n'.format((key_values)) if verbose else '', end='')
    return data, header, header0, wcs.WCS(header), header_keywords, key_values


def _find_key_value(header1: Dict[str, Any], header2: Dict[str, Any], key: str):
    """
    First checks extension header for keyword; if fails, checks main header.
    This is neccessary because some telescopes put things like the EXPTIME
    in the main header, while others put it in the header for each
    extension and NOT the main one.

    input:
    ------
    key - str - keyword to look for

    output:
    -------
    value of keyword found in headers
    """
    try:
        value = header1[key]
    except KeyError:
        value = header2[key]
    return value


# END
