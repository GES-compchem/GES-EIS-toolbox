import traceback
import numpy as np
from numpy.testing import assert_array_almost_equal

from ges_eis_toolbox.spectra.spectrum import EIS_Spectrum

# Test the EIS_spectrum class constructor for the case of list
def test_EIS_spectrum___init___float():

    f, Z = [1, 10, 100], [1.0, 1.0j, 1.0 - 1.0j]

    try:
        EIS_Spectrum(f, Z)
    except:
        print(traceback.format_exc())
        assert False, "Unexpected exception raised on EIS_Spectrum class construction"
    else:
        assert True


# Test the EIS_spectrum class constructor for the case of numpy array
def test_EIS_spectrum___init___array():

    f, Z = np.array([1, 10, 100]), np.array([1.0, 1.0j, 1.0 - 1.0j])

    try:
        EIS_Spectrum(f, Z)
    except:
        print(traceback.format_exc())
        assert False, "Unexpected exception raised on EIS_Spectrum class construction"
    else:
        assert True


# Test the EIS_spectrum class constructor falure on wrong type or length mismatch
def test_EIS_spectrum___init___fail():

    f = np.array([1, 10, 100])

    try:
        EIS_Spectrum(f, "A string")
    except TypeError:
        assert True
    else:
        assert False, "Expected exception was not raised on EIS_Spectrum class construction"

    try:
        EIS_Spectrum(f, [1, 2, 3, 4])
    except ValueError:
        assert True
    else:
        assert False, "Expected exception was not raised on EIS_Spectrum class construction"


# Test the EIS_spectrum properties
def test_EIS_spectrum_properties():

    f = np.array([0.1, 1, 10, 100])
    Z = np.array([1.0, 1.0j, 1.0 - 1.0j, -1.5j])

    spectrum = EIS_Spectrum(f, Z)

    assert_array_almost_equal(spectrum.frequency, f, decimal=6)
    assert_array_almost_equal(spectrum.impedance, Z, decimal=6)
    assert_array_almost_equal(spectrum.real_Z, [1.0, 0.0, 1.0, 0.0], decimal=6)
    assert_array_almost_equal(spectrum.imag_Z, [0.0, 1.0, -1.0, -1.5], decimal=6)
    assert_array_almost_equal(spectrum.norm_Z, [1.0, 1.0, 1.4142135624, 1.5], decimal=6)
    assert_array_almost_equal(
        spectrum.phi_Z, [0.0, 1.5707963268, -0.7853981634, -1.570796328], decimal=6
    )

    # Test also that the arithmetic is correct
    Za = spectrum.real_Z + 1.j*spectrum.imag_Z
    Zb = spectrum.norm_Z*np.exp(1.j*spectrum.phi_Z)
    assert_array_almost_equal(Za, Zb, decimal=6)
    