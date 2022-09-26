from numpy.testing import assert_array_almost_equal

from ges_eis_toolbox.database.generator import Range, Generator


# TESTING THE RANGE CLASS
# ------------------------------------------------------------------------------------------

# Test the constructor of the Range class
def test_Range___init__():

    # Test the case with single float value
    try:
        Range(0.0, 1.0)
    except:
        assert False, "Unexpected exception raised during 'Range' construction"
    else:
        assert True

    # Test the case with a float list
    try:
        Range([0.0, 0.1], [1.0, 0.6])
    except:
        assert False, "Unexpected exception raised during 'Range' construction"
    else:
        assert True


# Test the constructor of the Range class failures
def test_Range___init___fail():

    # Test the case with wrong type
    try:
        Range(0.0, "this is a string")
    except TypeError:
        assert True
    else:
        assert (
            False
        ), "An expected TypeError exception was not raised during 'Range' construction"

    # Test the case with wrong type in list
    try:
        Range([0.0, 0.0], ["A", "B"])
    except TypeError:
        assert True
    else:
        assert (
            False
        ), "An expected TypeError exception was not raised during 'Range' construction"

    # Test the case of list length mismatch
    try:
        Range([0.0, 0.0], [1.0])
    except ValueError:
        assert True
    else:
        assert (
            False
        ), "An expected ValueError exception was not raised during 'Range' construction"

    # Test the case of max smaller than min with single float value
    try:
        Range(1.0, 0.0)
    except ValueError:
        assert True
    else:
        assert (
            False
        ), "An expected ValueError exception was not raised during 'Range' construction"

    # Test the case of max smaller than min with float list
    try:
        Range([0.0, 1.0], [1.0, 0.0])
    except ValueError:
        assert True
    else:
        assert (
            False
        ), "An expected ValueError exception was not raised during 'Range' construction"


# Test the __len__ function of the Range class
def test_Range___len__():

    r_1d = Range(0.1, 1.2)
    r_2d = Range([0.4, 0.1], [0.5, 1.2])

    assert len(r_1d) == 1
    assert len(r_2d) == 2


# Test the generate_step function of the Range class
def test_Range_generate_step():

    r_1d = Range(0.1, 1.2)
    r_2d = Range([0.4, 0.1], [0.5, 1.2])

    x_1d = r_1d.generate_step([2], 5)
    assert_array_almost_equal(x_1d, [0.65], decimal=6)

    x_2d = r_2d.generate_step([3, 2], 5)
    assert_array_almost_equal(x_2d, [0.475, 0.65], decimal=6)


# Test the Range class properties
def test_Range_properties():

    r_1d = Range(0.1, 1.2)
    r_2d = Range([0.4, 0.1], [0.5, 1.2])

    assert_array_almost_equal(r_1d.min, [0.1], decimal=10)
    assert_array_almost_equal(r_1d.max, [1.2], decimal=10)
    assert_array_almost_equal(r_1d.delta, [1.1], decimal=10)

    assert_array_almost_equal(r_2d.min, [0.4, 0.1], decimal=10)
    assert_array_almost_equal(r_2d.max, [0.5, 1.2], decimal=10)
    assert_array_almost_equal(r_2d.delta, [0.1, 1.1], decimal=10)

