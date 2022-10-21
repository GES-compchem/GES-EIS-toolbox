from os import listdir
from os.path import isfile, join
from numpy.testing import assert_array_almost_equal, assert_almost_equal

from ges_eis_toolbox.circuit.circuit_string import CircuitString
from ges_eis_toolbox.database.generator import Range, Generator, SamplingMode
from ges_eis_toolbox.database.data_entry import DataPoint


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


# Test the generate_step function of the Range class with linear scheme
def test_Range_generate_step_linear():

    r_1d = Range(0.1, 1.0)
    r_2d = Range([0.41, 0.1], [0.5, 1.0])

    x_1d = r_1d.generate_step([2], 10, SamplingMode.linear)
    assert_array_almost_equal(x_1d, [0.3], decimal=6)

    x_2d = r_2d.generate_step([3, 2], 10, SamplingMode.linear)
    assert_array_almost_equal(x_2d, [0.44, 0.3], decimal=6)


# Test the generate_step function of the Range class with logarithmic scheme
def test_Range_generate_step_logarithmic():

    r_1d = Range(0.1, 1.0)
    r_2d = Range([0.41, 0.1], [0.5, 1.0])

    x_1d = r_1d.generate_step([2], 11, SamplingMode.logarithmic)
    assert_array_almost_equal(x_1d, [0.1584893192], decimal=6)

    x_2d = r_2d.generate_step([3, 2], 11, SamplingMode.logarithmic)
    assert_array_almost_equal(x_2d, [0.4351507146, 0.1584893192], decimal=6)


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


# TESTING THE GENERATOR CLASS
# ------------------------------------------------------------------------------------------

# Test the Generator class constructor
def test_Generator___init__():

    cs = CircuitString("R0-p(R1,C0)-p(R2,C1)")
    p = {
        "R0": {"R0": Range(0.1, 1.0)},
        "p(C0,R0)": {"R0": Range(1.0, 10.0), "C0": Range(0.01, 1.0)},
    }

    try:
        Generator(cs, p)
    except:
        assert False, "Unexpected exception occured on Generator class construction"
    else:
        assert True


# Test the generate_parameterization function of the Generator class
def test_Generator_generate_parameterization():

    cs = CircuitString("R0-p(R1,C0)-p(R2,C1)")
    p = {
        "R0": {"R0": Range(0.1, 1.0)},
        "p(C0,R0)": {"R0": Range(1.0, 10.0), "C0": Range(0.01, 1.0)},
    }

    gen = Generator(cs, p, steps_limit=4, sampling_scheme=SamplingMode.linear)

    try:
        results = gen.generate_parameterization(0)
    except:
        assert False, "Unexpected exception occurred on generate_parameterization call"
    else:
        assert len(results) == cs.number_of_components
        assert_almost_equal(results["R0"], 0.1, decimal=6)
        assert_almost_equal(results["R1"], 1.0, decimal=6)
        assert_almost_equal(results["R2"], 1.0, decimal=6)
        assert_almost_equal(results["C0"], 0.01, decimal=6)
        assert_almost_equal(results["C1"], 0.01, decimal=6)

    try:
        results = gen.generate_parameterization(12)
    except:
        assert False, "Unexpected exception occurred on generate_parameterization call"
    else:
        assert len(results) == cs.number_of_components
        assert_almost_equal(results["R0"], 0.1, decimal=6)
        assert_almost_equal(results["R1"], 1.0, decimal=6)
        assert_almost_equal(results["R2"], 1.0, decimal=6)
        assert_almost_equal(results["C0"], 0.67, decimal=6)
        assert_almost_equal(results["C1"], 0.01, decimal=6)


# Test the failure modes of the generate_parameterization function of the Generator class
def test_Generator_generate_parameterization_failure():

    cs = CircuitString("R0-p(R1,C0)-p(R2,C1)")
    p = {
        "R0": {"R0": Range(0.1, 1.0)},
        "p(C0,R0)": {"R0": Range(1.0, 10.0), "C0": Range(0.01, 1.0)},
    }

    gen = Generator(cs, p, steps_limit=4)

    try:
        gen.generate_parameterization(-1)
    except:
        assert True
    else:
        assert False, "An exception was expected for a call to a negative index"

    try:
        gen.generate_parameterization(400)
    except:
        assert True
    else:
        assert False, "An exception was expected for a call greated than the maximum index"


# Test the on_the_fly_dataset function of the Generator class
def test_Generator_on_the_fly_dataset():

    freq = [0.1, 1., 10., 100.]
    cs = CircuitString("R0-p(R1,C0)-p(R2,C1)")
    p = {
        "R0": {"R0": Range(0.1, 1.0)},
        "p(C0,R0)": {"R0": Range(1.0, 10.0), "C0": Range(0.01, 1.0)},
    }

    gen = Generator(cs, p, steps_limit=4)

    try:
        dataset = gen.on_the_fly_dataset(freq)
    except:
        assert False, "Unexpected exception raised during on the fly dataset generation"
    else:
        assert len(dataset) == 400
        
        for dp in dataset:
            assert type(dp) == DataPoint
            assert len(dp.spectrum.frequency) == 4
            assert len(dp.spectrum.impedance) == 4
            assert_array_almost_equal(dp.spectrum.frequency, freq, decimal=6)


# Test the save_dataset function of the Generator class
def test_Generator_save_dataset(tmpdir):

    freq = [0.1, 1., 10., 100.]
    cs = CircuitString("R0-p(R1,C0)-p(R2,C1)")
    p = {
        "R0": {"R0": Range(0.1, 1.0)},
        "p(C0,R0)": {"R0": Range(1.0, 10.0), "C0": Range(0.01, 1.0)},
    }

    gen = Generator(cs, p, steps_limit=4)

    try:
        gen.save_dataset(freq, folder=tmpdir)
    except:
        assert False, "Exception raised on save_dataset call"
    else:

        if len(listdir(tmpdir)) != 400:
            assert False, "wrong number of files found in the destination folder"

        for i in range(400):
            filename = f"simulation_{i}.json"
            if not isfile(join(tmpdir, filename)):
                assert False, f"File {filename} not found"


# Test the Generator properties
def test_Generator_properties():

    cs = CircuitString("R0-p(R1,C0)-p(R2,C1)")
    p = {
        "R0": {"R0": Range(0.1, 1.0)},
        "p(C0,R0)": {"R0": Range(1.0, 10.0), "C0": Range(0.01, 1.0)},
    }

    gen = Generator(cs, p, steps_limit=4)

    assert gen.number_of_steps == 4
    assert gen.number_of_simulations == 400
    assert str(gen.circuit) == "R0-p(R1,C0)-p(R2,C1)"
