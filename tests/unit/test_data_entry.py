import json, pytest
import matplotlib.pyplot as plt

from os.path import isfile, join
from numpy.testing import assert_array_almost_equal, assert_almost_equal
from ges_eis_toolbox.circuit.equivalent_circuit import EquivalentCircuit
from ges_eis_toolbox.spectra.spectrum import EIS_Spectrum
from ges_eis_toolbox.database.experiment import Instrument
from ges_eis_toolbox.database.data_entry import DataPoint, DataOrigin
from ges_eis_toolbox.exceptions import FileNotFound

# test consistency save operation
def test_DataPoint_save(tmpdir):
    
    # Initialize a DataPoint object
    start = DataPoint(
        DataOrigin.Real,
        user="tester",
        instrument=Instrument.BIOLOGIC,
        equivalent_circuit=EquivalentCircuit("R0-C0", parameters={"R0": 1., "C0": 0.1}),
        spectrum=EIS_Spectrum([0.1, 1., 10.], [1., 1.5, 1.2+0.5j])
    )

    # Save the datapoint to a file
    start.save("test", folder=tmpdir)
    
    # Check if the file has been created
    file_path = join(tmpdir, "test.json")
    assert isfile(file_path) == True

    json_data = {}
    with open(file_path, 'r') as file:
        json_data = json.load(file)
    
    for key, value in json_data.items():
        print(key, value)

    assert json_data["origin"] == "REAL"
    assert json_data["user"] == "tester"
    assert json_data["instrument"] == "BIOLOGIC"
    assert json_data["circuit"] == "R0-C0"
    assert_almost_equal(json_data["parameters"]["R0"], 1.0, decimal=6)
    assert_almost_equal(json_data["parameters"]['C0'], 0.1, decimal=6)
    assert_array_almost_equal(json_data["frequency"], [0.1, 1.0, 10.0], decimal=6)
    assert_array_almost_equal(json_data["real Z"], [1.0, 1.5, 1.2], decimal=6)
    assert_array_almost_equal(json_data["imag Z"], [0.0, 0.0, 0.5], decimal=6)


# test the failure of load on non existing file and non json files
def test_DataPoint_load_failure(tmpdir):
    file_path = join(tmpdir, "test.json")

    # Try to load a non existing file
    try:
        DataPoint.load(file_path)
    except FileNotFound:
        assert True
    else:
        assert False, "FileNotFound exception was not raised"
    
    # Create an invalid json file
    with open(file_path, "w") as file:
        file.write("this is not a .json file")
    
    # Try to load a non valid json file
    try:
        DataPoint.load(file_path)
    except json.decoder.JSONDecodeError:
        assert True
    else:
        assert False, "json.decoder.JSONDecodeError exception was not raised"
    

# test consistency between save and load operations
def test_DataPoint_save_load(tmpdir):
    
    # Initialize a DataPoint object
    start = DataPoint(
        DataOrigin.Real,
        user="tester",
        instrument=Instrument.BIOLOGIC,
        equivalent_circuit=EquivalentCircuit("R0-C0", parameters={"R0": 1., "C0": 0.1}),
        spectrum=EIS_Spectrum([0.1, 1., 10.], [1., 1.5, 1.2+0.5j])
    )

    # Save the datapoint to a file
    start.save("test", folder=tmpdir)
    
    # Check if the file has been created
    file_path = join(tmpdir, "test.json")
    assert isfile(file_path) == True

    # Load the file and create a new instance of the DataPoint class
    loaded = DataPoint.load(file_path)

    # Check that all the memeber values match
    assert loaded.origin == start.origin
    assert loaded.user == start.user
    assert loaded.instrument == start.instrument
    assert loaded.equivalent_circuit == start.equivalent_circuit
    assert_array_almost_equal(loaded.spectrum.frequency, start.spectrum.frequency, decimal=6)
    assert_array_almost_equal(loaded.spectrum.impedance, start.spectrum.impedance, decimal=6)
    

# test the execution of the bode_plot function without exceptions
def test_DataPoint_bode_plot():

    # Initialize a DataPoint object
    dp = DataPoint(
        DataOrigin.Real,
        user="tester",
        instrument=Instrument.BIOLOGIC,
        equivalent_circuit=EquivalentCircuit("R0-C0", parameters={"R0": 1., "C0": 0.1}),
        spectrum=EIS_Spectrum([0.1, 1., 10.], [1., 1.5, 1.2+0.5j])
    )

    # Initialize axes
    fig, (ax1, ax2) = plt.subplots(nrows=2)

    try:
        dp.bode_plot((ax1, ax2))
    except:
        assert False, "Exception raised on bode plot generation"
    else:
        assert True


# test the execution of the nyquist_plot function without exceptions
def test_DataPoint_nyquist_plot():

    # Initialize a DataPoint object
    dp = DataPoint(
        DataOrigin.Real,
        user="tester",
        instrument=Instrument.BIOLOGIC,
        equivalent_circuit=EquivalentCircuit("R0-C0", parameters={"R0": 1., "C0": 0.1}),
        spectrum=EIS_Spectrum([0.1, 1., 10.], [1., 1.5, 1.2+0.5j])
    )

    # Initialize axes
    fig, ax1 = plt.subplots()

    try:
        dp.nyquist_plot(ax1)
    except:
        assert False, "Exception raised on nyquist plot generation"
    else:
        assert True