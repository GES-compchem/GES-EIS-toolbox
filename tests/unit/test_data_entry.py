from os.path import isfile, join
from numpy.testing import assert_array_almost_equal
from ges_eis_toolbox.circuit.equivalent_circuit import EquivalentCircuit
from ges_eis_toolbox.spectra.spectrum import EIS_Spectrum
from ges_eis_toolbox.database.experiment import Instrument
from ges_eis_toolbox.database.data_entry import DataPoint, DataOrigin

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
    
