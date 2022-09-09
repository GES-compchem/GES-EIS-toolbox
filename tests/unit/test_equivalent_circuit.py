import traceback
import numpy as np
from numpy.testing import assert_array_almost_equal

from ges_eis_toolbox.exceptions import InvalidParametrization
from ges_eis_toolbox.circuit.circuit_string import CircuitString
from ges_eis_toolbox.circuit.equivalent_circuit import EquivalentCircuit

# Test that the EquivalentCircuit constructor runs without exceptions
def test_EquivalentCircuit___init__():

    cs = CircuitString("R0-p(C0,R1)-R2-C1")

    try:
        EquivalentCircuit(cs)
    except Exception:
        print(traceback.format_exc())
        assert False, f"Unexpected exception raised during EquivalentCircuit construction"
    else:
        assert True

    p = {"R0": 0.1, "C0": 0.01, "R1": 10, "R2": 0.001, "C1": 0.1}

    try:
        EquivalentCircuit(cs, parameters=p)
    except Exception:
        print(traceback.format_exc())
        assert False, f"Unexpected exception raised during EquivalentCircuit construction"
    else:
        assert True


# Test that the EquivalentCircuit constructor raises a TypeError exception when given wrong type
def test_EquivalentCircuit_TypeError___init__():

    try:
        EquivalentCircuit(4)
    except TypeError:
        assert True
    else:
        assert False, "An exception was expected from EquivalentCircuit constructor"


# Test that the EquivalentCircuit constructor raises an InvalidParametrization exception
def test_EquivalentCircuit_InvalidParametrization___init__():

    cs = CircuitString("R0-p(C0,R1)")

    wrong_label = {"R0": 0.1, "C0": 0.01, "R2": 10}
    wrong_type = {"R0": 0.1, "L0": 0.01, "R1": 10}
    wrong_number = {"R0": 0.1, "C0": 0.01}

    faulty = [wrong_label, wrong_type, wrong_number]

    for p in faulty:

        try:
            EquivalentCircuit(cs, parameters=p)
        except InvalidParametrization:
            assert True
        else:
            assert False, "An exception was expected from EquivalentCircuit constructor"


# Test the __getitem__ method of the EquivalentCircuit class
def test_EquivalentCircuit___getitem__():

    circ = EquivalentCircuit("R0-p(C0,R1)", parameters={"R0": 0.1, "C0": 0.01, "R1": 10})

    assert circ["R0"] == 0.1
    assert circ["C0"] == 0.01
    assert circ["R1"] == 10


# Test the __setitem__ method of the EquivalentCircuit class
def test_EquivalentCircuit___setitem__():

    circ = EquivalentCircuit("R0-p(C0,R1)", parameters={"R0": 0.1, "C0": 0.01, "R1": 10})
    circ["R0"] = 0.5
    assert circ["R0"] == 0.5


# Test the simulate function of the EquivalentCircuit class
def test_EquivalentCircuit_simulate():

    c = EquivalentCircuit("R0-p(C0,R1)", parameters={"R0": 0.1, "C0": 0.01, "R1": 10})
    result = c.simulate([0.1, 10, 1000, 100000])

    expected = np.array([
            10.06067682 - 6.25847783e-01j,
            0.34704523 - 1.55223096e00j,
            0.10002533 - 1.59154540e-02j,
            0.1 - 1.59154943e-04j,
        ])
    
    assert_array_almost_equal(result, expected, decimal=6)


# Test the EquivalentCircuit class properties
def test_EquivalentCircuit_properties():

    c = EquivalentCircuit("R0-p(C0,R1)", parameters={"R0": 0.1, "C0": 0.01, "R1": 10})

    assert type(c.circuit_string) == CircuitString
    assert c.circuit_string.value == "R0-p(C0,R1)"