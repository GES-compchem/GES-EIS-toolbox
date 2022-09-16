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


# Test the __eq__ method of the EquivalentCircuit class
def test_EquivalentCircuit___eq__():

    a = EquivalentCircuit("R0-p(C0,R1)", parameters={"R0": 0.1, "C0": 0.01, "R1": 10})
    b = EquivalentCircuit("R0-p(C0,R1)", parameters={"R0": 0.1, "C0": 0.01, "R1": 10})
    c = EquivalentCircuit("R1-p(C0,R0)", parameters={"R0": 10, "C0": 0.01, "R1": 0.1})

    assert a==b
    assert not a==c

# Test the __ne__ method of the EquivalentCircuit class
def test_EquivalentCircuit___ne__():

    a = EquivalentCircuit("R0-p(C0,R1)", parameters={"R0": 0.1, "C0": 0.01, "R1": 10})
    b = EquivalentCircuit("R0-p(C0,R1)", parameters={"R0": 0.1, "C0": 0.01, "R1": 10})
    c = EquivalentCircuit("R1-p(C0,R0)", parameters={"R0": 10, "C0": 0.01, "R1": 0.1})

    assert not a!=b
    assert a!=c


# Test the reorder method of the EquivalentCircuit class
def test_EquivalentCircuit_reorder():

    parameters = {"C0": 1, "C2": 2, "L2": 3, "R0": 4, "R1": 5}
    circ = EquivalentCircuit("R1-p(C0,R0)-L2-C2", parameters=parameters)

    circ.reorder()
    assert circ.circuit_string.value == "C0-L0-R0-p(C1,R1)"
    assert circ.parameters == {"C1": 1, "C0": 2, "L0": 3, "R1": 4, "R0": 5}


# Test the simulate function of the EquivalentCircuit class
def test_EquivalentCircuit_simulate():

    c = EquivalentCircuit("R0-p(C0,R1)", parameters={"R0": 0.1, "C0": 0.01, "R1": 10})
    result = c.simulate([0.1, 10, 1000, 100000])

    expected = np.array(
        [
            10.06067682 - 6.25847783e-01j,
            0.34704523 - 1.55223096e00j,
            0.10002533 - 1.59154540e-02j,
            0.1 - 1.59154943e-04j,
        ]
    )

    assert_array_almost_equal(result, expected, decimal=6)


# Test the simulate function of the EquivalentCircuit class with components defined by 2 or more parameters
def test_EquivalentCircuit_simulate_multiple_parameters():

    c = EquivalentCircuit(
        "R0-p(CPE0,R1)", parameters={"R0": 0.1, "CPE0": [0.01, 1.2], "R1": 10}
    )
    result = c.simulate([0.1, 10, 1000, 100000])

    expected = np.array(
        [
            10.248929 - 5.625940e-01j,
            -0.073121 - 6.875082e-01j,
            0.099145 - 2.633084e-03j,
            0.099997 - 1.048071e-05j,
        ]
    )

    assert_array_almost_equal(result, expected, decimal=6)


# Test the EquivalentCircuit class properties
def test_EquivalentCircuit_properties():

    c = EquivalentCircuit("R0-p(C0,R1)", parameters={"R0": 0.1, "C0": 0.01, "R1": 10})

    assert type(c.circuit_string) == CircuitString
    assert c.circuit_string.value == "R0-p(C0,R1)"

    assert type(c.parameters) == dict
    assert c.parameters == {"R0": 0.1, "C0": 0.01, "R1": 10}
