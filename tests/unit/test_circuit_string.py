import traceback

from ges_eis_toolbox.exceptions import InvalidSyntax, InvalidComponent
from ges_eis_toolbox.circuit.circuit_string import CircuitString


# Test that the CircuitString constructor runs without exceptions
def test_CircuitString___init__():

    string = "R0-p(C0,R1)-R2-C1"

    try:
        CircuitString(string)
    except Exception:
        print(traceback.format_exc())
        assert False, f"Unexpected exception raised during CircuitString construction"
    else:
        assert True


# Test that the CircuitString constructor returns exception on wrong input type
def test_CircuitString___init___TypeError():

    not_a_stirng = 1

    try:
        CircuitString(not_a_stirng)
    except TypeError:
        assert True
    else:
        assert False, f"A TypeError exception was expected from CircuitString constructor"


# Test the __str__ and __repr__ metods of the CircuitString class
def test_CircuitString___str____repr__():

    string = "R0-p(C0,R1)-R2-C1"
    obj = CircuitString(string)

    assert str(obj) == string
    assert repr(obj) == string


# Test the __add__ method of the CircuitString class
def test_CircuitString___add__():

    a = CircuitString("R0-p(C0,R1)")
    b = CircuitString("R2-C1")

    try:
        c = a + b
    except Exception:
        print(traceback.format_exc())
        assert False, f"Unexpected exception raised during CircuitString addition (+)"
    else:
        assert True

    assert type(c) == CircuitString
    assert c.value == "R0-p(C0,R1)-R2-C1"


# Test the __iadd__ method of the CircuitString class
def test_CircuitString___iadd__():

    a = CircuitString("R0-p(C0,R1)")
    b = CircuitString("R2-C1")

    try:
        a += b
    except Exception:
        print(traceback.format_exc())
        assert False, f"Unexpected exception raised during CircuitString self addition (+=)"
    else:
        assert True

    assert type(a) == CircuitString
    assert a.value == "R0-p(C0,R1)-R2-C1"
    assert b.value == "R2-C1"


# Test the validation function of the CircuitString class
def test_CircutString__validate():

    normal = CircuitString("R0-p(R1,C0)")
    try:
        normal._validate()
    except:
        print(traceback.format_exc())
        assert False, f"Unexpected exception raised during CircuitString validation"
    else:
        assert True

    invalid_syntax = [
        CircuitString("R0-p(R1, C0)"),
        CircuitString("R0-(R1,C0)"),
        CircuitString("R0-p(R1,C0))"),
    ]

    for obj in invalid_syntax:
        try:
            obj._validate()
        except InvalidSyntax:
            assert True
        else:
            assert (
                False
            ), f"An InvalidSyntax exception was expected from CircuitString validation"

    invalid_component = CircuitString("R0-p(R1,X0)")
    try:
        invalid_component._validate()
    except InvalidComponent:
        assert True
    else:
        assert (
            False
        ), f"An InvalidComponent exception was expected from CircuitString validation"


# Test the decompose_series function of the CircuitString class
def test_CircuitString_decompose_series():

    string = "R0-p(C0,R1)-R2-C1"
    obj = CircuitString(string)

    blocks = obj.decompose_series()

    for cs in blocks:
        assert type(cs) == CircuitString

    assert [cs.value for cs in blocks] == ["R0", "p(C0,R1)", "R2", "C1"]


# Test the list_components function of the CircuitString class
def test_CircuitString_list_components():

    string = "R0-p(C0,R1)-R2-C1"
    obj = CircuitString(string)

    components = obj.list_components()

    for s in components:
        assert type(s) == str

    assert components == ["C0", "C1", "R0", "R1", "R2"]


# Test the list_components function of the CircuitString class with the unsorted option
def test_CircuitString_list_components_unsorted():

    string = "R0-p(C0,R1)-R2-C1"
    obj = CircuitString(string)

    components = obj.list_components(sort=False)

    for s in components:
        assert type(s) == str

    assert components == ["R0", "C0", "R1", "R2", "C1"]


# Test the remove_numbers function of the CircuitString class
def test_CircuitString_remove_numbers():

    string = "R0-p(C0,R1)-R2-C1"
    obj = CircuitString(string)

    cs = obj.remove_numbers()

    assert type(cs) == CircuitString
    assert cs.value == "R-p(C,R)-R-C"


# Test the reorder function of the CircuitString class
def test_CircuitString_reorder():

    string = "R0-p(R1,C0)-R2-C1-p(R3,p(L1,C2))"
    obj = CircuitString(string)

    reordered = obj.reorder()

    assert type(reordered) == CircuitString
    assert reordered.value == "C1-R0-R2-p(C0,R1)-p(R3,p(C2,L1))"


# Test the reorder_labels function of the CircuitString class
def test_CircuitString_reorder_labels():

    string = "R1-p(R0,C0)-R2-C1-p(R4,p(R3,C2))"
    obj = CircuitString(string)

    reordered, conversion = obj.reorder_labels()

    assert type(reordered) == CircuitString
    assert reordered.value == "R0-p(R1,C0)-R2-C1-p(R3,p(R4,C2))"

    assert type(conversion) == dict
    assert conversion == {
        "R1": "R0",
        "R0": "R1",
        "C0": "C0",
        "R2": "R2",
        "C1": "C1",
        "R3": "R4",
        "R4": "R3",
        "C2": "C2",
    }


# Test the CircuitString properties
def test_CircuitString_properties():

    string = "R0-p(C0,R1)-R2-C1"
    obj = CircuitString(string)

    assert obj.value == string
    assert obj.number_of_components == 5


# Test the is_simple_series property
def test_CircuitString_is_simple_series():

    obj_1 = CircuitString("R0-p(C0,R1)-R2-C1")
    obj_2 = CircuitString("R0-C0-R1-R2-C1")

    assert obj_1.is_simple_series == False
    assert obj_2.is_simple_series == True


# Test the is_pure_parallel proprty
def test_CircuitString_is_pure_parallel():

    obj_f1 = CircuitString("R0-p(C0,R1)-R2-C1")
    obj_f2 = CircuitString("p(C0,p(R1, L1))-p(C1, R2)")

    assert obj_f1.is_pure_parallel == False
    assert obj_f2.is_pure_parallel == False

    obj_t1 = CircuitString("p(C0,R1)")
    obj_t2 = CircuitString("p(C0,p(R1, L1))")
    assert obj_t1.is_pure_parallel == True
    assert obj_t2.is_pure_parallel == True
