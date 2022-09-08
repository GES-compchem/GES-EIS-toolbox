from ges_eis_toolbox.utils import remove_numbers

# Test the remove number function
def test_remove_number():

    input = "ThisIsA0Test1String1234567890!"

    output = remove_numbers(input)
    assert output == "ThisIsATestString!"
    