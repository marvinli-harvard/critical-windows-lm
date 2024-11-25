"""
utils.py

Some basic utilities
"""


from utils.configuration import *


# String and Arithmetic Manipulations
def ceildiv(a : int , b : int) -> int: 
    ## https://stackoverflow.com/questions/14822184/is-there-a-ceiling-equivalent-of-operator-in-python
    return -(a // -b)

def num_to_chr(x : int) -> str:
    return f"({chr(65+x)})"

def add_parans(x : int) -> str:
    return f"({x})"

