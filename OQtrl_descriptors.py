from abc import ABC, abstractmethod
from typing import Literal
from dataclasses import dataclass, field, fields, MISSING

#Type Validation abstract class
class Validator(ABC):

    def __set_name__(self, owner, name):
        self.private_name = '_' + name

    def __get__(self, obj, obj_type=None):
        return getattr(obj, self.private_name)

    def __set__(self, obj, value):
        self.validate(value)
        setattr(obj, self.private_name, value)

    @abstractmethod
    def validate(self, value):
        """Note: subclasses must implement this method"""


#Option Descriptor
class OneOf(Validator):
    
    def __init__(self, default = MISSING, *options):
    
        self.default = default
        self.options = set(options)

    def __get__(self, obj, obj_type=None):

        return getattr(obj, self.private_name, self.default)

    def validate(self, value):
        if self.default is None and value is None:
            return
        
        if value not in self.options:
            raise ValueError(f'Expected {value!r} to be one of {self.options!r}')

#Bitstring descriptor
class bit_string(Validator):

    def __init__(self, default = None, minsize=None, maxsize=None):
        
        self.default = default
        self.minsize = minsize
        self.maxsize = maxsize

    def __get__(self, obj, obj_type=None):
        return getattr(obj, self.private_name, self.default)

    def validate(self, value):
        
        if self.default is None and value is None:
            return

        if not isinstance(value, str):
            raise TypeError(f'Expected {value!r} to be an str')
        
        if not all(char in '01' for char in value):
            raise ValueError("Bitpattern must be a string of 0's and 1's")
        
        if self.minsize is not None and len(value) < self.minsize:
            raise ValueError(
                f'Expected {value!r} to be no smaller than {self.minsize!r}'
            )
        if self.maxsize is not None and len(value) > self.maxsize:
            raise ValueError(
                f'Expected {value!r} to be no bigger than {self.maxsize!r}'
            )

class cond_real(Validator):

    def __init__(self, default = None, types =None ,minvalue:int=None, maxvalue:int=None, predicate=None):
        
        self.default = default
        self.allowed = set([int,float])
        if types is not None and types not in self.allowed:
            raise TypeError(f'Types should be one of {self.allowed}')
        self.types = types
        self.minvalue = minvalue
        self.maxvalue = maxvalue
        self.predicate = predicate
    
    def __get__(self, obj, obj_type=None):
        return getattr(obj, self.private_name, self.default)
    
    def validate(self, value):
        #if value is None, pass the validation
        if value is None:
            return

        if not isinstance(value, self.types):
            raise TypeError(f'Expected {value!r} to be an {self.types}')
        
        if self.minvalue is not None and value < self.minvalue:
            raise ValueError(
                f'Expected {value!r} to be no smaller than {self.minvalue!r}'
            )
        if self.maxvalue is not None and value > self.maxvalue:
            raise ValueError(
                f'Expected {value!r} to be no bigger than {self.maxvalue!r}'
            )
        
        if self.predicate is not None and not self.predicate(value):
            raise ValueError(
                f'Expected {self.predicate!r} to be True for {value!r}'
            )