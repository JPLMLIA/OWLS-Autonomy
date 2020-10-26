import pytest

from cli import TOGA_wrapper

class TestTogaWrapper:
    '''Test toga wrapper functions'''
    
    def test_override_config(self):
        d = {'a' : 1, 'b' : 2, 'c' : {'ca' : 31, 'cb': 32, 'cc': 33}, 'd' : 4}
        n = {'a' : 5, 'e' : 7, 'c' : {'ca' : 0, 'd' : 7}}
        t = {'a' : 5, 'b' : 2, 'c' : {'ca' : 0, 'cb': 32, 'cc': 33, 'd' : 7}, 
             'd' : 4, 'e' : 7}
        assert(TOGA_wrapper.get_override_config(d, n) == t)