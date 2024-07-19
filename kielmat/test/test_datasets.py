# import pytest
from kielmat.datasets import (
    keepcontrol,
    mobilised,
    fairpark
)
from kielmat.utils.kielmat_dataclass import KielMATRecording

def test_mobilised():
    assert mobilised.fetch_dataset() == None
    assert type(mobilised.load_recording()) == KielMATRecording

def test_keepcontrol():
    assert keepcontrol.fetch_dataset() == None
    assert type(keepcontrol.load_recording()) == KielMATRecording

# def test_fairpark():
#     # assert fairpark.fetch_dataset() == None
#     assert type(fairpark.load_recording()) == KielMATRecording