import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from rope_skip.geometry import parse_line, parse_floor_lines, estimate_vanishing_point

def test_parse_line():
    assert parse_line([[1,2],[3,4]]) == ((1.0,2.0),(3.0,4.0))

def test_parse_floor_lines():
    assert len(parse_floor_lines([[[0,0],[10,10]]])) == 1

def test_vanishing_point_empty():
    assert estimate_vanishing_point([]) is None
