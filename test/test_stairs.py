"""
test of sensible stair placement
"""
import yaml
import numpy as numpy

from roguecraft.roguecraft import *
from roguecraft.structures import *


def test_stair_placement():
    """ check that stair placement joins two rooms """
    legend = yaml.load(open('roguecraft/res/legend.yaml'), 
                       Loader=yaml.Loader)['legend']
    struct_cfg = yaml.load(open('roguecraft/res/structures.yaml'),
                           Loader=yaml.Loader)['structures']
    dims = Dimensions(2, 5, 30, 30)
    rc = RoomConstraints(8, 12, 2)

    db = DungeonBuilder(1680342140, dims, rc)
    db.build()
    db.write("build/test")
