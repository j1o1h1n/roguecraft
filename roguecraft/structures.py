"""
Load structure files.
"""
import logging
import typing
import math
import glob
import itertools
import numpy as np  # type: ignore
import yaml

logger = logging.getLogger(__name__)


def score_placement(dst, bp, p: tuple[int, int, int], 
                    normalise: bool=True) -> float:
    """
    Determine how close a fit the blueprint is to the destination location.

    This is calculated by counting how many points differ between the two
    matrices.
    """
    y, z, x = p
    h, l, w = bp.shape
    arr = dst[y:y+h,z:z+l,x:x+w]
    score = np.sum(np.equal(bp, arr))
    if not normalise:
        return score
    nf = np.sum(bp >= 0)
    n_score = score / nf
    return n_score


class Structure:

    def __init__(self, name, config, legend):
        self.name: str = name
        self.config: dict[str,typing.Any] = config
        self.tags: list[str] = self.config['tags']
        self.dimensions: tuple[int,int,int] = self.config['dimensions']
        # x,z offset when placing the structure
        self.offsets = [0, 0] # self.config.get('offsets', [0, 0])
        self.blueprint = self.parse_blueprint(self.config['blueprint'], legend)

    def parse_blueprint(self, blueprint_text, legend):
        # TODO is this really height, *length*, width? - y,z,x
        height, width, length = self.dimensions
        rows = blueprint_text.strip().split(' ')

        # check the plan has the correct dimensions
        if not sum((len(r) for r in rows)) == width * height * length \
            or not all((len(r) == length for r in rows)):
            raise Exception(f"expected blueprint to have dimensions {width}x{length}x{height}")

        # create a numpy blueprint from the text
        blueprint = np.array([-1] * (width * length * height), dtype=int)
        wl = width * length
        for y in range(height):
            # get the rows for this y-slice
            blueprint_slice = "".join([rows[i] for i in range(y, len(rows), height)])
            # find the offsets into the bleuprint
            p, q = y * wl, y * wl + wl
            # use the legend to map characters to integers and set the y slice
            # in the blueprint array
            blueprint[p:q] = [legend[c] for c in blueprint_slice]

        return blueprint.reshape(height, width, length)


class StructureBuilder:

    def __init__(self):
        self.legend = yaml.load(open('roguecraft/res/legend.yaml'), Loader=yaml.Loader)['legend']
        self.structures = yaml.load(open('roguecraft/res/structures.yaml'), Loader=yaml.Loader)['structures']
        self.tags = {}
        for name in self.structures:
            cfg = self.structures[name]
            tags = cfg.get('tags', [])
            for tag in tags:
                if tag not in self.tags:
                    self.tags[tag] = []
                self.tags[tag].append(name)

    def lookup_by_tag(self, tag):
        " return name of structures that have the given tag "
        return self.tags.get(tag, [])

    def build_structure(self, name: str, block_data, pos: tuple[int,int,int], jitter=0):
        logger.debug(f"build {name} at {pos}")
        y, x, z = pos
        structure = Structure(name, self.structures[name], self.legend)
        h, w, l = structure.dimensions
        dx, dz = structure.offsets
        x, z = x + dx, z + dz

        # expand the blueprint to the full dimensions
        bp = np.array([-1] * math.prod(block_data.shape), dtype=int).reshape(block_data.shape)

        # clip the blueprint when it extends past the top of the dungeon
        y_max = bp.shape[0]
        if y + h > y_max:
            h = y_max - y
            structure.blueprint = structure.blueprint[0:h]

        if jitter:
            # see if a 'better' location for the structure can be found nearby
            pos = y, z, x
            best = (-1.0, "", (0,0,0))
            for dz, dx in itertools.permutations(range(-jitter, jitter + 1), 2):
                y, z, x = pos
                pos1 = (y, z + dz, x + dx)
                s = score_placement(block_data, structure.blueprint, pos1)
                if s > best[0]:
                    best = (s, name, pos1)
            _, _, (y, z, x) = best

        bp[y:y+h,z:z+l,x:x+w] = structure.blueprint

        # used MaskedArray to copy into the block_data where the blueprint is not -1
        return np.ma.array(block_data, mask=(bp >= 0)).filled(bp)


if __name__ == "__main__":
    # simple test
    sb = StructureBuilder()
    sb.build_structure('nw_spiral_stair', None, (0,0,0))
