"""
Load structure files.
"""
import glob
import yaml
import numpy as np


class Structure:

    def __init__(self, name, config, legend):
        self.name = name
        self.config = config
        self.tags = self.config['tags']
        self.dimensions = self.config['dimensions']
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
        self.legend = yaml.load(open('structures/legend.yaml'), Loader=yaml.Loader)['legend']
        self.structures = yaml.load(open('structures/structures.yaml'), Loader=yaml.Loader)['structures']

    def build_structure(self, name, block_data, pos):
        x, y, z = pos
        structure = Structure(name, self.structures[name], self.legend)


if __name__ == "__main__":
    # simple test
    sb = StructureBuilder()
    sb.build_structure('nw_spiral_stair', None, (0,0,0))
