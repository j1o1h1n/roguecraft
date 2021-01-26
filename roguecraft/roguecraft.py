#!/usr/bin/env python3
"""
Generate roguelike dungeons for minecraft.
"""
import argparse
import logging
import numpy as np
import json
import math
import random
import python_nbt.nbt as nbt
import typing

import roguecraft.structures as structures

logger = logging.getLogger()


def tsp(cities, rand=None):
    """
    Simulated annealing solver for traveling salesperson problem.  This is used
    to plot a sensible looking pathway from room to room on a level.

    cities:
        an array of x,y points

    Inspired by work by Eric Phanson.

    * https://ericphanson.com/blog/2016/the-traveling-salesman-and-10-lines-of-python/

    Example:

    >>> rand = random.Random(31415)
    >>> cities = numpy.array([rand.sample(range(100), 2) for x in range(10)], dtype=int)
    >>> tsp(cities, rand)
    [1, 0, 6, 3, 7, 4, 9, 8, 5, 2]
    """
    if rand is None:
        rand = random.Random()
    count = len(cities)
    # create a random tour
    tour = rand.sample(range(count),count)

    # build the distance matrix
    dist_matrix = np.zeros(count * count).reshape(count, count)
    for i in range(count):
        for j in range(count):
            dist_matrix[i][j] = np.linalg.norm(cities[i] - cities[j])

    def dist(t):
        return sum([dist_matrix[t[i]][t[i+1]] for i in range(len(t) - 1)])

    # check for improvement every 1000 steps after 20000
    steps = 0
    last_dist = dist(tour)

    for temperature in np.logspace(0,5,num=100000)[::-1]:
        # swap the positions of two random cities
        i,j = sorted(rand.sample(range(count),2))
        new_tour =  tour[:i] + tour[j:j+1] +  tour[i+1:j] + tour[i:i+1] + tour[j+1:]
        changes = [j,j-1,i,i-1]

        def delta_dist_matrix(t):
            return sum([dist_matrix[t[(k+1) % count]][t[(k) % count]] for k in changes])

        # only need to find the change for the swapped cities
        dist_tour = delta_dist_matrix(tour)
        dist_new_tour = delta_dist_matrix(new_tour)
        change = (dist_tour - dist_new_tour) / temperature

        if math.exp(change) > rand.random():
            tour = new_tour

        # stop when the delta distance falls below 0.01% over 1000 steps
        steps += 1
        if steps > 20000 and steps % 1000 == 0:
            d = dist(tour)
            if abs(d - last_dist) / d <= 0.0001:
                break
            last_dist = d

    return tour


class Rect:
    " a rectangle on the map. used to characterize a room "
    def __init__(self, x, y, w, h):
        self.x1 = x
        self.y1 = y
        self.x2 = x + w
        self.y2 = y + h
        self.w = w
        self.h = h
        self.tl = self.x1, self.y1
        self.tr = self.x2, self.y1
        self.bl = self.x1, self.y2
        self.br = self.x2, self.y2
        center_x = (self.x1 + self.x2) // 2
        center_y = (self.y1 + self.y2) // 2
        self.center = (center_x, center_y)

    def intersects(self, other):
        " returns true if this rectangle intersects with another one "
        return (self.x1 <= other.x2 and self.x2 >= other.x1 and
                self.y1 <= other.y2 and self.y2 >= other.y1)

    def intersection(self, other):
        " returns the intersection between this rectangle and another "
        if not self.intersects(other):
            return None
        x1, y1 = max(self.x1, other.x1), max(self.y1, other.y1)
        x2, y2 = min(self.x2, other.x2), min(self.y2, other.y2)
        return Rect(x1, y1, x2 - x1, y2 - y1)

    def __repr__(self):
        return f"<Rect {self.x1},{self.y1},{self.x2},{self.y2}>"

    @staticmethod
    def build_rect(x1, y1, x2, y2):
        # build a rect from two pairs of points that may not be correctly ordered
        if x1 >= x2:
            x1, x2 = x2, x1
        if y1 >= y2:
            y1, y2 = y2, y1
        return Rect(x1, y1, x2 - x1, y2 - y1)


class Room(Rect):

    def __init__(self, label, x, y, w, h):
        Rect.__init__(self, x, y, w, h)
        self.label = label


class Passage:

    def __init__(self, *rects):
        " a passage is a collection of connected rectangles "
        self.rects = rects

    def __add__(self, other):
        rects = self.rects + other.rects
        return Passage(*rects)

    def __repr__(self):
        return repr(self.rects)

    @staticmethod
    def connect_tsp(rand, *rooms, room_number_offset=0):
        """ connect the rooms with passages using tsp ordering """
        rects = []
        cities = np.array([r.center for r in rooms])
        tour = tsp(cities, rand)
        new_rooms = [rooms[t] for t in tour]
        rooms = new_rooms

        # rename the rooms
        for i, r in enumerate(rooms):
            r.label = str(i + 1 + room_number_offset)

        for i in range(len(rooms)):
            r1, r2 = rooms[i], rooms[(i+1) % len(rooms)]
            x1, y1 = r1.center
            x2, y2 = r2.center
            if rand.randint(0, 1):
                # horizontal and then vertical tunnel from centre to centre
                h = Rect.build_rect(x1, y1, x2, y1)
                v = Rect.build_rect(x2, y1, x2, y2)
            else:
                # vertical and then horizontal tunnel from centre to centre
                v = Rect.build_rect(x1, y1, x1, y2)
                h = Rect.build_rect(x1, y2, x2, y2)
            rects.append(h)
            rects.append(v)
        return Passage(*rects)

    @staticmethod
    def connect_partition(rand: random.Random, *rooms: Room):
        logger.debug(f"connecting {len(rooms)} rooms")
        def partition(axis, v, rects):
            if axis == 'x':
                n = 0
            elif axis == 'y':
                n = 0
            else:
                raise Exception("Expected x or y")
            return ([r for r in rects if r.center[n] <= v],
                    [r for r in rects if r.center[n] > v])

        min_x = min(r.x1 for r in rooms)
        max_x = max(r.x2 for r in rooms)
        min_y = min(r.y1 for r in rooms)
        max_y = max(r.y2 for r in rooms)

        # look for a nice partition
        best: typing.Tuple[typing.List[Room], typing.List[Room]]= [], []
        best_r = len(rooms)

        for x in range(min_x, max_x):
            a, b = partition('x', x, rooms)
            r = abs(len(a) - len(b))
            if r < best_r:
                best = a, b
                best_r = r
        for y in range(min_y, max_y):
            a, b = partition('y', y, rooms)
            r = abs(len(a) - len(b))
            if r < best_r:
                best = a, b
                best_r = r

        if len(best[0]) > 1 and len(best[1]) > 1:
            cycle1 = Passage.connect_tsp(rand, *best[0])
            cycle2 = Passage.connect_tsp(rand, *best[1], room_number_offset=len(best[0]))
            cycle3 = Passage.connect_ordered(rand, best[0][0], best[1][0])
            return cycle1 + cycle2 + cycle3
        else:
            return Passage.connect_tsp(rand, *rooms)

    @staticmethod
    def connect_ordered(rand, *rooms):
        """ connect the rooms with passages from center to center """
        rects = []
        for i in range(len(rooms) - 1):
            r1, r2 = rooms[i], rooms[i+1]
            x1, y1 = r1.center
            x2, y2 = r2.center
            if rand.randint(0, 1):
                # horizontal and then vertical tunnel from centre to centre
                h = Rect.build_rect(x1, y1, x2, y1)
                v = Rect.build_rect(x2, y1, x2, y2)
            else:
                # vertical and then horizontal tunnel from centre to centre
                v = Rect.build_rect(x1, y1, x1, y2)
                h = Rect.build_rect(x1, y2, x2, y2)
            rects.append(h)
            rects.append(v)
        return Passage(*rects)


class LevelBuilder:

    def __init__(self, level_width, level_height, seed=None, room_max_size=10,
                 room_min_size=6, max_rooms=30):
        if seed is None:
            self.seed = random.randint(0, 2 ** 32 - 1)
        else:
            self.seed = seed
        self.rand = random.Random(self.seed)
        self.level_width = level_width
        self.level_height = level_height
        self.room_max_size = room_max_size
        self.room_min_size = room_min_size
        self.max_rooms = max_rooms
        self.rooms = []
        self.passages = None

    def build(self):
        n = 1
        for _ in range(self.max_rooms):
            w = self.rand.randint(self.room_min_size, self.room_max_size)
            h = self.rand.randint(self.room_min_size, self.room_max_size)
            x = self.rand.randint(1, self.level_width - w - 2)
            y = self.rand.randint(1, self.level_height - h - 2)
            room = Room(str(n), x, y, w, h)
            intersection = (o for o in self.rooms if o.intersects(room))
            if any(intersection):
                continue
            n = n + 1
            self.rooms.append(room)

        self.passages = Passage.connect_partition(self.rand, *self.rooms)

        self.floor_plan = np.zeros(self.level_width * self.level_height, dtype=int) \
                            .reshape(self.level_height, self.level_width)

        for room in self.rooms:
            self.floor_plan[room.y1:room.y2, room.x1:room.x2] = 2

        for passage in self.passages.rects:
            self.floor_plan[passage.y1:passage.y2 + 1, passage.x1:passage.x2 + 1] = 2

        return self.floor_plan


def create_palette(d):
    palette = nbt.NBTTagCompound()
    for k in d:
        palette[f'minecraft:{k}'] = nbt.NBTTagInt(d[k])
    return palette, nbt.NBTTagInt(len(d))


class Dimensions:

    def __init__(self, levels, height, width, length):
        """
        Dungeon dimensions
          levels - number of levels in the dungeon.
          height - height of a dungeon level (y)
          width - level width (x)
          length - length of a dungeon level (z)
        """
        self.levels = levels
        self.height = height
        self.width = width
        self.length = length
        # convenience properties
        self.total_height = self.levels * self.height
        self.total_area = self.total_height * width * length
        self.measurements = (self.total_height, length, width)


class RoomConstraints:

    def __init__(self, min_size, max_size, max_rooms):
        self.min_size = min_size
        self.max_size = max_size
        self.max_rooms = max_rooms


class DungeonLevel:

    def __init__(self, level, rooms, passages, floor_plan):
        self.level = level
        self.rooms = rooms
        self.passages = passages
        self.floor_plan = floor_plan

    @staticmethod
    def build(rand: random.Random, level: int, dim: Dimensions, rc: RoomConstraints):
        rooms: typing.List[Room] = []
        for _ in range(rc.max_rooms):
            w = rand.randint(rc.min_size, rc.max_size)
            h = rand.randint(rc.min_size, rc.max_size)
            x = rand.randint(1, dim.width - w - 2)
            y = rand.randint(1, dim.height - h - 2)
            room = Room(str(len(rooms) + 1), x, y, w, h)
            intersection = (o for o in rooms if o.intersects(room))
            if any(intersection):
                continue
            rooms.append(room)

        passages = Passage.connect_partition(rand, *rooms)

        floor_plan = np.zeros(dim.width * dim.height, dtype=int) \
                              .reshape(dim.height, dim.width)

        for room in rooms:
            floor_plan[room.y1:room.y2, room.x1:room.x2] = 2

        for passage in passages.rects:
            floor_plan[passage.y1:passage.y2 + 1, passage.x1:passage.x2 + 1] = 2

        return DungeonLevel(level, rooms, passages, floor_plan)


class DungeonBuilder:
    " Build the entire dungeon. Write the dungeon to a schema file or a map. "
    def __init__(self, seed: int, dimensions: Dimensions, room_constraints: RoomConstraints):
        if seed is None:
            self.seed: int = random.randint(0, 2 ** 32 - 1)
        else:
            self.seed = seed
        self.rand = random.Random(self.seed)
        self.dimensions = dimensions
        self.room_constraints = room_constraints
        self.dungeon: nbt.NBTTagCompound = None
        self.block_data: np.array = None
        self.levels: typing.List[DungeonLevel] = []

    def create_template(self):
        dungeon = nbt.NBTTagCompound()
        dungeon['Version'] = nbt.NBTTagInt(2)
        dungeon['DataVersion'] = nbt.NBTTagInt(2584)
        dungeon['Width'] = nbt.NBTTagShort(self.dimensions.width)
        dungeon['Length'] = nbt.NBTTagShort(self.dimensions.length)
        dungeon['Height'] = nbt.NBTTagShort(self.dimensions.total_height)
        dungeon['BlockData'] = nbt.NBTTagList(tag_type_id=10)
        dungeon['BlockEntities'] = nbt.NBTTagByteArray()
        dungeon['Offset'] = nbt.NBTTagByteArray([0, 0, 0])

        # TODO handle palette better
        std_stairs = 'half=bottom,shape=straight,waterlogged=false'
        p, pm = create_palette({
            'stone': 0,
            'smooth_stone': 1,
            'air': 2,
            f'stone_brick_stairs[facing=north,{std_stairs}]': 3,
            f'stone_brick_stairs[facing=east,{std_stairs}]': 4,
            f'stone_brick_stairs[facing=south,{std_stairs}]': 5,
            f'stone_brick_stairs[facing=west,{std_stairs}]': 6,
        })
        dungeon['Palette'] = p
        dungeon['PaletteMax'] = pm
        self.dungeon = dungeon

        block_data = np.zeros(self.dimensions.total_area, dtype=int)
        self.block_data = block_data.reshape(*self.dimensions.measurements)

    def build(self):
        self.create_template()
        for level in range(self.dimensions.levels):
            logger.debug(f"building level {level+1} of {self.dimensions.levels}")
            self.build_level(level)
        self.build_stairs()

    def build_level(self, level: int):
        self.levels.append(DungeonLevel.build(self.rand, level, self.dimensions,
                                              self.room_constraints))

    def build_stairs(self):
        # TODO find places to put stairs to connect the levels
        pass

    def write(self, file_name: str):
        " write schematic file, metadata and ascii art map "
        self.write_schema(f"{file_name}.schem")
        self.write_map(f"{file_name}.map")
        self.write_schema(f"{file_name}.metadata")

    def write_schema(self, file_name):
        bd = [int(v) for v in self.block_data.reshape(math.prod(self.block_data.shape))]
        self.dungeon['BlockData'] = nbt.NBTTagByteArray(bd)
        nbt.write_to_nbt_file(file_name, self.dungeon)

    def write_map(self, file_name):
        pass

    def write_metadata(self, file_name):
        pass


def show_level(builder):
    print(f"""Dungeon(seed={builder.seed} width={builder.level_width}, """
          f"""height={builder.level_height}, room_max_sz={builder.room_max_size}, """
          f"""room_min_sz={builder.room_min_size},max_rooms={builder.max_rooms})""")
    h, w = builder.floor_plan.shape
    lines = []
    for y in range(h):
        line = [{0: '#', 2: '.', 3: '>'}[c] for c in builder.floor_plan[y]]
        lines.append(line)
    for room in builder.rooms:
        x, y = room.center
        line = lines[y]
        line = line[:x] + [room.label] + line[x + len(room.label):]
        lines[y] = line
    print("\n".join(["".join(l) for l in lines]))


def main(parser, args):
    dims = Dimensions(args.levels, args.width, args.length, args.height)
    constraints = RoomConstraints(args.min, args.max, args.rooms)

    builder = DungeonBuilder(args.seed, dims, constraints)
    builder.build()
    builder.write(args.name)
    # builder = LevelBuilder(width, length, seed=args.seed, room_min_size=args.min,
    #                        room_max_size=args.max, max_rooms=args.rooms)

    # floor_plan = builder.build()
    # for y in range(1, height - 1):
    #     block_data[y] = floor_plan

    # # make the room ceilings and floors smooth stone
    # block_data[-1][floor_plan == 2] = 1
    # block_data[0][floor_plan == 2] = 1

    # # add some stairs
    # room = builder.rooms[0]
    # sb = structures.StructureBuilder()

    # p1 = 1, room.y1, room.x1
    # p2 = 1, room.y1, room.x2
    # p3 = 1, room.y2, room.x1
    # p4 = 1, room.y2, room.x2
    # block_data = sb.build_structure('nw_spiral_stair', block_data, p1)
    # block_data = sb.build_structure('ne_spiral_stair', block_data, p2)
    # block_data = sb.build_structure('sw_spiral_stair', block_data, p3)
    # block_data = sb.build_structure('se_spiral_stair', block_data, p4)

    # fn = f"{args.name}.schem"
    # write_dungeon(fn, dungeon, block_data)

    # if args.debug:
    #     show_level(builder)


def parse_args():
    parser = argparse.ArgumentParser(usage="""
Create a dungeon level suitable for importing with worldedit.

Example:
    ./roguecraft.py -w 60 -l 60 -m 3 -M 20 -R 40 -D
""")
    parser.add_argument("-L", "--levels", type=int, required=True,
                        help="number of levels down the dungeon goes")
    parser.add_argument("-w", "--width", type=int, required=True,
                        help="dungeon width")
    parser.add_argument("-l", "--length", type=int, required=True,
                        help="dungeon length")
    parser.add_argument("--height", default=5, type=int, required=False,
                        help="dungeon height (min 4)")
    parser.add_argument("-m", "--min", default=5, type=int, required=True,
                        help="room minimum size")
    parser.add_argument("-M", "--max", default=14, type=int, required=True,
                        help="room maximum size")
    parser.add_argument("-R", "--rooms", default=30, type=int, required=True,
                        help="maximum number of rooms")
    parser.add_argument("--seed", default=None, type=int, required=False,
                        help='random seed (int)')
    parser.add_argument("-D", "--debug", action="store_true",
                        help="debug logging output")
    parser.add_argument("name", default="build/dungeon", nargs='?',
                        help="Schematic filename (default is build/dungeon)")
    args = parser.parse_args()

    level = logging.DEBUG if args.debug else logging.INFO
    format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=level, format=format)
    return parser, args


if __name__ == "__main__":
    parser, args = parse_args()
    main(parser, args)


