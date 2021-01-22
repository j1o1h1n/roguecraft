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

logger = logging.getLogger()


def compass_rose(x, z, posn='ne'):
    """
    Return the compass points around the x,z coordinates
    """
    if posn == 'se':
        x, z = x - 1, z - 1
    elif posn == 'sw':
        x, z = x, z - 1
    elif posn == 'ne':
        x, z = x - 1, z
    elif posn == 'nw':
        pass
    else:
        raise Error(f"unexpected position {posn}")
    
    se = z + 1, x + 1
    ne = z, x + 1
    nw = z, x
    sw = z + 1, x
    return {'ne': ne, 'nw': nw, 'se': se, 'sw': sw}

class Rect:
    " a rectangle on the map. used to characterize a room "
    def __init__(self, x, y, w, h):
        self.x1 = x
        self.y1 = y
        self.x2 = x + w
        self.y2 = y + h
 
    def center(self):
        center_x = (self.x1 + self.x2) // 2
        center_y = (self.y1 + self.y2) // 2
        return (center_x, center_y)
 
    def intersect(self, other):
        " returns true if this rectangle intersects with another one "
        return (self.x1 <= other.x2 and self.x2 >= other.x1 and
                self.y1 <= other.y2 and self.y2 >= other.y1)

    def __repr__(self):
        return f"<Room {self.x1},{self.y1},{self.x2},{self.y2}>"


class Passage(Rect):
    " a horizontal or vertical line "
    def __init__(self, x1, y1, x2, y2):
        if x1 >= x2:
            x1, x2 = x2, x1
        if y1 >= y2:
            y1, y2 = y2, y1
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2

    def __repr__(self):
        return f"<Passage {self.x1},{self.y1},{self.x2},{self.y2}>"


class Stairs:
    " a spiral stair up "
    def __init__(self, room):
        self.room = room


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
        self.passages = []
        self.stairs = []

    def build(self):
        for _ in range(self.max_rooms):
            w = self.rand.randint(self.room_min_size, self.room_max_size)
            h = self.rand.randint(self.room_min_size, self.room_max_size)
            x = self.rand.randint(1, self.level_width - w - 2)
            y = self.rand.randint(1, self.level_height - h - 2)
            room = Rect(x, y, w, h)
            intersection = (o for o in self.rooms if o.intersect(room))
            if any(intersection):
                continue
            x1, y1 = room.center()

            # connect to the previous room
            if self.rooms:
                x2, y2 = self.rooms[-1].center()

                if self.rand.randint(0, 1):
                    # horizontal and then vertical tunnel
                    h = Passage(x1, y1, x2, y1)
                    v = Passage(x2, y1, x2, y2)
                else:
                    # vertical and then horizontal tunnel
                    v = Passage(x1, y1, x1, y2)
                    h = Passage(x1, y2, x2, y2)
                self.passages.append(h)
                self.passages.append(v)

            self.rooms.append(room)

        # add stairs up
        r = self.rand.choice(self.rooms)
        self.stairs.append(Stairs(r))
        logger.debug(f"stair at {r}")
        r = self.rand.choice(self.rooms)
        self.stairs.append(Stairs(r))
        logger.debug(f"stair at {r}")

        self.outline = np.zeros(self.level_width * self.level_height, dtype=int) \
                .reshape(self.level_height, self.level_width)

        for room in self.rooms:
            self.outline[room.y1:room.y2, room.x1:room.x2] = 2

        for passage in self.passages:
            # logger.debug(f"passage: {passage} - [{passage.y1}:{passage.y2 + 1}, {passage.x1}:{passage.x2 + 1}]")
            self.outline[passage.y1:passage.y2 + 1, passage.x1:passage.x2 + 1] = 2

        return self.outline


    # TODO - generalise to stairs in any direction, look for free space to build
    def build_stairs(self, block_data):
        # stair block ids
        NORTH, EAST, SOUTH, WEST = 3, 4, 5, 6
        SPIRAL_STAIRS = {
            # ##  #<  <.  ..
            # #^  #.  #.  v.
            "nw_ccw_spiral": (['se', 'ne', 'nw', 'sw'], [NORTH, WEST, WEST, SOUTH]),
            # ##  >#  .>  ..
            # ^#  .#  .#  .v
            "ne_ccw_spiral": (['sw', 'nw', 'ne', 'se'], [NORTH, EAST, EAST, SOUTH]),
            # v#  .#  .#  .^
            # ##  >#  .>  ..
            "se_ccw_spiral": (['nw', 'sw', 'se', 'ne'], [SOUTH, EAST, EAST, NORTH]),
            # #v  #.  #.  ^.
            # ##  #<  <.  ..
            "sw_ccw_spiral": (['ne', 'se', 'sw', 'nw'], [SOUTH, WEST, WEST, NORTH]),
        }

        def set_bricks(y, points, *blocks):
            for (z, x), b in zip(points, blocks):
                block_data[y + 1][z][x] = b

        def build_stair(spiral_directions, steps, compass):
            spiral = [compass[d] for d in spiral_directions]
            block = 0
            air = 2
            for i in range(4):
                blocks = [air] * i + [steps[i]] + [block] * (3 - i)
                set_bricks(i, spiral, *blocks)


        stair = self.stairs[0]

        # top left corner stair 
        x, z = stair.room.x1, stair.room.y1
        compass = compass_rose(x - 1, z, 'nw')
        spiral, steps = SPIRAL_STAIRS['nw_ccw_spiral']
        build_stair(spiral, steps, compass)

        # # top right corner stair 
        x, z = stair.room.x2, stair.room.y1
        compass = compass_rose(x, z, 'ne')
        spiral, steps = SPIRAL_STAIRS['ne_ccw_spiral']
        build_stair(spiral, steps, compass)

        # bottom left corner stair 
        x, z = stair.room.x1 - 1, stair.room.y2
        compass = compass_rose(x, z, 'sw')
        spiral, steps = SPIRAL_STAIRS['sw_ccw_spiral']
        build_stair(spiral, steps, compass)

        # bottom right corner stair
        x, z = stair.room.x2, stair.room.y2
        compass = compass_rose(x, z, 'se')
        spiral, steps = SPIRAL_STAIRS['se_ccw_spiral']
        build_stair(spiral, steps, compass)

        stair = self.stairs[1]
        # in the middle of the room
        x, z = (stair.room.x1 + stair.room.x2) // 2, (stair.room.y1 + stair.room.y2) // 2
        compass = compass_rose(x, z, 'nw')
        spiral, steps = SPIRAL_STAIRS['nw_ccw_spiral']
        build_stair(spiral, steps, compass)



def show_level(builder):
    print(f"""Dungeon(seed={builder.seed} width={builder.level_width}, """
          f"""height={builder.level_height}, room_max_sz={builder.room_max_size}, """
          f"""room_min_sz={builder.room_min_size},max_rooms={builder.max_rooms})""")
    h, w = builder.outline.shape
    for stair in builder.stairs:
        x, y = stair.room.x1, stair.room.y1
        builder.outline[y][x] = 3
        builder.outline[y][x+1] = 3
        builder.outline[y+1][x+1] = 3
        builder.outline[y+1][x] = 3
    for y in range(h):
        line = [{0: '#', 2: '.', 3: '>'}[c] for c in builder.outline[y]]
        print("".join(line))


def load_schematic(filename):
    return nbt.read_from_nbt_file(filename)


def create_template(width, length, height):
    dungeon = nbt.NBTTagCompound()
    dungeon['Version'] = nbt.NBTTagInt(2)
    dungeon['DataVersion'] = nbt.NBTTagInt(2584)
    dungeon['Width'] = nbt.NBTTagShort(width)
    dungeon['Length'] = nbt.NBTTagShort(length)
    dungeon['Height'] = nbt.NBTTagShort(height)
    dungeon['BlockData'] = nbt.NBTTagList(tag_type_id=10)
    dungeon['BlockEntities'] = nbt.NBTTagByteArray()

    # TODO handle palette better
    dungeon['Palette'] = nbt.NBTTagCompound()
    dungeon['Palette']['minecraft:stone'] = nbt.NBTTagInt(0)
    dungeon['Palette']['minecraft:smooth_stone'] = nbt.NBTTagInt(1)
    dungeon['Palette']['minecraft:air'] = nbt.NBTTagInt(2)
    dungeon['Palette']['minecraft:stone_brick_stairs[facing=north,half=bottom,shape=straight,waterlogged=false]'] = nbt.NBTTagInt(3)
    dungeon['Palette']['minecraft:stone_brick_stairs[facing=east,half=bottom,shape=straight,waterlogged=false]'] = nbt.NBTTagInt(4)
    dungeon['Palette']['minecraft:stone_brick_stairs[facing=south,half=bottom,shape=straight,waterlogged=false]'] = nbt.NBTTagInt(5)
    dungeon['Palette']['minecraft:stone_brick_stairs[facing=west,half=bottom,shape=straight,waterlogged=false]'] = nbt.NBTTagInt(6)
    dungeon['PaletteMax'] = nbt.NBTTagInt(7)

    block_data = np.zeros(width * length * height, dtype=int).reshape(height, length, width)

    return dungeon, block_data


def write_dungeon(name, dungeon, block_data):
    bd = [int(v) for v in block_data.reshape(math.prod(block_data.shape))]

    dungeon['BlockData'] = nbt.NBTTagByteArray(bd)

    nbt.write_to_nbt_file(name, dungeon)


def main(parser, args):
    items = json.loads(open('items.json').read())

    width, length, height = args.width, args.length, args.height
    dungeon, block_data = create_template(width, length, height)
    # set floor and ceiling to be all stone
    block_data[0] = 0
    block_data[-1] = 0

    builder = LevelBuilder(width, length, seed=args.seed, room_min_size=args.min,
                           room_max_size=args.max, max_rooms=args.rooms)
    outline = builder.build()
    for y in range(1, height - 1):
        block_data[y] = outline

    # make the room ceilings smooth stone
    block_data[-1][outline == 2] = 1

    # add stairs
    builder.build_stairs(block_data)

    write_dungeon(f"{args.name}.schem", dungeon, block_data)

    if args.debug:
        show_level(builder)


def parse_args():
    parser = argparse.ArgumentParser(usage="""
Create a dungeon level suitable for importing with worldedit.

Example:
    ./roguecraft.py -w 60 -l 60 -m 3 -M 20 -R 40 -D 
""")
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
    parser.add_argument("name", default="dungeon", nargs='?',
                        help="Filename for the schema (default is dungeon)")
    args = parser.parse_args()

    level = logging.DEBUG if args.debug else logging.INFO
    format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=level, format=format)
    return parser, args


if __name__ == "__main__":
    parser, args = parse_args()
    main(parser, args)


