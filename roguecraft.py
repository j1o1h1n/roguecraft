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
                    h = Passage(x1, y1, x2, y1 + 1)
                    v = Passage(x2, y1, x2 + 1, y2)
                else:
                    # vertical and then horizontal tunnel
                    v = Passage(x1, y1, x1 + 1, y2)
                    h = Passage(x1, y2, x2, y2 + 1)
                self.passages.append(h)
                self.passages.append(v)

            self.rooms.append(room)

        arr = np.zeros(self.level_width * self.level_height, dtype=int) \
                .reshape(self.level_height, self.level_width)

        for room in self.rooms:
            arr[room.y1:room.y2, room.x1:room.x2] = 2

        for passage in self.passages:
            arr[passage.y1:passage.y2, passage.x1:passage.x2] = 2

        return arr


def show_level(arr):
    h, w = arr.shape
    for y in range(h):
        line = [{0: '#', 2: '.'}[c] for c in arr[y]]
        print("".join(line))


def create_template(width, length, height):
    dungeon = nbt.read_from_nbt_file("dungeon.schem")
    dungeon['Length'] = nbt.NBTTagShort(length)
    dungeon['Height'] = nbt.NBTTagShort(height)
    dungeon['Width'] = nbt.NBTTagShort(width)
    block_data = np.zeros(width * length * height, dtype=int).reshape(height, length, width)
    return dungeon, block_data


def write_dungeon(name, dungeon, block_data):
    bd = [int(v) for v in block_data.reshape(math.prod(block_data.shape))]

    dungeon['BlockData'] = nbt.NBTTagByteArray(bd)

    nbt.write_to_nbt_file('template.schem', dungeon)


def main(parser, args):
    items = json.loads(open('items.json').read())
    level = {
        'rooms': [],
        'corridors': [],
        'stairs': [],
    }

    width, length, height = 40, 60, 5
    dungeon, block_data = create_template(width, length, height)
    # set floor and ceiling to be all stone
    block_data[0] = 0
    # FIXME make the ceiling air for the moment
    block_data[-1] = 2

    builder = LevelBuilder(width, length, room_min_size=4, room_max_size=10,
                           max_rooms=30)
    outline = builder.build()
    show_level(outline)
    for y in range(1, height - 1):
        block_data[y] = outline

    write_dungeon("dungeon.schem", dungeon, block_data)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-D", "--debug", help="debug logging output", 
                        action="store_true")
    args = parser.parse_args()

    level = logging.DEBUG if args.debug else logging.INFO
    format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=level, format=format)
    return parser, args


if __name__ == "__main__":
    parser, args = parse_args()
    main(parser, args)


