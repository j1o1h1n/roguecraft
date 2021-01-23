# Roguecraft

Generate rougelike dungeons for minecraft.

## Installation

This is a simple command line program.  The following libraries must be installed.

```
pip install numpy Python-NBT
```

## Usage

```
Create a dungeon level suitable for importing with worldedit.

Example:
    ./roguecraft.py -w 60 -l 60 -m 3 -M 20 -R 40 -D 

positional arguments:
  name                  Filename for the schema (default is dungeon)

optional arguments:
  -h, --help            show this help message and exit
  -w WIDTH, --width WIDTH
                        dungeon width
  -l LENGTH, --length LENGTH
                        dungeon length
  --height HEIGHT       dungeon height (min 4)
  -m MIN, --min MIN     room minimum size
  -M MAX, --max MAX     room maximum size
  -R ROOMS, --rooms ROOMS
                        maximum number of rooms
  --seed SEED           random seed (int)
  -D, --debug           debug logging output
```

## Example

Create a 60x60 dungeon with rooms of the default height, rooms from 3...20 height and width, with a maximum of 20 rooms per level. 

```
./roguecraft.py -w 60 -l 60 -m 3 -M 20 -R 40
```

Then copy the generated **dungeon.schem** to the minecraft/config/WorldEdit/schematics directory.

From inside the game use the WorldEdit commands

```
//schem load dungeon
//paste
```


## NBT Notes

| ID | Name | Description |
|--|--|
| 0  | End        | None. | 
| 1  | Byte       | A single signed byte (8 bits) | 
| 2  | Short      | A signed short (16 bits, big endian) | 
| 3  | Int        | A signed short (32 bits, big endian) | 
| 4  | Long       | A signed long (64 bits, big endian) | 
| 5  | Float      | A floating point value (32 bits, big endian, IEEE 754-2008, binary32) | 
| 6  | Double     | A floating point value (64 bits, big endian, IEEE 754-2008, binary64) | 
| 7  | Byte_Array | TAG_Int length  | 
| 8  | String     | TAG_Short length  | 
| 9  | List       | TAG_Byte tagId |
| 10 | Compound   | A sequential list of Named Tags. This array keeps going until a TAG_End is found. |

* https://minecraft.gamepedia.com/NBT_format

### NBT Example

* https://github.com/TowardtheStars/Python-NBT

```
>>> import python_nbt.nbt as nbt
>>> file = nbt.read_from_nbt_file("rail-1000.schem")
>>> file
{'type_id': 10, 'value': {
    'Version': {'type_id': 3, 'value': 2}, 
    'DataVersion': {'type_id': 3, 'value': 2584}, 
    'Metadata': {'type_id': 10, 'value': {
        'WEOffsetX': {'type_id': 3, 'value': -2}, 
        'WEOffsetY': {'type_id': 3, 'value': -1}, 
        'WEOffsetZ': {'type_id': 3, 'value': 1}}
    }, 
    'Offset': {'type_id': 11, 'value': [-17, 236, -22]},
    'Length': {'type_id': 2, 'value': 40}, 
    'Height': {'type_id': 2, 'value': 5}, 
    'Width': {'type_id': 2, 'value': 5}, 
    'PaletteMax': {'type_id': 3, 'value': 8}, 
    'Palette': {'type_id': 10, 'value': {
        'minecraft:stone': {'type_id': 3, 'value': 0}, 
        'minecraft:smooth_stone': {'type_id': 3, 'value': 1}, 
        'minecraft:air': {'type_id': 3, 'value': 2}, 
        'minecraft:rail[shape=north_south]': {'type_id': 3, 'value': 3}},
        'minecraft:powered_rail[powered=true,shape=north_south]': {'type_id': 3, 'value': 4}, 
        'minecraft:redstone_wire[east=side,north=none,power=15,south=none,west=side]': {'type_id': 3, 'value': 5},
        'minecraft:redstone_torch[lit=true]': {'type_id': 3, 'value': 6}, 
        'minecraft:jack_o_lantern[facing=north]': {'type_id': 3, 'value': 7}
    }, 
    'BlockData': {'type_id': 7, 'value': [0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, ...
1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 7, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0]}, 
    'BlockEntities': {'type_id': 9, 'value': [], 'tag_type_id': 10}
    }
}1``
```

## Block Data

Specifies the main storage array which contains Width * Height * Length entries. Each entry is specified as a varint and refers to an index within the Palette. The entries are indexed by x + z * Width + y * Width * Length.

### Fields

DataVersion: https://minecraft.gamepedia.com/Data_version

e.g. Java Edition 1.16.4  - DataVersion 2584

## Minecraft NBT Spec

Named Binary Tag specification

NBT (Named Binary Tag) is a tag based binary format designed to carry large amounts of binary data with smaller amounts of additional data.
An NBT file consists of a single GZIPped Named Tag of type TAG_Compound.

A Named Tag has the following format:

```
    byte tagType
    TAG_String name
    [payload] 
```

The tagType is a single byte defining the contents of the payload of the tag.

The name is a descriptive name, and can be anything (eg "cat", "banana", "Hello World!"). It has nothing to do with the tagType.
The purpose for this name is to name tags so parsing is easier and can be made to only look for certain recognized tag names.
Exception: If tagType is TAG_End, the name is skipped and assumed to be "".

The [payload] varies by tagType.

Note that ONLY Named Tags carry the name and tagType data. Explicitly identified Tags (such as TAG_String above) only contains the payload.

The tag types and respective payloads are:

```
    TYPE: 0  NAME: TAG_End
    Payload: None.
    Note:    This tag is used to mark the end of a list.
             Cannot be named! If type 0 appears where a Named Tag is expected, the name is assumed to be "".
             (In other words, this Tag is always just a single 0 byte when named, and nothing in all other cases) 
    TYPE: 1  NAME: TAG_Byte
    Payload: A single signed byte (8 bits)

    TYPE: 2  NAME: TAG_Short
    Payload: A signed short (16 bits, big endian)

    TYPE: 3  NAME: TAG_Int
    Payload: A signed short (32 bits, big endian)

    TYPE: 4  NAME: TAG_Long
    Payload: A signed long (64 bits, big endian)

    TYPE: 5  NAME: TAG_Float
    Payload: A floating point value (32 bits, big endian, IEEE 754-2008, binary32)

    TYPE: 6  NAME: TAG_Double
    Payload: A floating point value (64 bits, big endian, IEEE 754-2008, binary64) 
    TYPE: 7  NAME: TAG_Byte_Array
    Payload: TAG_Int length 
             An array of bytes of unspecified format. The length of this array is <length> bytes

    TYPE: 8  NAME: TAG_String
    Payload: TAG_Short length 
             An array of bytes defining a string in UTF-8 format. The length of this array is <length> bytes

    TYPE: 9  NAME: TAG_List
    Payload: TAG_Byte tagId
             TAG_Int length
             A sequential list of Tags (not Named Tags), of type <typeId>. The length of this array is <length> Tags
    Notes:   All tags share the same type. 
    TYPE: 10 NAME: TAG_Compound
    Payload: A sequential list of Named Tags. This array keeps going until a TAG_End is found.
             TAG_End end
    Notes:   If there's a nested TAG_Compound within this tag, that one will also have a TAG_End, so simply reading until the next TAG_End will not work.
             The names of the named tags have to be unique within each TAG_Compound
             The order of the tags is not guaranteed.
```
