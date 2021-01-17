

## NBT Files

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

### Format

* https://minecraft.gamepedia.com/NBT_format

### Example

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

    byte tagType
    TAG_String name
    [payload] 
The tagType is a single byte defining the contents of the payload of the tag.

The name is a descriptive name, and can be anything (eg "cat", "banana", "Hello World!"). It has nothing to do with the tagType.
The purpose for this name is to name tags so parsing is easier and can be made to only look for certain recognized tag names.
Exception: If tagType is TAG_End, the name is skipped and assumed to be "".

The [payload] varies by tagType.

Note that ONLY Named Tags carry the name and tagType data. Explicitly identified Tags (such as TAG_String above) only contains the payload.

The tag types and respective payloads are:

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

Decoding example:
(Use http://www.minecraft.net/docs/test.nbt to test your implementation)

First we start by reading a Named Tag.
After unzipping the stream, the first byte is a 10. That means the tag is a TAG_Compound (as expected by the specification).

The next two bytes are 0 and 11, meaning the name string consists of 11 UTF-8 characters. In this case, they happen to be "hello world".
That means our root tag is named "hello world". We can now move on to the payload.

From the specification, we see that TAG_Compound consists of a series of Named Tags, so we read another byte to find the tagType.
It happens to be an 8. The name is 4 letters long, and happens to be "name". Type 8 is TAG_String, meaning we read another two bytes to get the length,
then read that many bytes to get the contents. In this case, it's "Bananrama".

So now we know the TAG_Compound contains a TAG_String named "name" with the content "Bananrama"

We move on to reading the next Named Tag, and get a 0. This is TAG_End, which always has an implied name of "". That means that the list of entries
in the TAG_Compound is over, and indeed all of the NBT file.

So we ended up with this:

	TAG_Compound("hello world"): 1 entries
	{
	   TAG_String("name"): Bananrama
	}

For a slightly longer test, download http://www.minecraft.net/docs/bigtest.nbt
You should end up with this:

	TAG_Compound("Level"): 11 entries
	{
	   TAG_Short("shortTest"): 32767
	   TAG_Long("longTest"): 9223372036854775807
	   TAG_Float("floatTest"): 0.49823147
	   TAG_String("stringTest"): HELLO WORLD THIS IS A TEST STRING ÅÄÖ!
	   TAG_Int("intTest"): 2147483647
	   TAG_Compound("nested compound test"): 2 entries
	   {
	      TAG_Compound("ham"): 2 entries
	      {
	         TAG_String("name"): Hampus
	         TAG_Float("value"): 0.75
	      }
	      TAG_Compound("egg"): 2 entries
	      {
	         TAG_String("name"): Eggbert
	         TAG_Float("value"): 0.5
	      }
	   }
	   TAG_List("listTest (long)"): 5 entries of type TAG_Long
	   {
	      TAG_Long: 11
	      TAG_Long: 12
	      TAG_Long: 13
	      TAG_Long: 14
	      TAG_Long: 15
	   }
	   TAG_Byte("byteTest"): 127
	   TAG_List("listTest (compound)"): 2 entries of type TAG_Compound
	   {
	      TAG_Compound: 2 entries
	      {
	         TAG_String("name"): Compound tag #0
	         TAG_Long("created-on"): 1264099775885
	      }
	      TAG_Compound: 2 entries
	      {
	         TAG_String("name"): Compound tag #1
	         TAG_Long("created-on"): 1264099775885
	      }
	   }
	   TAG_Byte_Array("byteArrayTest (the first 1000 values of (n*n*255+n*7)%100, starting with n=0 (0, 62, 34, 16, 8, ...))"): [1000 bytes]
	   TAG_Double("doubleTest"): 0.4931287132182315
	}
