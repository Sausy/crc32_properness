import crc

config = crc.Configuration(
    width=32,
    polynomial=0x1_F4AC_FB13,  # the 1_ does not make a difference ....
    init_value=0xFFFF_FFFF,
    # init_value=0x0000_0000,
    final_xor_value=0xFFFF_FFFF,
    reverse_input=True, # our CRC is read with the least significant bit first
    reverse_output=True, #True,
)
calculator = crc.Calculator(config)


def input(value: int, bits: int) -> bytes:
    """make a byte array from an int of a given bit width.
    The first byte ist the least significant byte."""
    out = []
    for _ in range(bits // 8):
        out.append(value % 256)
        value = value // 256
    
    #print("out: ", out)
    return bytes(out)


def int_to_hex_list(num):
    # Ensure the input is within 32-bit range
    if not 0 <= num <= 0xFFFFFFFF:
        raise ValueError("Input integer should be within the range [0, 4294967295]")

    # Convert integer to hexadecimal string
    hex_string = hex(num)[2:]

    # Ensure the length of the hex string is a multiple of 2
    if len(hex_string) % 2 != 0:
        hex_string = "0" + hex_string

    # Pad the hex string with zeros to ensure its length is a multiple of 8
    hex_string = hex_string.rjust(8, "0")

    # Split the hex string into 8-bit (2-byte) chunks
    hex_values = [hex_string[i : i + 2] for i in range(0, len(hex_string), 2)]

    return hex_values


def convert_inputHex(inp_data):
    data_ = inp_data[0:4]
    res_str = ""
    separator = ", "
    res = ["{:02X}".format(n) for n in data_]
    res_str = res_str.join(res)
    res_str = int(res_str, 16)
    # print("data: ", res_str)

    crc32_calc = calculator.checksum(input(res_str, 32))

    # now get it as list
    crc32_calc = int_to_hex_list(crc32_calc)

    inp_data_list = list(inp_data)

    for i in range(4):
        # print("i: {} .... {}|{}".format(i, crc32_calc[i], crc32_calc))
        inp_data_list[i + 4] = int(crc32_calc[i], 16)

    inp_data = tuple(inp_data_list)

    return inp_data

print("input 0x01")
print(f"0x{calculator.checksum(input(0x01, 8)):08x}")

print("\ninput 0x01 0x02 0x03 0x04")
print(f"0x{calculator.checksum(input(0x0102_0304, 32)):08x}")
print("\ninput 0x04 0x03 0x02 0x01")
print(f"0x{calculator.checksum(input(0x04030201, 32)):08x}")
print("\ninput 0x80, 0x40, 0xC0, 0x02")
print(f"0x{calculator.checksum(input(0x8040C002, 32)):08x}")
print("\ninput 0x02, 0xC0, 0x40, 0x80")
print(f"0x{calculator.checksum(input(0x02C04080, 32)):08x}")

print(f"0x{calculator.checksum(input(0x00_0000_0000_0001, 56)):08x}")
print(f"0x{calculator.checksum(input(0x80_0000_0000_0000, 56)):08x}")
print(f"0x{calculator.checksum(input(0x01_2345_6789_ABCD, 56)):08x}")
print(f"0x{calculator.checksum(input(0x10_2030_4050_6070, 56)):08x}")

print(f"0x{calculator.checksum(input(0x0000_0001, 32)):08x}")
print(f"0x{calculator.checksum(input(0x8000_0000, 32)):08x}")
print(f"0x{calculator.checksum(input(0x0123_4567, 32)):08x}")
print(f"0x{calculator.checksum(input(0x1020_3040, 32)):08x}")


print(f"0x{calculator.checksum(input(0x0001, 16)):08x}")
print(f"0x{calculator.checksum(input(0x8000, 16)):08x}")
print(f"0x{calculator.checksum(input(0x0123, 16)):08x}")
print(f"0x{calculator.checksum(input(0x1020, 16)):08x}")


print(f"0x{calculator.checksum(input(0x0001_0000, 32)):08x}")
print(f"0x{calculator.checksum(input(0x0111_0000, 32)):08x}")

# data_out = (0x00, 0x01, 0x00, 0x00, 0x45, 0xC2, 0x33, 0xB5, 0, 0, 0, 0, 0, 0, 0, 0)
data_out = (0x01, 0x11, 0x00, 0x00, 0x45, 0xC2, 0x33, 0xB5, 0, 0, 0, 0, 0, 0, 0, 0)

res = convert_inputHex(data_out)


foo = input(0x0001_0000, 32)
print("full input: ", foo)
for i in range(3):
    print("input", foo[i])
foo = calculator.checksum(input(0x0001_0000, 32))
print("foo :", foo)
# print(f"0x{calculator.checksum(input(0x0080_0000, 32)):08x}")
# print(f"0x{calculator.checksum(input(0x0101_0000, 32)):08x}")
# print(f"0x{calculator.checksum(input(0x0211_0000, 32)):08x}")
