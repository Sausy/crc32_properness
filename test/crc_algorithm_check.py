def generate_crc32_table(polynomial):
    crc_table = []
    for dividend in range(256):
        cur_byte = dividend << 24
        for bit in range(8):
            if (cur_byte & 0x80000000) != 0:
                cur_byte = (cur_byte << 1) ^ polynomial
            else:
                cur_byte <<= 1
        crc_table.append(cur_byte)
    return crc_table

def reflect_bits(data, bits=8):
    reflected = 0
    for i in range(bits):
        if data & (1 << i):
            reflected |= 1 << (bits - 1 - i)
    return reflected

def crc32(crc_table, buf):
    crc = 0xFFFFFFFF

    for byte in buf:
        idx = ((crc >> 24) ^ reflect_bits(byte)) & 0xFF
        crc = crc_table[idx] ^ (crc << 8)


    crc = reflect_bits(crc, 32)
    
    return crc ^ 0xFFFFFFFF

def print_crc32_table(crc_table):
    print("Table Size =", len(crc_table))
    print("crcTable =")
    for i, crc in enumerate(crc_table):
        print(f"[0x{crc:08X}]", end="")
        if (i + 1) % 8 == 0:
            print()
    print()

# Example usage
POLYNOMIAL = 0xF4ACFB13  # Example polynomial
crc_table = generate_crc32_table(POLYNOMIAL)
print_crc32_table(crc_table)

data = b"hello world"
crc_result = crc32(crc_table, data)
print("CRC Result:", f"{crc_result:08X}")
