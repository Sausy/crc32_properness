import numpy as np
from crc import Calculator, Configuration

# return the summed up values  
def sum_values(data):
    return bin(data).count('1')

config = Configuration(
    width=32,
    polynomial=0xf1922815,
    init_value=0xffffffff,
    final_xor_value=0xffffffff,
    reverse_input=True,
    reverse_output=True,
)
calculator = Calculator(config)

weight = []

n = 64
k = 32
r = 32

for i in range(0, n+1):
    weight.append(0)

data = 0x00000000
telegram = 0x0000000000000000
for i in range(0, 2**k):
    data = i.to_bytes(4, 'big')  # Convert data to 4 bytes
    # Calculate the CRC32
    crc_result = calculator.checksum(data)

    data_int = int.from_bytes(data, 'big')
    telegram = (data_int << r) | crc_result
    telegram = telegram & 0xffffffffffffffff

    weight[sum_values(telegram)] += 1

    if i%1_000_000 == 0:
        print("chunk: ", i, " \t max size: ", 2**k)

print("Weight distribution:")
for i in range(0, n+1):
    print(f"{i}: {weight[i]}")