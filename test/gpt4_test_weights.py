import numpy as np
from crc import Calculator, Configuration

# return the summed up values  
def sum_values(data):
    return data.bit_count()

config = Configuration(
    width=8,
    polynomial=0x31,
    init_value=0xff,
    final_xor_value=0xff,
    reverse_input=True,
    reverse_output=True,
)
calculator = Calculator(config)

weight = []

n = 16
k = 8
r = 8

for i in range(0, n+1):
    weight.append(0)

data = 0x00
telegram = 0x0000
for i in range(0, 2**k):
    data = i
    # Calculate the CRC8
    crc_result = calculator.checksum(data)
    #print(f"CRC8 Result: {crc_result:02X}")

    telegram = (data << r) | crc_result
    telegram = telegram & 0xFFFF

    print(f"Data: {data:02X}, CRC: {crc_result:02X}, Telegram: {telegram:02X}")

    weight[sum_values(telegram)] += 1

# Print the result
#print(f"CRC8 Result: {crc_result:02X}")

print("Weight distribution:")
for i in range(0, n+1):
    print(f"{i}: {weight[i]}")



