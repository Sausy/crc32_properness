import numpy as np
from crc import Calculator, Configuration

def calculate_systematic_generator_matrix(polynom, k, r):
    n = k + r
    
    # Create the identity matrix for the data part
    identity_matrix = np.eye(k, dtype=int)
    
    # Create the generator matrix part for the parity bits
    parity_matrix = np.zeros((k, r), dtype=int)
    for i in range(k):
        dataword = np.zeros(k, dtype=int)
        dataword[i] = 1
        crc_value = calculate_crc(polynom, dataword, r)
        parity_matrix[i, :] = np.array([int(x) for x in format(crc_value, f'0{r}b')])
    
    # Combine the identity matrix and parity matrix to form the generator matrix
    generator_matrix = np.hstack((identity_matrix, parity_matrix))
    
    return generator_matrix

def calculate_parity_check_matrix(polynom, k, r):
    n = k + r
    
    # Parity check matrix consists of two parts: an identity matrix and a parity part
    identity_matrix = np.eye(r, dtype=int)
    parity_part = np.zeros((r, k), dtype=int)
    
    for i in range(k):
        dataword = np.zeros(k, dtype=int)
        dataword[i] = 1
        crc_value = calculate_crc(polynom, dataword, r)
        parity_part[:, i] = np.array([int(x) for x in format(crc_value, f'0{r}b')])
    
    parity_check_matrix = np.hstack((parity_part.T, identity_matrix))
    
    return parity_check_matrix

def calculate_crc(polynom, dataword, r):
    config = Configuration(
        width=r,
        polynomial=polynom,
        init_value=0,
        final_xor_value=0,
        reverse_input=True,
        reverse_output=True
    )
    
    calculator = Calculator(config)
    dataword_bits = ''.join(map(str, dataword))
    crc_value = calculator.checksum(bytes(int(dataword_bits, 2).to_bytes((len(dataword_bits) + 7) // 8, byteorder='big')))
    
    return crc_value

# Example usage
polynom = 0xB  # Example polynomial, you can change it to any valid polynomial
r = 3
k = 4  # Example dataword length, change as needed

generator_matrix = calculate_systematic_generator_matrix(polynom, k, r)
parity_check_matrix = calculate_parity_check_matrix(polynom, k, r)

print("Systematic Generator Matrix:")
print(generator_matrix)

print("\nParity Check Matrix:")
print(parity_check_matrix)