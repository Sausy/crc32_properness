# CRC Calculation Library

## Overview
This library provides methods to calculate CRC values using various configurations and polynomial settings.

## Requirements
- Compiler: g++ with C++11 support
- OS: Windows

## Compilation
Use the provided Makefile:

```bash
make
```


## Definitions

$0x25 = 0010 0101$ is taken as   

$$0\cdot x^7 + 0\cdot x^6 + 1\cdot x^5 + 0\cdot x^4 + 0\cdot x^3 + 1\cdot x^2 + 0\cdot x^1 + 1\cdot x^0.$$

$$ = 1\cdot x^5    + 1\cdot x^2 + 1\cdot x^0.$$

$$ = x^5  +   x^2 + x^0.$$

### Generator Matrix and parity check Matrix
https://www.ece.uvic.ca/~agullive/cycliccodes405-511-2016.pdf 

Generator Matrix is $G$ 
and Parity check matrix is $H$ 

please note that (for polynom 0xB for CRC4)

$$h(x) = (1+x^2+x3)(1+x) = 1+x+x^2+x^4$$


Certainly! Let's delve deeper into Step 5, which involves the practical computation of weight distribution from a parity check matrix $H$. This involves computational steps to determine how many combinations of rows can produce vectors of each possible weight.

### Detailed Practical Computation

**Parity Check Matrix $H$**

For a linear block code, the parity check matrix $H$ has dimensions $(n-k) \times n$, where $n$ is the total number of bits in each codeword, and $k$ is the number of message bits. The matrix $H$ ensures that for any valid codeword $c$, $Hc^T = 0$.

#### 1. **Binary Vector Operations**

Each row of $H$ represents a constraint that any valid codeword must satisfy. A binary vector operation in this context involves:
- **XOR Operation**: Logical XOR operation between two binary vectors (rows) is defined as:
  $$[v_1 \oplus v_2]_i = v_{1i} \oplus v_{2i} \quad \text{for all } i \text{ from } 1 \text{ to } n$$
  where $v_{1i}$ and $v_{2i}$ are the $i$-th bits of vectors $v_1$ and $v_2$ respectively, and $\oplus$ denotes the XOR operation.

#### 2. **Generating All Row Combinations**

We consider all possible subsets of rows from $H$. The number of such subsets (excluding the empty set) is $2^{n-k} - 1$. For each subset:
- **Subset Selection**: A subset of rows $R \subset \{1, 2, ..., n-k\}$ is selected. The selection can be represented as a binary number where each bit position indicates whether a row is included.
- **Vector Combination**: The corresponding rows are combined using XOR:
  $$v = \bigoplus_{i \in R} H_i$$
  where $H_i$ is the $i$-th row of $H$, and $\bigoplus$ denotes the XOR operation applied across multiple vectors.

#### 3. **Calculating the Weight of Each Vector**

For each resultant vector $v$ from the combination process:
- **Weight Calculation**: The weight $w(v)$ of a vector $v$ is the number of 1's in $v$:
  $$w(v) = \sum_{i=1}^n v_i$$
  where $v_i$ is the $i$-th bit of vector $v$.

#### 4. **Building the Weight Distribution**

For every vector $v$ calculated:
- **Distribution Tallying**: Increase the count of vectors that have weight $w(v)$ in a distribution array $D$ where $D[w]$ represents the number of vectors of weight $w$:
  $$D[w(v)] = D[w(v)] + 1$$
- The array $D$ will be of size $n+1$ to account for all possible weights from 0 to $n$.

### Summary of Formulas and Steps

1. **Vector XOR**: $[v_1 \oplus v_2]_i = v_{1i} \oplus v_{2i}$.
2. **Vector Combination**: $v = \bigoplus_{i \in R} H_i$.
3. **Weight Calculation**: $w(v) = \sum_{i=1}^n v_i$.
4. **Distribution Update**: $D[w(v)] = D[w(v)] + 1$.

These steps will calculate the weight distribution for a parity check matrix, reflecting the number of vectors at each weight level that can arise from different combinations of rows, providing insights into the error detection and correction capabilities of the code.
