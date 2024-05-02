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

