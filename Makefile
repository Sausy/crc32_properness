# Compiler
CC := gcc
# Compiler flags
CFLAGS := -Wall -Wextra -g

# Source files
SRCS := $(wildcard *.c)
# Object files
OBJS := $(SRCS:.c=.o)

# Target executable
TARGET := myprogram

# Default target
all: $(TARGET)

# Rule to build the executable
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $^ -o $@

# Rule to build object files
%.o: %.c %.h
	$(CC) $(CFLAGS) -c $< -o $@

# Clean rule
clean:
	rm -f $(OBJS) $(TARGET)