# Compiler settings - Can change these to your preferred compiler and options
CXX = g++
CXXFLAGS = -Wall -std=c++11 -Iinclude
BUILD_DIR = ./build

# Detect the operating system
ifeq ($(OS),Windows_NT)
    RM = del /Q
    FixPath = $(subst /,\,$1)
    MKDIR = if not exist "$(subst /,\,$1)" mkdir $(subst /,\,$1)
    EXEC_SUFFIX = .exe
else
    RM = rm -rf
    FixPath = $1
    MKDIR = mkdir -p $1
    EXEC_SUFFIX =
endif

# Executables
MAIN = $(BUILD_DIR)/main$(EXEC_SUFFIX)
TESTS = $(patsubst ./include/test_%.cpp,$(BUILD_DIR)/test_%$(EXEC_SUFFIX),$(wildcard ./include/test_*.cpp))

# Default target
all: $(MAIN) $(TESTS)

# Main executable
$(MAIN): ./main.cpp $(BUILD_DIR)/crc.o
	$(CXX) $(CXXFLAGS) $^ -o $@

# Test executables
$(BUILD_DIR)/test_%$(EXEC_SUFFIX): ./include/test_%.cpp $(BUILD_DIR)/crc.o
	$(CXX) $(CXXFLAGS) $^ -o $@

# Compile .cpp to .o
$(BUILD_DIR)/%.o: ./include/%.cpp
	$(MKDIR) $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean up
clean:
	$(RM) $(call FixPath,$(BUILD_DIR)/*)

.PHONY: all clean
