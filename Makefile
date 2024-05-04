# Compiler settings - Can change these to your preferred compiler and options
CXX = g++
CXXFLAGS = -g -Wall -std=c++17 -Iinclude -Og
# maybe add -o3 for optimization
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
INC_DIR = ./include

MAIN = $(BUILD_DIR)/main$(EXEC_SUFFIX)
TESTS = $(patsubst $(INC_DIR)/test_%.cpp,$(BUILD_DIR)/test_%$(EXEC_SUFFIX),$(wildcard $(INC_DIR)/test_*.cpp))

LIST_O_TESTS = $(patsubst $(INC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(wildcard $(INC_DIR)/test_*.cpp))
LIST_O_ALL = $(patsubst $(INC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(wildcard $(INC_DIR)/*.cpp))
# we need to remove/filter out all *.o files that have "test_" in their name
LIST_O := $(filter-out $(LIST_O_TESTS),$(LIST_O_ALL))


# Compile .cpp to .o
$(BUILD_DIR)/%.o: $(INC_DIR)/%.cpp
	$(MKDIR) $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Test executables
$(BUILD_DIR)/test_%$(EXEC_SUFFIX) : $(INC_DIR)/test_%.cpp $(BUILD_DIR)/crc.o
	$(CXX) $(CXXFLAGS) $^ -o $@

# Main executable
$(MAIN): ./main.cpp $(LIST_O)
	$(CXX) $(CXXFLAGS) $^ -o $@

# Default target
all: $(MAIN) $(TESTS)


# Clean up
clean:
	$(RM) $(call FixPath,$(BUILD_DIR)/*)

.PHONY: all clean
