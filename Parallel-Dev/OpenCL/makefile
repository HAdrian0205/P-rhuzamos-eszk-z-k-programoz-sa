CC = gcc
CFLAGS = -g -Wall -I./include -I$(OPENCL_INC)
LDFLAGS = -L$(OPENCL_LIB64) -lOpenCL

# Binaries
TARGET = main

# Source files
SRCS = src/main.c src/opencl_utils.c src/input_utils.c src/time_utils.c

# Object files
OBJS = $(SRCS:.c=.o)

# Default target
all: $(TARGET)

# Link the target binary
$(TARGET): $(OBJS)
	$(CC) -o $@ $^ $(LDFLAGS)

# Compile source files to object files
src/%.o: src/%.c
	$(CC) $(CFLAGS) -c $< -o $@

# Clean up
clean:
	rm -f $(TARGET) $(OBJS)

# Phony targets
.PHONY: all clean
