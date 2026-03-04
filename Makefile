# Makefile for Boat Deep Learning Framework

.PHONY: all clean install uninstall test examples

# Configuration
CC = gcc
CFLAGS = -std=c11 -Wall -Wextra -O2 -fPIC -DBOAT_BUILDING_DLL
INCLUDES = -Iinclude
LIBS = -lm

# Directories
SRC_DIR = src
BUILD_DIR = build
LIB_DIR = $(BUILD_DIR)/lib
OBJ_DIR = $(BUILD_DIR)/obj

# Source files
CORE_SRCS = $(wildcard $(SRC_DIR)/core/*.c)
OPS_SRCS = $(wildcard $(SRC_DIR)/ops/*.c) $(wildcard $(SRC_DIR)/ops/autodiff/*.c)
GRAPH_SRCS = $(wildcard $(SRC_DIR)/graph/*.c)
LAYERS_SRCS = $(wildcard $(SRC_DIR)/layers/*.c)
OPTIMIZERS_SRCS = $(wildcard $(SRC_DIR)/optimizers/*.c)
LOSS_SRCS = $(wildcard $(SRC_DIR)/loss/*.c)
SCHEDULERS_SRCS = $(wildcard $(SRC_DIR)/schedulers/*.c)
MODEL_SRCS = $(wildcard $(SRC_DIR)/model/*.c)
DATA_SRCS = $(wildcard $(SRC_DIR)/data/*.c)
FORMAT_SRCS = $(wildcard $(SRC_DIR)/format/*.c)

ALL_SRCS = $(CORE_SRCS) $(OPS_SRCS) $(GRAPH_SRCS) $(LAYERS_SRCS) \
           $(OPTIMIZERS_SRCS) $(SCHEDULERS_SRCS) $(LOSS_SRCS) $(MODEL_SRCS) \
           $(DATA_SRCS) $(FORMAT_SRCS) \
           $(SRC_DIR)/autodiff.c

# Object files
OBJS = $(patsubst $(SRC_DIR)/%.c,$(OBJ_DIR)/%.o,$(ALL_SRCS))

# Library
LIB_NAME = libboat.so
LIB = $(LIB_DIR)/$(LIB_NAME)

# Main targets
all: $(LIB)

$(LIB): $(OBJS)
	@mkdir -p $(LIB_DIR)
	$(CC) -shared $(CFLAGS) $(OBJS) -o $@ $(LIBS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

# Clean
clean:
	rm -rf $(BUILD_DIR)

# Install (system-wide)
PREFIX = /usr/local
install: $(LIB)
	@mkdir -p $(PREFIX)/lib
	@mkdir -p $(PREFIX)/include/boat
	cp $(LIB) $(PREFIX)/lib/
	cp -r include/boat/* $(PREFIX)/include/boat/

# Uninstall
uninstall:
	rm -f $(PREFIX)/lib/$(LIB_NAME)
	rm -rf $(PREFIX)/include/boat

# Test
test:
	@echo "Running tests..."
	# TODO: Add test runner

# Examples
examples:
	@echo "Building examples..."
	# TODO: Build examples

# Development
dev: CFLAGS += -g -DDEBUG
dev: all

# Release
release: CFLAGS += -O3 -DNDEBUG
release: all