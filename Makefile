# Makefile for Boat Deep Learning Framework

.PHONY: all clean install uninstall test examples

# Configuration
# Force bash recipes: the GnuWin32 make on this machine's PATH
# runs recipes via cmd.exe, where MSYS paths and the lld search break.
SHELL := /bin/bash
CC = gcc
CFLAGS = -std=c11 -Wall -Wextra -O2 -fPIC -DBOAT_BUILDING_DLL
INCLUDES = -Iinclude
LIBS = -lm

# Platform detection: PE DLL + import lib on Windows/MinGW, .so on Linux.
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
    LIB_NAME = libboat.so
    IMPLIB =
else
    # Windows/MinGW: PE DLL + import lib. GNU ld crashes (silent "ld
    # returned 5") on objects with many PE import relocations in this
    # environment; LLVM lld links the same objects correctly, so the
    # Windows build uses -fuse-ld=lld (D:\llvm on PATH).
    LIB_NAME = boat.dll
    # export-all: the public headers declare BOAT_API but ~100 definitions
    # lack the dllexport attribute (a latent bug the static cmake build
    # hides); export-all makes every global symbol visible regardless.
    IMPLIB = -Wl,--out-implib,$(LIB_DIR)/libboat.dll.a -Wl,--export-all-symbols
    # -B points gcc/collect2 at lld directly (a PATH export does not
    # survive the MSYS->Windows path conversion for the recipe shell).
    LDFLAGS = -fuse-ld=lld -B/d/llvm/bin
endif

# Version information (auto-generated header)
VERSION_MAJOR = 0
VERSION_MINOR = 1
VERSION_PATCH = 0
VERSION_STRING = $(VERSION_MAJOR).$(VERSION_MINOR).$(VERSION_PATCH)
VERSION_H = include/boat/version.h

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
FORMAT_SRCS = $(filter-out $(SRC_DIR)/format/onnxruntime.c, $(wildcard $(SRC_DIR)/format/*.c))

ALL_SRCS = $(CORE_SRCS) $(OPS_SRCS) $(GRAPH_SRCS) $(LAYERS_SRCS) \
           $(OPTIMIZERS_SRCS) $(SCHEDULERS_SRCS) $(LOSS_SRCS) $(MODEL_SRCS) \
           $(DATA_SRCS) $(FORMAT_SRCS) \
           $(SRC_DIR)/autodiff.c

# Object files
OBJS = $(patsubst $(SRC_DIR)/%.c,$(OBJ_DIR)/%.o,$(ALL_SRCS))

# Library
LIB = $(LIB_DIR)/$(LIB_NAME)

# Main targets
all: $(VERSION_H) $(LIB)

$(VERSION_H): include/boat/version.h.in
	@HASH=$$(git rev-parse --short HEAD 2>/dev/null || echo unknown); DESC=$$(git describe --tags --always --dirty 2>/dev/null || echo unknown); sed -e s/@BOAT_VERSION_MAJOR@/$(VERSION_MAJOR)/ -e s/@BOAT_VERSION_MINOR@/$(VERSION_MINOR)/ -e s/@BOAT_VERSION_PATCH@/$(VERSION_PATCH)/ -e s/@BOAT_VERSION_STRING@/$(VERSION_STRING)/ -e s/@BOAT_GIT_HASH@/$$HASH/ -e s/@BOAT_GIT_DESCRIBE@/$$DESC/ $< > $@

$(LIB): $(OBJS)
	@mkdir -p $(LIB_DIR)
	$(CC) -shared $(CFLAGS) $(LDFLAGS) $(OBJS) -o $@ $(LIBS) $(IMPLIB)

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
# CPU-only test suite: everything except backends needing external SDKs
# or data (CUDA, PyTorch, Safetensors, HuggingFace, GGUF, ONNX(-runtime),
# TensorFlow). The same set is what the cmake CPU build runs under ctest.
TEST_EXCLUDE = $(wildcard tests/*cuda*.c tests/*pytorch*.c tests/*safetensors*.c 	tests/*huggingface*.c tests/*gguf*.c tests/*onnx*.c tests/*onnxruntime*.c 	tests/*tensorflow*.c)
TEST_SRCS = $(filter-out $(TEST_EXCLUDE),$(wildcard tests/*.c) $(wildcard tests/unit/*.c))
TEST_BINS = $(patsubst tests/%,build/test/%,$(patsubst %.c,%.exe,$(TEST_SRCS)))
TEST_LIBS = -L$(LIB_DIR) -lboat $(LIBS)

build/test/%.exe: tests/%.c $(LIB)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(LDFLAGS) $(INCLUDES) $< -o $@ $(TEST_LIBS)

test: all $(TEST_BINS)
	@echo "Running $$(words $(TEST_BINS)) tests..."
	@fail=0; pass=0; \
	for t in $(TEST_BINS); do \
		if PATH="$(LIB_DIR):$$PATH" ./$$t > build/test/$$(basename $$t).log 2>&1; then \
			echo "  PASS $$(basename $$t)"; pass=$$((pass+1)); \
		else \
			echo "  FAIL $$(basename $$t)"; tail -3 build/test/$$(basename $$t).log; fail=$$((fail+1)); \
		fi; \
	done; \
	echo "Tests: $$pass passed, $$fail failed"; \
	test $$fail -eq 0

# Static archive (used by MATLAB_in_C's deep learning builtins)
LIB_STATIC = $(LIB_DIR)/libboat.a
$(LIB_STATIC): $(OBJS)
	@mkdir -p $(LIB_DIR)
	$(AR) rcs $@ $(OBJS)

static: $(VERSION_H) $(LIB_STATIC)

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
