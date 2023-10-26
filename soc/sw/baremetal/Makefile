#################################
# RISCV Toolchain
#################################

PREFIX = riscv64-unknown-elf-

GCC = $(PREFIX)gcc
CXX = $(PREFIX)g++
CP = $(PREFIX)objcopy
OBJDUMP = $(PREFIX)objdump
DG = $(PREFIX)gdb
SIZE = $(PREFIX)size


#################################
# Flags
#################################

# SoC Settings
ARCH = rv64imafdc
ABI = lp64d
ARCHFLAGS = -march=$(ARCH) -mabi=$(ABI)

CFLAGS  = -std=gnu99 -O2 -fno-common -fno-builtin-printf -Wall
CFLAGS += $(ARCHFLAGS)
LDFLAGS = -static

include libgloss.mk

PROGRAMS = airsim-packettest airsim-packettest-double-rdcycle airsim-packettest-single-rdcycle inverted-pendulum-pid inverted-pendulum-pid-nonblock


.DEFAULT_GOAL := default


#################################
# Build
#################################

spiflash.img: spiflash.py
	python3 $<

%.o: %.S
	$(GCC) $(CFLAGS) -D__ASSEMBLY__=1 -c $< -o $@

%.o: %.c mmio.h spiflash.h
	$(GCC) $(CFLAGS) -c $< -o $@

%.riscv: %.o $(libgloss)
	$(GCC) $(LDFLAGS) $< -o $@

%.dump: %.riscv
	$(OBJDUMP) -D $< > $@


#################################
# Install
#################################

INSTALL_DIR = ../../sim/firesim/deploy/workloads/
ROOTFS_PATH = ../../../target-design/chipyard/software/firemarshal/boards/default/installers/firesim/dummy.rootfs

.PHONY: install
install: default
	@mkdir -p $(INSTALL_DIR)
	$(foreach prog,$(PROGRAMS), \
		( \
			mkdir -p $(INSTALL_DIR)/$(prog)/ && \
			cp $(prog).riscv $(INSTALL_DIR)/$(prog)/ && \
			echo '{' > $(INSTALL_DIR)/$(prog).json && \
			echo '  "benchmark_name": "$(prog)",' >> $(INSTALL_DIR)/$(prog).json && \
			echo '  "common_simulation_outputs": ["uartlog"],' >> $(INSTALL_DIR)/$(prog).json && \
			echo '  "common_bootbinary": "$(prog).riscv",' >> $(INSTALL_DIR)/$(prog).json && \
			echo '  "common_rootfs": "$(ROOTFS_PATH)"' >> $(INSTALL_DIR)/$(prog).json && \
			echo '}' >> $(INSTALL_DIR)/$(prog).json \
		);)



#################################
# Recipes
#################################

.PHONY: clean
clean:
	rm -f *.riscv *.o *.dump
	$(if $(libgloss),rm -rf $(libgloss_builddir)/)

.PHONY: default
default: $(addsuffix .riscv, $(PROGRAMS)) spiflash.img

.PHONY: dumps
dumps: $(addsuffix .dump, $(PROGRAMS))
