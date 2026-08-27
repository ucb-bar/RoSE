use crate::common::prv::*;
use crate::common::static_cfg::DecoderStaticCfg;
use anyhow::Result;
use log::{debug, trace};
use object::elf::SHF_EXECINSTR;
use object::{Object, ObjectSection};
use rvdasm::disassembler::*;
use rvdasm::insn::Insn;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use rustc_data_structures::fx::FxHashMap;

pub struct InstructionIndex {
    u_insn_maps: HashMap<u64, FxHashMap<u64, Insn>>,
    k_insn_map: FxHashMap<u64, Insn>,
    m_insn_map: FxHashMap<u64, Insn>,
    empty_map: FxHashMap<u64, Insn>,
}

impl InstructionIndex {
    pub fn get(&self, space: Prv, ctx: u64) -> &FxHashMap<u64, Insn> {
        match space {
            Prv::PrvUser => {
                if self.u_insn_maps.contains_key(&ctx) {
                    &self.u_insn_maps[&ctx]
                } else {
                    &self.empty_map
                }
            }
            Prv::PrvSupervisor => &self.k_insn_map,
            Prv::PrvMachine => &self.m_insn_map,
            _ => panic!("Unsupported privilege level: {:?}", space),
        }
    }
}

pub fn build_instruction_index(cfg: DecoderStaticCfg) -> Result<InstructionIndex> {
    // Machine-space instruction map (SBI)
    let mut m_elf_file = File::open(cfg.sbi_binary)?;
    let mut m_elf_buffer = Vec::new();
    m_elf_file.read_to_end(&mut m_elf_buffer)?;
    let m_elf = object::File::parse(&*m_elf_buffer)?;
    let m_elf_arch = m_elf.architecture();
    // Determine architecture and create a disassembler from the SBI binary
    let xlen = if m_elf_arch == object::Architecture::Riscv64 {
        Xlen::XLEN64
    } else if m_elf_arch == object::Architecture::Riscv32 {
        Xlen::XLEN32
    } else {
        panic!("Unsupported architecture: {:?}", m_elf_arch);
    };
    let dasm = Disassembler::new(xlen);
    // Machine-space ELF may be a unikernel (e.g. Zephyr) whose executable code is spread across
    // several sections (`text`, `reset`, `exceptions`, `rom_start`, ISR tables, ...) and whose
    // section is named `text` (no dot) rather than `.text`. Disassemble EVERY executable section
    // keyed by its own vaddr -- exactly like the user-space path below -- so section-name and
    // entry!=text base offsets can't drop or misalign the boot/trap/ISR code the unwinder needs.
    let mut m_insn_map = FxHashMap::default();
    for section in m_elf.sections() {
        if let object::SectionFlags::Elf { sh_flags } = section.flags() {
            if sh_flags & (SHF_EXECINSTR as u64) != 0 {
                let addr = section.address();
                let data = section.data()?;
                let sec_map = dasm.disassemble_all(&data, addr);
                debug!(
                    "[insn_index] machine section `{}` @ {:#x}: {} insns",
                    section.name().unwrap_or("<unnamed>"),
                    addr,
                    sec_map.len()
                );
                m_insn_map.extend(sec_map);
            }
        }
    }
    if m_insn_map.is_empty() {
        return Err(anyhow::anyhow!(
            "No executable instructions found in SBI ELF"
        ));
    }
    debug!(
        "[insn_index] found {} machine-space instructions",
        m_insn_map.len()
    );
    // Determine architecture and create a disassembler from the application binary
    let mut u_insn_maps = HashMap::new();
    for (binary, asid) in cfg.application_binary_asid_tuples.clone() {
        let mut u_elf_file = File::open(binary)?;
        let mut u_elf_buffer = Vec::new();
        u_elf_file.read_to_end(&mut u_elf_buffer)?;
        let u_elf = object::File::parse(&*u_elf_buffer)?;
        let u_elf_arch = u_elf.architecture();
        assert!(
            u_elf_arch == m_elf_arch,
            "User and machine ELF architectures must match"
        );
        // User-space instruction map
        let mut u_insn_map = FxHashMap::default();
        for section in u_elf.sections() {
            if let object::SectionFlags::Elf { sh_flags } = section.flags() {
                if sh_flags & (SHF_EXECINSTR as u64) != 0 {
                    let addr = section.address();
                    let data = section.data()?;
                    let sec_map = dasm.disassemble_all(&data, addr);
                    debug!(
                        "user application section `{}` @ {:#x}: {} insns",
                        section.name().unwrap_or("<unnamed>"),
                        addr,
                        sec_map.len()
                    );
                    u_insn_map.extend(sec_map);
                }
            }
        }
        if u_insn_map.is_empty() {
            return Err(anyhow::anyhow!(
                "No executable instructions found in app ELF"
            ));
        }
        u_insn_maps.insert(asid.parse::<u64>()?, u_insn_map);
    }

    let mut k_insn_map = FxHashMap::default();
    // Kernel-space instruction map
    if cfg.kernel_binary != "" {
        let mut k_elf_file = File::open(cfg.kernel_binary)?;
        let mut k_elf_buffer = Vec::new();
        k_elf_file.read_to_end(&mut k_elf_buffer)?;
        let k_elf = object::File::parse(&*k_elf_buffer)?;
        let k_elf_arch = k_elf.architecture();
        assert!(
            k_elf_arch == m_elf_arch,
            "Kernel and user ELF architectures must match"
        );

        for section in k_elf.sections() {
            if let object::SectionFlags::Elf { sh_flags } = section.flags() {
                if sh_flags & (SHF_EXECINSTR as u64) != 0 {
                    let addr = section.address();
                    let data = section.data()?;
                    let sec_map = dasm.disassemble_all(&data, addr);
                    debug!(
                        "kernel binary section `{}` @ {:#x}: {} insns",
                        section.name().unwrap_or("<unnamed>"),
                        addr,
                        sec_map.len()
                    );
                    k_insn_map.extend(sec_map);
                }
            }
        }
        if k_insn_map.is_empty() {
            return Err(anyhow::anyhow!(
                "No executable instructions found in kernel ELF"
            ));
        }

        // Merge driver instruction maps into kernel map
        for (binary, entry) in cfg.driver_binary_entry_tuples {
            let mut driver_elf_file = File::open(binary.clone())?;
            let mut driver_elf_buffer = Vec::new();
            driver_elf_file.read_to_end(&mut driver_elf_buffer)?;
            let driver_elf = object::File::parse(&*driver_elf_buffer)?;
            let driver_elf_arch = driver_elf.architecture();
            assert!(
                driver_elf_arch == m_elf_arch,
                "Driver and user ELF architectures must match"
            );
            let driver_text_section = driver_elf
                .section_by_name(".text")
                .ok_or_else(|| anyhow::anyhow!("No .text section found"))?;
            let driver_text_data = driver_text_section.data()?;
            let driver_entry_point = u64::from_str_radix(entry.trim_start_matches("0x"), 16)?;
            let driver_insn_map = dasm.disassemble_all(&driver_text_data, driver_entry_point);
            debug!(
                "driver binary `{}` @ {:#x}: {} insns",
                binary,
                driver_entry_point,
                driver_insn_map.len()
            );
            k_insn_map.extend(driver_insn_map);
        }

        // Apply kernel jump label patch log
        if cfg.kernel_jump_label_patch_log != "" {
            let jump_label_patch_log = File::open(cfg.kernel_jump_label_patch_log)?;
            let jump_label_patch_log_reader = BufReader::new(jump_label_patch_log);
            for line in jump_label_patch_log_reader.lines() {
                let line = line?;
                let parts = line.split(',').collect::<Vec<&str>>();
                let addr = u64::from_str_radix(parts[0], 16)?;
                let raw_insn = u32::from_str_radix(parts[1], 16)?;
                // trace!("patching kernel-space instruction at {:#x} with {:#x}", addr, raw_insn);
                let new_insn = dasm.disassmeble_one(raw_insn);
                if let Some(new_insn) = new_insn {
                    k_insn_map.insert(addr, new_insn);
                } else {
                    trace!(
                        "error disassembling instruction at {:#x}: {:#x}",
                        addr,
                        raw_insn
                    );
                    continue;
                }
            }
            debug!("[insn_index] patched kernel-space instructions");
        }
    }

    Ok(InstructionIndex {
        u_insn_maps,
        k_insn_map,
        m_insn_map,
        empty_map: FxHashMap::default(),
    })
}
