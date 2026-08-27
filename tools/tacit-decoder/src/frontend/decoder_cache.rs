use rustc_data_structures::fx::FxHashMap;
use rvdasm::insn::Insn;

pub struct BasicBlockStats {
    pub target_pc: u64,
    pub num_instructions: u64,
}

pub struct DecoderCache {
    cache: FxHashMap<u64, BasicBlockStats>,
}

impl DecoderCache {
    pub fn new() -> Self {
        Self { cache: FxHashMap::default() }
    }

    pub fn get(&self, pc: u64) -> Option<&BasicBlockStats> {
        self.cache.get(&pc)
    }

    pub fn insert(&mut self, pc: u64, basic_block_stats: BasicBlockStats) {
        self.cache.insert(pc, basic_block_stats);
    }

    pub fn contains_key(&self, pc: u64) -> bool {
        self.cache.contains_key(&pc)
    }

    pub fn reset(&mut self) {
        self.cache.clear();
    }
}