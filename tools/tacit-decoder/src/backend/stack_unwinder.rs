use anyhow::Result;
use log::debug;
use std::sync::Arc;

use crate::backend::event::{Entry, EventKind, TrapReason};
use crate::common::prv::Prv;
use crate::common::symbol_index::{SymbolIndex, SymbolInfo};

#[derive(Debug, Clone)]
pub struct Frame {
    pub symbol: SymbolInfo,
    pub addr: u64,
}

pub struct StackUnwinder {
    // addr -> symbol info <name, index, line, file>
    pub func_symbol_map: Arc<SymbolIndex>,
    // stack model
    pub frame_stack: Vec<Frame>,
    // current privilege level
    pub curr_prv: Prv,
    // current context
    pub curr_ctx: u64,
}

pub struct StackUpdateResult {
    pub frames_opened: Option<Frame>,
    pub frames_closed: Vec<Frame>,
}

impl StackUnwinder {
    pub fn new(
        func_symbol_map: Arc<SymbolIndex>,
    ) -> Result<Self> {
        Ok(Self {
            func_symbol_map: func_symbol_map,
            frame_stack: Vec::new(),
            curr_prv: Prv::PrvMachine, // placeholder, will be set by the first sync start event
            curr_ctx: 0,               // placeholder, will be set by the first context change event
        })
    }

    fn push_frame(&mut self, prv: Prv, ctx: u64, addr: u64) -> Option<Frame> {
        let map = self.func_symbol_map.get(prv, ctx);
        if let Some((start, symbol)) = map.get_key_value(&addr) {
            let frame = Frame {
                symbol: symbol.clone(),
                addr: *start,
            };
            self.frame_stack.push(frame.clone());
            Some(frame)
        } else {
            None
        }
    }

    fn pop_frame(&mut self) -> Option<Frame> {
        self.frame_stack.pop().map(|frame| frame)
    }

    pub fn step(&mut self, entry: &Entry) -> Option<StackUpdateResult> {
        let Entry::Event { kind, .. } = entry else {
            return None;
        };
        match kind {
            EventKind::SyncStart {
                runtime_cfg: _,
                start_pc: _,
                start_prv,
                start_ctx,
            } => self.step_sync_start(start_prv, start_ctx),
            EventKind::InferrableJump { arc } => self.step_ij(arc.1),
            EventKind::UninferableJump { arc } => self.step_uj(arc.1),
            EventKind::Trap {
                reason,
                prv_arc,
                arc,
                ctx,
            } => self.step_trap(reason, prv_arc, arc.1, *ctx),
            _ => return None,
        }
    }

    pub fn step_sync_start(
        &mut self,
        start_prv: &Prv,
        start_ctx: &u64,
    ) -> Option<StackUpdateResult> {
        self.curr_prv = start_prv.clone();
        self.curr_ctx = start_ctx.clone();
        None
    }

    pub fn step_ij(&mut self, to_addr: u64) -> Option<StackUpdateResult> {
        let frame = self.push_frame(self.curr_prv, self.curr_ctx, to_addr);
        debug!("stack unwinder push frame {:?}", frame);
        if let Some(frame) = frame {
            return Some(StackUpdateResult {
                frames_opened: Some(frame),
                frames_closed: Vec::new(),
            });
        } else {
            return None;
        }
    }

    pub fn step_uj(&mut self, to_addr: u64) -> Option<StackUpdateResult> {
        let target = to_addr;
        if let Some((start, _)) = self
            .func_symbol_map
            .range(self.curr_prv, self.curr_ctx, target)
        {
            if start == target {
                if let Some(frame) = self.push_frame(self.curr_prv, self.curr_ctx, target) {
                    return Some(StackUpdateResult {
                        frames_opened: Some(frame),
                        frames_closed: Vec::new(),
                    });
                }
            }
        }

        // Otherwise, if it's an indirect jump and we still have frames,
        //    treat it like a return within the unwinding loop.
        let mut closed: Vec<Frame> = Vec::new();
        loop {
            if !self.frame_stack.is_empty() {
                let frame = self.peek_head_frames();
                // if the stack prv is different, we are done
                if frame.symbol.prv != self.curr_prv {
                    return Some(StackUpdateResult {
                        frames_opened: None,
                        frames_closed: closed,
                    });
                }
                // if the stack addr is within the current function, we are done
                if let Some((start, end)) =
                    self.func_symbol_map
                        .range(self.curr_prv, self.curr_ctx, frame.addr)
                {
                    if start <= target && end > target {
                        return Some(StackUpdateResult {
                            frames_opened: None,
                            frames_closed: closed,
                        });
                    }
                }
                closed.push(self.pop_frame().unwrap());
            } else {
                return Some(StackUpdateResult {
                    frames_opened: None,
                    frames_closed: closed,
                });
            }
        }
    }

    pub fn step_trap(
        &mut self,
        reason: &TrapReason,
        prv_arc: &(Prv, Prv),
        to_addr: u64,
        ctx: Option<u64>,
    ) -> Option<StackUpdateResult> {
        self.curr_prv = prv_arc.1;
        match reason {
            TrapReason::Exception | TrapReason::Interrupt => {
                let frame = self.push_frame(self.curr_prv, self.curr_ctx, to_addr);
                if let Some(frame) = frame {
                    return Some(StackUpdateResult {
                        frames_opened: Some(frame),
                        frames_closed: Vec::new(),
                    });
                } else {
                    panic!("failed to push frame for exception or interrupt, got target 0x{:08x}", to_addr);
                }
            }
            TrapReason::Return => {
                let mut closed: Vec<Frame> = Vec::new();
                // if context changed and we are returning to user-space, clear the stack
                if self.curr_prv == Prv::PrvUser && ctx.is_some() && ctx.unwrap() != self.curr_ctx {
                    while let Some(frame) = self.pop_frame() {
                        closed.push(frame);
                    }
                    self.curr_ctx = ctx.unwrap();
                    return Some(StackUpdateResult {
                        frames_opened: None,
                        frames_closed: closed,
                    });
                }
                // otherwise, pop until we find the frame with the same prv as the current prv or lower
                loop {
                    if !self.frame_stack.is_empty() {
                        let frame = self.peek_head_frames();
                        if frame.symbol.prv <= self.curr_prv {
                            return Some(StackUpdateResult {
                                frames_opened: None,
                                frames_closed: closed,
                            });
                        }
                        closed.push(self.pop_frame().unwrap());
                    } else {
                        return Some(StackUpdateResult {
                            frames_opened: None,
                            frames_closed: closed,
                        });
                    }
                }
            }
        }
    }

    pub fn flush(&mut self) -> Option<StackUpdateResult> {
        // just return the frames, in reverse order
        let mut closed = Vec::new();
        while let Some(frame) = self.pop_frame() {
            closed.push(frame);
        }
        return Some(StackUpdateResult {
            frames_opened: None,
            frames_closed: closed,
        });
    }

    pub fn peek_all_frames(&self) -> Vec<Frame> {
        self.frame_stack.clone()
    }

    pub fn peek_head_frames(&self) -> &Frame {
        self.frame_stack.last().unwrap()
    }
}
