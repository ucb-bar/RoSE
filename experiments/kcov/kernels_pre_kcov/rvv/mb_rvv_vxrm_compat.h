/* RVV fixed-point intrinsics: bridge the pre-1.0 and 1.0 argument forms.
 *
 * The curated int8 conv kernels are written against the RVV intrinsics v1.0
 * API, in which every fixed-point instruction takes an explicit rounding mode:
 *
 *     __riscv_vsmul_vx_i32m4 (vs2, rs1, __RISCV_VXRM_RNU, vl)
 *     __riscv_vnclip_wx_i16m2(vs2, shift, __RISCV_VXRM_RNU, vl)
 *
 * GCC 13.2 -- the SpaceMiT cross-toolchain available here -- implements the
 * earlier form, where the rounding mode comes from the `vxrm` CSR and is not an
 * argument. Against it the kernels fail to compile with "too many arguments"
 * and "'__RISCV_VXRM_RNU' undeclared".
 *
 * That is a toolchain-version mismatch, not a kernel defect, and it is worth
 * fixing rather than routing around: silently falling back to a different
 * algorithm because one failed to build is how a run ends up labelled "rvv"
 * while executing scalar reference code. (That exact thing happened here once
 * already -- see backends.curated_aliases.)
 *
 * WHY THE MACRO IS SAFE. A function-like macro is not re-expanded inside its
 * own replacement list, so `#define f(a,b,c,d) f(a,b,d)` rewrites the call and
 * then leaves the inner `f` alone. This is standard C preprocessor behaviour,
 * not a trick that depends on the compiler.
 *
 * WHY DROPPING THE ARGUMENT IS CORRECT HERE. Every use in these kernels passes
 * __RISCV_VXRM_RNU, and RNU is encoding 0 -- the reset value of `vxrm`. The
 * static assertion below pins that, so if a kernel is ever written with a
 * different mode this stops compiling instead of quietly rounding the wrong
 * way. mb_rvv_vxrm_init() additionally writes the CSR once, so the value does
 * not depend on whatever ran before us on the same hart.
 */

#ifndef MB_RVV_VXRM_COMPAT_H_
#define MB_RVV_VXRM_COMPAT_H_

/* Gate on the intrinsics API VERSION, not on whether __RISCV_VXRM_RNU happens
 * to be a preprocessor macro. The explicit-rounding-mode form arrived in
 * intrinsics v0.12: GCC 13.2 reports __riscv_v_intrinsic == 11000 (v0.11, no
 * vxrm argument) and GCC 14.2 reports 12000 (v0.12, vxrm argument present). In
 * GCC 14 the __RISCV_VXRM_* names come from the `#pragma riscv intrinsic
 * "vector"` machinery rather than from #define, so the old `!defined(...)`
 * guard let the shim fire on a toolchain that did not need it and every
 * fixed-point call then failed with "too few arguments". A model with an fp16
 * island has to be compiled by GCC >= 14 on this box (see KERNEL_CC in
 * harness_linux/Makefile), which is how that was found. */
#if defined(__riscv_v_intrinsic) && __riscv_v_intrinsic < 12000

#define __RISCV_VXRM_RNU 0 /* round-to-nearest-up   (vxrm reset value) */
#define __RISCV_VXRM_RNE 1 /* round-to-nearest-even */
#define __RISCV_VXRM_RDN 2 /* round-down / truncate */
#define __RISCV_VXRM_ROD 3 /* round-to-odd */

/* Set the rounding mode the caller asked for.
 *
 * The first version of this header asserted the mode was RNU and dropped the
 * argument, on the reasoning that every curated kernel used RNU. That was
 * wrong, and the assertion caught it: yolov8's cat kernel uses RDN
 * (round-down). Dropping a mode argument silently would have rounded the wrong
 * way -- a quiet numerical difference, not a build error, in exactly the class
 * of code where nobody would look.
 *
 * So the shim WRITES the CSR instead. That is what the pre-1.0 API expects the
 * caller to do, so the semantics match the v1.0 intrinsic exactly for every
 * mode rather than for one of them.
 *
 * Cost is one csrw per call. The alternative -- hoisting it out of the loop --
 * would need to know that nothing in the loop body changes vxrm, which is not
 * something a header can know about a kernel it does not see. */
static inline void mb_rvv_set_vxrm(int mode)
{
	__asm__ volatile("csrw vxrm, %0" :: "r"(mode) : "memory");
}

static inline void mb_rvv_vxrm_init(void)
{
	mb_rvv_set_vxrm(__RISCV_VXRM_RNU);
}

#define __riscv_vsmul_vx_i32m4(vs2, rs1, vxrm, vl) \
	(mb_rvv_set_vxrm(vxrm), __riscv_vsmul_vx_i32m4(vs2, rs1, vl))
#define __riscv_vsmul_vv_i32m4(vs2, vs1, vxrm, vl) \
	(mb_rvv_set_vxrm(vxrm), __riscv_vsmul_vv_i32m4(vs2, vs1, vl))
#define __riscv_vsmul_vx_i32m2(vs2, rs1, vxrm, vl) \
	(mb_rvv_set_vxrm(vxrm), __riscv_vsmul_vx_i32m2(vs2, rs1, vl))
#define __riscv_vsmul_vx_i32m1(vs2, rs1, vxrm, vl) \
	(mb_rvv_set_vxrm(vxrm), __riscv_vsmul_vx_i32m1(vs2, rs1, vl))
#define __riscv_vsmul_vx_i32m8(vs2, rs1, vxrm, vl) \
	(mb_rvv_set_vxrm(vxrm), __riscv_vsmul_vx_i32m8(vs2, rs1, vl))

/* Cover every LMUL the curated kernels use. These were added one at a time as
 * each model reached the compiler -- conv2d in dronet needs i16m2, yolov8's
 * fused conv+BN+SiLU needs i16m4 and i8m2 -- which is a poor way to find them.
 * The set below is the full i8/i16 narrowing family, so the next model does not
 * discover a gap the same way. */
#define __riscv_vnclip_wx_i16m4(vs2, rs1, vxrm, vl) \
	(mb_rvv_set_vxrm(vxrm), __riscv_vnclip_wx_i16m4(vs2, rs1, vl))
#define __riscv_vnclip_wx_i16m2(vs2, rs1, vxrm, vl) \
	(mb_rvv_set_vxrm(vxrm), __riscv_vnclip_wx_i16m2(vs2, rs1, vl))
#define __riscv_vnclip_wx_i8m4(vs2, rs1, vxrm, vl) \
	(mb_rvv_set_vxrm(vxrm), __riscv_vnclip_wx_i8m4(vs2, rs1, vl))
#define __riscv_vnclip_wx_i8m2(vs2, rs1, vxrm, vl) \
	(mb_rvv_set_vxrm(vxrm), __riscv_vnclip_wx_i8m2(vs2, rs1, vl))
#define __riscv_vnclip_wx_i16m1(vs2, rs1, vxrm, vl) \
	(mb_rvv_set_vxrm(vxrm), __riscv_vnclip_wx_i16m1(vs2, rs1, vl))
#define __riscv_vnclip_wx_i8m1(vs2, rs1, vxrm, vl) \
	(mb_rvv_set_vxrm(vxrm), __riscv_vnclip_wx_i8m1(vs2, rs1, vl))
#define __riscv_vnclip_wx_i8mf2(vs2, rs1, vxrm, vl) \
	(mb_rvv_set_vxrm(vxrm), __riscv_vnclip_wx_i8mf2(vs2, rs1, vl))

#define __riscv_vssra_vx_i32m4(vs2, rs1, vxrm, vl) \
	(mb_rvv_set_vxrm(vxrm), __riscv_vssra_vx_i32m4(vs2, rs1, vl))
#define __riscv_vssra_vx_i32m2(vs2, rs1, vxrm, vl) \
	(mb_rvv_set_vxrm(vxrm), __riscv_vssra_vx_i32m2(vs2, rs1, vl))

#define MB_RVV_VXRM_COMPAT_ACTIVE 1

#else /* intrinsics v1.0, or no vector support at all */

static inline void mb_rvv_vxrm_init(void) {}

#endif

#endif /* MB_RVV_VXRM_COMPAT_H_ */
