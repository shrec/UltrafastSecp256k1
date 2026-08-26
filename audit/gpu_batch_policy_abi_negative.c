/*
 * gpu_batch_policy_abi_negative.c  --  task GPU_ABI_CORE_VALIDATOR_292
 * (topic gpu_abi_core_validator_v4).
 *
 * Independent negative-compilation + runtime C consumer of the A0-accepted
 * GPU_ABI_NATIVE_CORE_264 ABI.  It has THREE personalities selected by the
 * preprocessor so a single translation unit carries every C-side obligation:
 *
 *  (1) CLEAN LAYOUT CONSUMER  (no -DNEG_CASE)
 *      Re-derives, from first principles, the full request32/result72 layout
 *      (sizes, alignment, every field offset, size macros, status/domain
 *      distinctness) as _Static_asserts.  These predicates are written here,
 *      not copied from the core; if the accepted header's layout drifts, this
 *      TU fails to compile.  Compiles under
 *        cc -std=c11 -fsyntax-only
 *        cc -m32 -std=c11 -ffreestanding -fsyntax-only   (freestanding, no libc)
 *
 *  (2) NEGATIVE-COMPILATION PROBE  (-DNEG_CASE=N, 1..8)
 *      Exactly one deliberately-FALSE static assertion is activated.  The
 *      accepted header MUST reject it: the driver requires each NEG_CASE to
 *      FAIL to compile.  A header that silently accepted a weakened ABI would
 *      let one of these compile and be caught.  (These compile-time negatives
 *      are necessary evidence but, per the card, do NOT substitute for the
 *      real isolated compile/run source mutations executed by the Python
 *      driver.)
 *
 *  (3) HOSTED RUNTIME CONSUMER  (__STDC_HOSTED__ && !NEG_CASE)
 *      An additional, independent C consumer that dlopen()s a shared object
 *      exporting lb_gpu_batch_policy_query and actually CALLS the accepted
 *      oracle: a valid CPU request must yield OK with selected_backend=CPU,
 *      selected_path=CPU_INLINE and available=CPU|CUDA; a null result must
 *      yield ERR_RESULT_NULL; a bad-ABI result against a NULL request must
 *      yield ERR_RESULT_ABI (result-first precedence).  It interprets only the
 *      documented ABI, never the oracle's internal predicates.
 */
#include "gpu_batch_policy_abi_contract.h"

#include <stdint.h>
#include <stddef.h>

/* ------------------------------------------------------------------------- *
 * (1) Independent clean layout obligations, re-derived from the documented
 *     ABI (not lifted from the core).  Compiled in every build.
 * ------------------------------------------------------------------------- */
_Static_assert(sizeof(lb_gpu_batch_policy_request) == 32u, "req size 32");
_Static_assert(alignof(lb_gpu_batch_policy_request) == 8u, "req align 8");
_Static_assert(sizeof(lb_gpu_batch_policy_result) == 72u, "res size 72");
_Static_assert(alignof(lb_gpu_batch_policy_result) == 8u, "res align 8");
_Static_assert(sizeof(lb_status) == 4u, "status uint32");
_Static_assert(sizeof(lb_backend_mask) == 4u, "mask uint32");
_Static_assert(LB_REQUEST_SIZE == 32u, "req size macro");
_Static_assert(LB_RESULT_SIZE == 72u, "res size macro");

_Static_assert(offsetof(lb_gpu_batch_policy_request, struct_size) == 0u, "req@0");
_Static_assert(offsetof(lb_gpu_batch_policy_request, abi_version) == 4u, "req@4");
_Static_assert(offsetof(lb_gpu_batch_policy_request, operation) == 8u, "req@8");
_Static_assert(offsetof(lb_gpu_batch_policy_request, backend_mask) == 12u, "req@12");
_Static_assert(offsetof(lb_gpu_batch_policy_request, item_count) == 16u, "req@16");
_Static_assert(offsetof(lb_gpu_batch_policy_request, concurrency) == 24u, "req@24");
_Static_assert(offsetof(lb_gpu_batch_policy_request, reserved) == 28u, "req@28");

_Static_assert(offsetof(lb_gpu_batch_policy_result, struct_size) == 0u, "res@0");
_Static_assert(offsetof(lb_gpu_batch_policy_result, abi_version) == 4u, "res@4");
_Static_assert(offsetof(lb_gpu_batch_policy_result, available_backends) == 8u, "res@8");
_Static_assert(offsetof(lb_gpu_batch_policy_result, selected_backend) == 16u, "res@16");
_Static_assert(offsetof(lb_gpu_batch_policy_result, selected_path) == 20u, "res@20");
_Static_assert(offsetof(lb_gpu_batch_policy_result, calibration) == 24u, "res@24");
_Static_assert(offsetof(lb_gpu_batch_policy_result, predicted_benefit) == 28u, "res@28");
_Static_assert(offsetof(lb_gpu_batch_policy_result, confidence_percent) == 32u, "res@32");
_Static_assert(offsetof(lb_gpu_batch_policy_result, predicted_inline_ns) == 40u, "res@40");
_Static_assert(offsetof(lb_gpu_batch_policy_result, predicted_selected_ns) == 48u, "res@48");
_Static_assert(offsetof(lb_gpu_batch_policy_result, calibration_age_ms) == 56u, "res@56");
_Static_assert(offsetof(lb_gpu_batch_policy_result, calibration_generation) == 64u, "res@64");

/* Status / domain distinctness that the ABI depends on. */
_Static_assert(LB_STATUS_OK == 0u, "OK is 0");
_Static_assert(LB_STATUS_ERR_RESULT_NULL != LB_STATUS_ERR_RESULT_ABI, "result errs distinct");
_Static_assert(LB_BATCH_OP_ECDSA_VERIFY == 1u && LB_BATCH_OP_GENERIC_MSM == 3u, "op range");
_Static_assert(LB_BACKEND_MASK_KNOWN ==
               (LB_BACKEND_MASK_CPU | LB_BACKEND_MASK_CUDA | LB_BACKEND_MASK_OPENCL |
                LB_BACKEND_MASK_METAL | LB_BACKEND_MASK_OTHER_GPU), "mask known");

/* ------------------------------------------------------------------------- *
 * (2) Negative-compilation probes: each must be REJECTED by the header.
 * ------------------------------------------------------------------------- */
#ifdef NEG_CASE
#  if NEG_CASE == 1
_Static_assert(sizeof(lb_gpu_batch_policy_request) == 31u, "NEG1 req size must fail");
#  elif NEG_CASE == 2
_Static_assert(offsetof(lb_gpu_batch_policy_result, selected_path) == 24u,
               "NEG2 selected_path offset must fail");
#  elif NEG_CASE == 3
_Static_assert(LB_RESULT_SIZE == 64u, "NEG3 result-size macro must fail");
#  elif NEG_CASE == 4
_Static_assert(LB_REQUEST_SIZE == 31u, "NEG4 request-size macro must fail");
#  elif NEG_CASE == 5
_Static_assert(alignof(lb_gpu_batch_policy_result) == 4u, "NEG5 result align must fail");
#  elif NEG_CASE == 6
_Static_assert(LB_STATUS_OK == LB_STATUS_ERR_RESULT_ABI, "NEG6 status distinctness must fail");
#  elif NEG_CASE == 7
_Static_assert(offsetof(lb_gpu_batch_policy_result, calibration_generation) == 56u,
               "NEG7 res@64 offset must fail");
#  elif NEG_CASE == 8
_Static_assert(sizeof(lb_gpu_batch_policy_result) == 64u, "NEG8 result size must fail");
#  else
#    error "unknown NEG_CASE"
#  endif
#endif /* NEG_CASE */

/* ------------------------------------------------------------------------- *
 * (3) Hosted runtime consumer that CALLS the accepted oracle.
 * ------------------------------------------------------------------------- */
#if __STDC_HOSTED__ && !defined(NEG_CASE)
#include <stdio.h>
#include <string.h>
#include <dlfcn.h>

typedef lb_status (*query_fn)(const void *, size_t, void *, size_t);

static uint32_t rd_u32(const void *base, size_t off)
{
    uint32_t v;
    memcpy(&v, (const unsigned char *)base + off, sizeof v);
    return v;
}
static void wr_u32(void *base, size_t off, uint32_t v)
{
    memcpy((unsigned char *)base + off, &v, sizeof v);
}
static void wr_u64(void *base, size_t off, uint64_t v)
{
    memcpy((unsigned char *)base + off, &v, sizeof v);
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "usage: %s <shared-object>\n", argv[0]);
        return 2;
    }
    void *h = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!h) { fprintf(stderr, "dlopen: %s\n", dlerror()); return 3; }
    query_fn query = (query_fn)dlsym(h, "lb_gpu_batch_policy_query");
    if (!query) { fprintf(stderr, "dlsym: %s\n", dlerror()); return 4; }

    _Alignas(8) unsigned char req[LB_REQUEST_SIZE];
    _Alignas(8) unsigned char res[LB_RESULT_SIZE];
    memset(req, 0, sizeof req);
    memset(res, 0, sizeof res);

    wr_u32(req, LB_REQ_OFF_STRUCT_SIZE, LB_REQUEST_SIZE);
    wr_u32(req, LB_REQ_OFF_ABI_VERSION, LB_GPU_BATCH_POLICY_ABI_V1);
    wr_u32(req, LB_REQ_OFF_OPERATION, LB_BATCH_OP_ECDSA_VERIFY);
    wr_u32(req, LB_REQ_OFF_BACKEND_MASK, LB_BACKEND_MASK_CPU);
    wr_u64(req, LB_REQ_OFF_ITEM_COUNT, 10u);
    wr_u32(req, LB_REQ_OFF_CONCURRENCY, 2u);
    wr_u32(req, LB_REQ_OFF_RESERVED, 0u);

    wr_u32(res, LB_RES_OFF_STRUCT_SIZE, LB_RESULT_SIZE);
    wr_u32(res, LB_RES_OFF_ABI_VERSION, LB_GPU_BATCH_POLICY_ABI_V1);

    lb_status st = query(req, sizeof req, res, sizeof res);
    if (st != LB_STATUS_OK) {
        fprintf(stderr, "expected OK, got %u\n", (unsigned)st);
        return 5;
    }
    if (rd_u32(res, LB_RES_OFF_SELECTED_BACKEND) != LB_BACKEND_CPU) { return 6; }
    if (rd_u32(res, LB_RES_OFF_SELECTED_PATH) != LB_PATH_CPU_INLINE) { return 7; }
    if (rd_u32(res, LB_RES_OFF_AVAILABLE_BACKENDS) !=
        (LB_BACKEND_MASK_CPU | LB_BACKEND_MASK_CUDA)) { return 8; }

    /* null result -> ERR_RESULT_NULL (no access) */
    if (query(req, sizeof req, NULL, 0u) != LB_STATUS_ERR_RESULT_NULL) { return 9; }

    /* result-first precedence: bad result ABI reported even for a NULL request */
    _Alignas(8) unsigned char bad[LB_RESULT_SIZE];
    memset(bad, 0, sizeof bad);
    wr_u32(bad, LB_RES_OFF_STRUCT_SIZE, LB_RESULT_SIZE);
    wr_u32(bad, LB_RES_OFF_ABI_VERSION, 0xDEADu);
    if (query(NULL, 0u, bad, sizeof bad) != LB_STATUS_ERR_RESULT_ABI) { return 10; }

    dlclose(h);
    printf("NEGATIVE_C_CONSUMER_OK\n");
    return 0;
}
#endif /* __STDC_HOSTED__ && !NEG_CASE */
