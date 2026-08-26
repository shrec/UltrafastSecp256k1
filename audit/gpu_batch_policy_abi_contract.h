/*
 * gpu_batch_policy_abi_contract.h
 *
 * Task-local, portable C/C++ ABI contract for the GPU batch-path policy
 * advisory query (task GPU_ABI_NATIVE_CORE_264, topic gpu_abi_native_core_v6).
 *
 * AUTHORITY RECONCILIATION (see report.md for full record)
 * -------------------------------------------------------------------------
 * This card is AUTHORITATIVE for the query *entry contract* of this prototype:
 * a FOUR-argument, size_t-carrying, status-returning function
 *
 *     lb_status lb_gpu_batch_policy_query(const void *request,
 *                                         size_t      request_size,
 *                                         void       *result,
 *                                         size_t      result_size);
 *
 * The frozen research spec docs/GPU_BATCH_POLICY_API_012.md
 * (GPU_BATCH_POLICY_API_012) supplies COMPATIBLE constant / enum / layout
 * facts only (operation, backend, path, advice, calibration numbering and the
 * struct field ordering).  That spec sketches a LEGACY TWO-argument boolean
 * signature:
 *
 *     bool lb_gpu_batch_policy_query_v1(const lb_gpu_batch_policy_request_v1*,
 *                                       lb_gpu_batch_policy_result_v1*);
 *
 * The legacy two-argument bool form is recorded here as a DISCREPANCY only.
 * It is NOT implemented and NOT claimed by this prototype: the caller passes
 * explicit physical sizes and receives an explicit status, because a caller's
 * physical byte count cannot be recovered from a bare pointer.
 *
 * Freestanding: this header depends only on freestanding headers
 * (<stdint.h>, <stddef.h>, and for C <stdalign.h>).  No hosted libc header.
 */
#ifndef GPU_BATCH_POLICY_ABI_CONTRACT_H
#define GPU_BATCH_POLICY_ABI_CONTRACT_H

#include <stdint.h>   /* uint32_t, uint64_t, UINT32_C          (freestanding) */
#include <stddef.h>   /* size_t, offsetof                      (freestanding) */

#if !defined(__cplusplus)
#include <stdalign.h> /* alignas, alignof                      (freestanding) */
#endif

#if defined(__cplusplus)
#define LB_STATIC_ASSERT(cond, msg) static_assert(cond, msg)
#else
#define LB_STATIC_ASSERT(cond, msg) _Static_assert(cond, msg)
#endif

#if defined(__cplusplus)
extern "C" {
#endif

/* ------------------------------------------------------------------------- *
 * Every public integer domain is its own uint32_t typedef with UINT32_C
 * constants.  Fixed width keeps request/result layout identical on ILP32 and
 * LP64, and removes implementation-defined enum sizing from the ABI.
 * ------------------------------------------------------------------------- */

/* ABI version accepted by this prototype (compatible with API_012 V1). */
#define LB_GPU_BATCH_POLICY_ABI_V1 UINT32_C(1)

/* Return status of the query.  Values are stable and part of the ABI. */
typedef uint32_t lb_status;
#define LB_STATUS_OK                  UINT32_C(0)
#define LB_STATUS_ERR_ADDRESS_WRAP    UINT32_C(1) /* request/result range wraps  */
#define LB_STATUS_ERR_ADDRESS_OVERLAP UINT32_C(2) /* physical ranges overlap     */
#define LB_STATUS_ERR_RESULT_NULL     UINT32_C(3) /* result null or physical < 4 */
#define LB_STATUS_ERR_RESULT_ABI      UINT32_C(4) /* result undersized / bad ABI */
#define LB_STATUS_ERR_REQUEST_ABI     UINT32_C(5) /* request null/undersized/ABI */
#define LB_STATUS_ERR_REQUEST_FIELD   UINT32_C(6) /* operation/mask/mbz/overflow */

/* Requested crypto operation (numbering from API_012). */
typedef uint32_t lb_batch_operation;
#define LB_BATCH_OP_INVALID        UINT32_C(0)
#define LB_BATCH_OP_ECDSA_VERIFY   UINT32_C(1)
#define LB_BATCH_OP_SCHNORR_VERIFY UINT32_C(2)
#define LB_BATCH_OP_GENERIC_MSM    UINT32_C(3)

/* Concrete compute backend identity (numbering from API_012). */
typedef uint32_t lb_compute_backend;
#define LB_BACKEND_NONE      UINT32_C(0)
#define LB_BACKEND_CPU       UINT32_C(1)
#define LB_BACKEND_CUDA      UINT32_C(2)
#define LB_BACKEND_OPENCL    UINT32_C(3)
#define LB_BACKEND_METAL     UINT32_C(4)
#define LB_BACKEND_OTHER_GPU UINT32_C(5)

/* Selected execution path (numbering from API_012). */
typedef uint32_t lb_batch_path;
#define LB_PATH_UNAVAILABLE UINT32_C(0)
#define LB_PATH_CPU_INLINE  UINT32_C(1)
#define LB_PATH_CPU_BATCH   UINT32_C(2)
#define LB_PATH_GPU_BATCH   UINT32_C(3)

/* Profitability advice (numbering from API_012). */
typedef uint32_t lb_batch_advice;
#define LB_ADVICE_UNKNOWN        UINT32_C(0)
#define LB_ADVICE_NOT_BENEFICIAL UINT32_C(1)
#define LB_ADVICE_BENEFICIAL     UINT32_C(2)

/* Calibration provenance (numbering from API_012). */
typedef uint32_t lb_calibration_state;
#define LB_CALIBRATION_NONE             UINT32_C(0)
#define LB_CALIBRATION_MEASURED_CURRENT UINT32_C(1)
#define LB_CALIBRATION_MEASURED_CACHED  UINT32_C(2)

/*
 * Caller backend-permission mask (own uint32_t domain).  A set bit PERMITS the
 * backend; a clear bit EXCLUDES it.  Selection may never pick a backend whose
 * bit is absent, and CPU capability never overrides a caller CPU exclusion.
 */
typedef uint32_t lb_backend_mask;
#define LB_BACKEND_MASK_NONE      UINT32_C(0x00000000)
#define LB_BACKEND_MASK_CPU       UINT32_C(0x00000001)
#define LB_BACKEND_MASK_CUDA      UINT32_C(0x00000002)
#define LB_BACKEND_MASK_OPENCL    UINT32_C(0x00000004)
#define LB_BACKEND_MASK_METAL     UINT32_C(0x00000008)
#define LB_BACKEND_MASK_OTHER_GPU UINT32_C(0x00000010)
#define LB_BACKEND_MASK_KNOWN                                                  \
    (LB_BACKEND_MASK_CPU | LB_BACKEND_MASK_CUDA | LB_BACKEND_MASK_OPENCL |     \
     LB_BACKEND_MASK_METAL | LB_BACKEND_MASK_OTHER_GPU)

/* ------------------------------------------------------------------------- *
 * Request: align8, size 32, field offsets 0,4,8,12,16,24,28.
 * ------------------------------------------------------------------------- */
typedef struct lb_gpu_batch_policy_request {
    alignas(8)
    uint32_t           struct_size;   /* offset 0  : caller-declared size (RD) */
    uint32_t           abi_version;   /* offset 4  : must equal V1             */
    lb_batch_operation operation;     /* offset 8                              */
    lb_backend_mask    backend_mask;  /* offset 12 : permitted backends        */
    uint64_t           item_count;    /* offset 16 : != 0                      */
    uint32_t           concurrency;   /* offset 24 : != 0                      */
    uint32_t           reserved;      /* offset 28 : MBZ                       */
} lb_gpu_batch_policy_request;

/* ------------------------------------------------------------------------- *
 * Result: align8, size 72, field offsets
 *   0,4,8,16,20,24,28,32,40,48,56,64.
 * ------------------------------------------------------------------------- */
typedef struct lb_gpu_batch_policy_result {
    alignas(8)
    uint32_t             struct_size;            /* offset 0  : declared (D)   */
    uint32_t             abi_version;            /* offset 4  : must equal V1  */
    uint64_t             available_backends;     /* offset 8  : capability bits*/
    lb_compute_backend   selected_backend;       /* offset 16                 */
    lb_batch_path        selected_path;          /* offset 20                 */
    lb_calibration_state calibration;            /* offset 24                 */
    lb_batch_advice      predicted_benefit;      /* offset 28                 */
    uint64_t             confidence_percent;     /* offset 32 : 0..100, 0=unk  */
    uint64_t             predicted_inline_ns;    /* offset 40                 */
    uint64_t             predicted_selected_ns;  /* offset 48                 */
    uint64_t             calibration_age_ms;     /* offset 56                 */
    uint64_t             calibration_generation; /* offset 64                 */
} lb_gpu_batch_policy_result;

/* Canonical sizes and byte offsets used by the alignment-agnostic core. */
#define LB_REQUEST_SIZE UINT32_C(32)
#define LB_RESULT_SIZE  UINT32_C(72)

#define LB_REQ_OFF_STRUCT_SIZE  0u
#define LB_REQ_OFF_ABI_VERSION  4u
#define LB_REQ_OFF_OPERATION    8u
#define LB_REQ_OFF_BACKEND_MASK 12u
#define LB_REQ_OFF_ITEM_COUNT   16u
#define LB_REQ_OFF_CONCURRENCY  24u
#define LB_REQ_OFF_RESERVED     28u

#define LB_RES_OFF_STRUCT_SIZE            0u
#define LB_RES_OFF_ABI_VERSION           4u
#define LB_RES_OFF_AVAILABLE_BACKENDS    8u
#define LB_RES_OFF_SELECTED_BACKEND      16u
#define LB_RES_OFF_SELECTED_PATH         20u
#define LB_RES_OFF_CALIBRATION           24u
#define LB_RES_OFF_PREDICTED_BENEFIT     28u
#define LB_RES_OFF_CONFIDENCE            32u
#define LB_RES_OFF_PREDICTED_INLINE_NS   40u
#define LB_RES_OFF_PREDICTED_SELECTED_NS 48u
#define LB_RES_OFF_CALIBRATION_AGE_MS    56u
#define LB_RES_OFF_CALIBRATION_GENERATION 64u

/* ------------------------------------------------------------------------- *
 * Layout equality assertions.  Fixed-width fields make these identical under
 * -m32 (ILP32); the explicit align8 forces struct alignment 8 even where the
 * i386 SysV ABI would otherwise align a 64-bit field to 4.
 * ------------------------------------------------------------------------- */
LB_STATIC_ASSERT(sizeof(lb_gpu_batch_policy_request) == 32,
                 "request must be exactly 32 bytes");
LB_STATIC_ASSERT(alignof(lb_gpu_batch_policy_request) == 8,
                 "request must be 8-byte aligned");
LB_STATIC_ASSERT(offsetof(lb_gpu_batch_policy_request, struct_size) == 0, "req@0");
LB_STATIC_ASSERT(offsetof(lb_gpu_batch_policy_request, abi_version) == 4, "req@4");
LB_STATIC_ASSERT(offsetof(lb_gpu_batch_policy_request, operation) == 8, "req@8");
LB_STATIC_ASSERT(offsetof(lb_gpu_batch_policy_request, backend_mask) == 12, "req@12");
LB_STATIC_ASSERT(offsetof(lb_gpu_batch_policy_request, item_count) == 16, "req@16");
LB_STATIC_ASSERT(offsetof(lb_gpu_batch_policy_request, concurrency) == 24, "req@24");
LB_STATIC_ASSERT(offsetof(lb_gpu_batch_policy_request, reserved) == 28, "req@28");

LB_STATIC_ASSERT(sizeof(lb_gpu_batch_policy_result) == 72,
                 "result must be exactly 72 bytes");
LB_STATIC_ASSERT(alignof(lb_gpu_batch_policy_result) == 8,
                 "result must be 8-byte aligned");
LB_STATIC_ASSERT(offsetof(lb_gpu_batch_policy_result, struct_size) == 0, "res@0");
LB_STATIC_ASSERT(offsetof(lb_gpu_batch_policy_result, abi_version) == 4, "res@4");
LB_STATIC_ASSERT(offsetof(lb_gpu_batch_policy_result, available_backends) == 8, "res@8");
LB_STATIC_ASSERT(offsetof(lb_gpu_batch_policy_result, selected_backend) == 16, "res@16");
LB_STATIC_ASSERT(offsetof(lb_gpu_batch_policy_result, selected_path) == 20, "res@20");
LB_STATIC_ASSERT(offsetof(lb_gpu_batch_policy_result, calibration) == 24, "res@24");
LB_STATIC_ASSERT(offsetof(lb_gpu_batch_policy_result, predicted_benefit) == 28, "res@28");
LB_STATIC_ASSERT(offsetof(lb_gpu_batch_policy_result, confidence_percent) == 32, "res@32");
LB_STATIC_ASSERT(offsetof(lb_gpu_batch_policy_result, predicted_inline_ns) == 40, "res@40");
LB_STATIC_ASSERT(offsetof(lb_gpu_batch_policy_result, predicted_selected_ns) == 48, "res@48");
LB_STATIC_ASSERT(offsetof(lb_gpu_batch_policy_result, calibration_age_ms) == 56, "res@56");
LB_STATIC_ASSERT(offsetof(lb_gpu_batch_policy_result, calibration_generation) == 64, "res@64");

LB_STATIC_ASSERT(sizeof(lb_gpu_batch_policy_request) == LB_REQUEST_SIZE, "req size macro");
LB_STATIC_ASSERT(sizeof(lb_gpu_batch_policy_result) == LB_RESULT_SIZE, "res size macro");

/* ------------------------------------------------------------------------- *
 * The authoritative four-argument, size-carrying, status-returning query.
 *
 * request/result are opaque byte regions; request_size/result_size are the
 * caller's PHYSICAL byte counts (R and P).  The declared sizes (RD, D) are the
 * struct_size fields read from inside those regions.  The core computes wrap
 * and overlap predicates on the physical ranges before any access, validates
 * the result ABI before request semantics, and touches only min(P,D,72) bytes
 * of an accessible result.  Returns LB_STATUS_OK on success, else an error.
 * ------------------------------------------------------------------------- */
lb_status lb_gpu_batch_policy_query(const void *request,
                                    size_t      request_size,
                                    void       *result,
                                    size_t      result_size);

#if defined(__cplusplus)
} /* extern "C" */
#endif

#endif /* GPU_BATCH_POLICY_ABI_CONTRACT_H */
