// =============================================================================
// Minimal OpenCL Platform Header (cl_platform.h)
// =============================================================================
// Provides type definitions needed by cl.h
// Based on Khronos OpenCL 1.2 spec (permissive license)
// =============================================================================

#ifndef __CL_PLATFORM_H
#define __CL_PLATFORM_H

#ifdef __cplusplus
extern "C" {
#endif

// Calling conventions
#ifdef _WIN32
#define CL_API_ENTRY
#define CL_API_CALL     __stdcall
#define CL_CALLBACK     __stdcall
#else
#define CL_API_ENTRY
#define CL_API_CALL
#define CL_CALLBACK
#endif

#define CL_API_SUFFIX__VERSION_1_0
#define CL_API_SUFFIX__VERSION_1_1
#define CL_API_SUFFIX__VERSION_1_2
#define CL_EXT_SUFFIX__VERSION_1_0_DEPRECATED
#define CL_EXT_SUFFIX__VERSION_1_1_DEPRECATED
#define CL_EXT_SUFFIX__VERSION_1_2_DEPRECATED
#define CL_EXT_SUFFIX__VERSION_2_0_DEPRECATED

// Scalar types
#include <stdint.h>
#include <stddef.h>

typedef int8_t          cl_char;
typedef uint8_t         cl_uchar;
typedef int16_t         cl_short;
typedef uint16_t        cl_ushort;
typedef int32_t         cl_int;
typedef uint32_t        cl_uint;
typedef int64_t         cl_long;
typedef uint64_t        cl_ulong;
typedef uint16_t        cl_half;
typedef float           cl_float;
typedef double          cl_double;

// Platform-dependent integer for bitfield
typedef cl_ulong        cl_bitfield;

// OpenCL object types (opaque pointers)
typedef struct _cl_platform_id *    cl_platform_id;
typedef struct _cl_device_id *      cl_device_id;
typedef struct _cl_context *        cl_context;
typedef struct _cl_command_queue *   cl_command_queue;
typedef struct _cl_mem *            cl_mem;
typedef struct _cl_program *        cl_program;
typedef struct _cl_kernel *         cl_kernel;
typedef struct _cl_event *          cl_event;
typedef struct _cl_sampler *        cl_sampler;

// Boolean
typedef cl_uint             cl_bool;
#define CL_FALSE            0
#define CL_TRUE             1
#define CL_BLOCKING         CL_TRUE
#define CL_NON_BLOCKING     CL_FALSE

// Property types
typedef cl_bitfield         cl_device_type;
typedef cl_uint             cl_platform_info;
typedef cl_uint             cl_device_info;
typedef cl_bitfield         cl_device_fp_config;
typedef cl_uint             cl_device_mem_cache_type;
typedef cl_uint             cl_device_local_mem_type;
typedef cl_bitfield         cl_device_exec_capabilities;
typedef cl_bitfield         cl_command_queue_properties;
typedef intptr_t            cl_context_properties;
typedef cl_uint             cl_context_info;
typedef cl_bitfield         cl_queue_properties;
typedef cl_uint             cl_command_queue_info;
typedef cl_uint             cl_channel_order;
typedef cl_uint             cl_channel_type;
typedef cl_bitfield         cl_mem_flags;
typedef cl_uint             cl_mem_object_type;
typedef cl_uint             cl_mem_info;
typedef cl_bitfield         cl_mem_migration_flags;
typedef cl_uint             cl_image_info;
typedef cl_uint             cl_buffer_create_type;
typedef cl_uint             cl_addressing_mode;
typedef cl_uint             cl_filter_mode;
typedef cl_uint             cl_sampler_info;
typedef cl_bitfield         cl_map_flags;
typedef cl_uint             cl_program_info;
typedef cl_uint             cl_program_build_info;
typedef cl_uint             cl_program_binary_type;
typedef cl_int              cl_build_status;
typedef cl_uint             cl_kernel_info;
typedef cl_uint             cl_kernel_arg_info;
typedef cl_uint             cl_kernel_arg_address_qualifier;
typedef cl_uint             cl_kernel_arg_access_qualifier;
typedef cl_bitfield         cl_kernel_arg_type_qualifier;
typedef cl_uint             cl_kernel_work_group_info;
typedef cl_uint             cl_event_info;
typedef cl_uint             cl_command_type;
typedef cl_uint             cl_profiling_info;

// Image format
typedef struct {
    cl_channel_order        image_channel_order;
    cl_channel_type         image_channel_data_type;
} cl_image_format;

// Image description (OpenCL 1.2)
typedef struct {
    cl_mem_object_type      image_type;
    size_t                  image_width;
    size_t                  image_height;
    size_t                  image_depth;
    size_t                  image_array_size;
    size_t                  image_row_pitch;
    size_t                  image_slice_pitch;
    cl_uint                 num_mip_levels;
    cl_uint                 num_samples;
    cl_mem                  buffer;
} cl_image_desc;

// Buffer region
typedef struct {
    size_t                  origin;
    size_t                  size;
} cl_buffer_region;

#ifdef __cplusplus
}
#endif

#endif /* __CL_PLATFORM_H */
