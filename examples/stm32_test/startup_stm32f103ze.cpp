/**
 * STM32F103ZET6 Startup Code (C++)
 * - Vector table
 * - Reset handler: data init, BSS zero, C++ static constructors, main()
 * - SystemInit: 72MHz PLL from 8MHz HSE
 *
 * Bare-metal, no HAL/CMSIS dependency
 */

#include <cstdint>
#include <cstring>

// Linker symbols
extern uint32_t _sidata;   // .data init values in Flash
extern uint32_t _sdata;    // .data start in RAM
extern uint32_t _edata;    // .data end in RAM
extern uint32_t _sbss;     // .bss start
extern uint32_t _ebss;     // .bss end
extern uint32_t _estack;   // initial stack pointer

// ============================================================================
// STM32F103 Register Definitions (bare-metal, no CMSIS)
// ============================================================================

// RCC registers
#define RCC_BASE        0x40021000UL
#define RCC_CR          (*(volatile uint32_t*)(RCC_BASE + 0x00))
#define RCC_CFGR        (*(volatile uint32_t*)(RCC_BASE + 0x04))
#define RCC_APB2ENR     (*(volatile uint32_t*)(RCC_BASE + 0x18))

// RCC_CR bits
#define RCC_CR_HSEON    (1U << 16)
#define RCC_CR_HSERDY   (1U << 17)
#define RCC_CR_PLLON    (1U << 24)
#define RCC_CR_PLLRDY   (1U << 25)

// RCC_CFGR bits
#define RCC_CFGR_SW_PLL     (0x2U << 0)   // PLL as system clock
#define RCC_CFGR_SWS_PLL    (0x2U << 2)   // PLL confirmed
#define RCC_CFGR_HPRE_DIV1  (0x0U << 4)   // AHB = SYSCLK
#define RCC_CFGR_PPRE1_DIV2 (0x4U << 8)   // APB1 = HCLK/2 (36MHz max)
#define RCC_CFGR_PPRE2_DIV1 (0x0U << 11)  // APB2 = HCLK
#define RCC_CFGR_PLLSRC_HSE (1U << 16)    // PLL source = HSE
#define RCC_CFGR_PLLMUL9    (0x7U << 18)  // PLL × 9 (8 × 9 = 72MHz)

// RCC_APB2ENR bits
#define RCC_APB2ENR_IOPAEN  (1U << 2)     // GPIOA clock
#define RCC_APB2ENR_USART1EN (1U << 14)   // USART1 clock

// Flash registers (wait states for 72MHz)
#define FLASH_BASE_ADDR 0x40022000UL
#define FLASH_ACR       (*(volatile uint32_t*)(FLASH_BASE_ADDR + 0x00))
#define FLASH_ACR_LATENCY_2 (0x2U)        // 2 wait states for 48-72MHz
#define FLASH_ACR_PRFTBE    (1U << 4)     // Prefetch buffer enable

// SCB (for VTOR)
#define SCB_VTOR        (*(volatile uint32_t*)(0xE000ED08UL))

// DWT (cycle counter for precise timing)
#define DWT_CTRL        (*(volatile uint32_t*)(0xE0001000UL))
#define DWT_CYCCNT      (*(volatile uint32_t*)(0xE0001004UL))
#define CoreDebug_DEMCR (*(volatile uint32_t*)(0xE000EDFCUL))
#define DWT_CTRL_CYCCNTENA (1U << 0)
#define DEMCR_TRCENA       (1U << 24)

// ============================================================================
// SystemInit: Configure 72MHz from 8MHz HSE via PLL
// ============================================================================
static void SystemInit() {
    // Flash: 2 wait states for 72MHz + prefetch buffer
    FLASH_ACR = FLASH_ACR_LATENCY_2 | FLASH_ACR_PRFTBE;

    // Enable HSE
    RCC_CR |= RCC_CR_HSEON;
    while (!(RCC_CR & RCC_CR_HSERDY)) {}  // Wait for HSE ready

    // Configure PLL: HSE × 9 = 72MHz
    // AHB = 72MHz, APB1 = 36MHz (max), APB2 = 72MHz
    RCC_CFGR = RCC_CFGR_PLLSRC_HSE
             | RCC_CFGR_PLLMUL9
             | RCC_CFGR_HPRE_DIV1
             | RCC_CFGR_PPRE1_DIV2
             | RCC_CFGR_PPRE2_DIV1;

    // Enable PLL
    RCC_CR |= RCC_CR_PLLON;
    while (!(RCC_CR & RCC_CR_PLLRDY)) {}  // Wait for PLL ready

    // Switch system clock to PLL
    RCC_CFGR = (RCC_CFGR & ~0x3U) | RCC_CFGR_SW_PLL;
    while ((RCC_CFGR & 0xCU) != RCC_CFGR_SWS_PLL) {}  // Wait for switch

    // Enable DWT cycle counter for precise timing
    CoreDebug_DEMCR |= DEMCR_TRCENA;
    DWT_CYCCNT = 0;
    DWT_CTRL |= DWT_CTRL_CYCCNTENA;
}

// ============================================================================
// Prototypes
// ============================================================================
extern "C" int main();
extern "C" void Reset_Handler();
extern "C" void Default_Handler();

// C++ static init/fini
typedef void (*init_func_ptr)();
extern init_func_ptr __init_array_start;
extern init_func_ptr __init_array_end;

// ============================================================================
// Default interrupt handler (infinite loop)
// ============================================================================
extern "C" void Default_Handler() {
    while (1) {}
}

// Weak aliases for all ISR handlers
extern "C" {
    void NMI_Handler()          __attribute__((weak, alias("Default_Handler")));
    void HardFault_Handler()    __attribute__((weak, alias("Default_Handler")));
    void MemManage_Handler()    __attribute__((weak, alias("Default_Handler")));
    void BusFault_Handler()     __attribute__((weak, alias("Default_Handler")));
    void UsageFault_Handler()   __attribute__((weak, alias("Default_Handler")));
    void SVC_Handler()          __attribute__((weak, alias("Default_Handler")));
    void DebugMon_Handler()     __attribute__((weak, alias("Default_Handler")));
    void PendSV_Handler()       __attribute__((weak, alias("Default_Handler")));
    void SysTick_Handler()      __attribute__((weak, alias("Default_Handler")));
}

// ============================================================================
// Vector Table (placed at 0x08000000 via linker)
// ============================================================================
__attribute__((section(".isr_vector"), used))
const uint32_t g_pfnVectors[] = {
    (uint32_t)&_estack,            // Initial SP
    (uint32_t)Reset_Handler,       // Reset
    (uint32_t)NMI_Handler,         // NMI
    (uint32_t)HardFault_Handler,   // Hard Fault
    (uint32_t)MemManage_Handler,   // MPU Fault
    (uint32_t)BusFault_Handler,    // Bus Fault
    (uint32_t)UsageFault_Handler,  // Usage Fault
    0, 0, 0, 0,                    // Reserved
    (uint32_t)SVC_Handler,         // SVCall
    (uint32_t)DebugMon_Handler,    // Debug Monitor
    0,                             // Reserved
    (uint32_t)PendSV_Handler,      // PendSV
    (uint32_t)SysTick_Handler,     // SysTick
    // IRQ 0..59 = Default_Handler
};

// ============================================================================
// Reset Handler: called on power-on / reset
// ============================================================================
extern "C" void Reset_Handler() {
    // Copy .data from Flash to RAM
    uint32_t* src = &_sidata;
    uint32_t* dst = &_sdata;
    while (dst < &_edata) {
        *dst++ = *src++;
    }

    // Zero .bss
    dst = &_sbss;
    while (dst < &_ebss) {
        *dst++ = 0;
    }

    // Initialize system clock (72MHz)
    SystemInit();

    // Call C++ static constructors
    for (init_func_ptr* p = &__init_array_start; p < &__init_array_end; p++) {
        (*p)();
    }

    // Jump to main
    main();

    // Should never reach here
    while (1) {}
}
