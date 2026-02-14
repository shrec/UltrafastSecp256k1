/**
 * STM32F103ZET6 Syscalls
 * Redirects printf() → USART1 (PA9=TX, 115200 baud)
 * Provides _sbrk for malloc/new
 *
 * Bare-metal, no HAL dependency
 */

#include <cstdint>
#include <cerrno>
#include <sys/stat.h>

// ============================================================================
// USART1 Register Definitions
// ============================================================================
#define GPIOA_BASE      0x40010800UL
#define GPIOA_CRH       (*(volatile uint32_t*)(GPIOA_BASE + 0x04))

#define USART1_BASE     0x40013800UL
#define USART1_SR       (*(volatile uint32_t*)(USART1_BASE + 0x00))
#define USART1_DR       (*(volatile uint32_t*)(USART1_BASE + 0x04))
#define USART1_BRR      (*(volatile uint32_t*)(USART1_BASE + 0x08))
#define USART1_CR1      (*(volatile uint32_t*)(USART1_BASE + 0x0C))

// RCC
#define RCC_BASE        0x40021000UL
#define RCC_APB2ENR     (*(volatile uint32_t*)(RCC_BASE + 0x18))
#define RCC_APB2ENR_IOPAEN  (1U << 2)
#define RCC_APB2ENR_USART1EN (1U << 14)

// USART_SR bits
#define USART_SR_TXE    (1U << 7)

// USART_CR1 bits
#define USART_CR1_UE    (1U << 13)  // USART enable
#define USART_CR1_TE    (1U << 3)   // Transmitter enable
#define USART_CR1_RE    (1U << 2)   // Receiver enable

static bool uart_initialized = false;

/**
 * Initialize USART1 on PA9 (TX) at 115200 baud
 * Call this AFTER SystemInit (72MHz clock configured)
 */
void uart_init() {
    if (uart_initialized) return;

    // Enable GPIOA + USART1 clocks
    RCC_APB2ENR |= RCC_APB2ENR_IOPAEN | RCC_APB2ENR_USART1EN;

    // PA9 = USART1_TX: Alternate function push-pull, 50MHz
    // CRH controls pins 8-15, PA9 is bits [7:4]
    uint32_t crh = GPIOA_CRH;
    crh &= ~(0xFU << 4);           // Clear PA9 config
    crh |=  (0xBU << 4);           // AF push-pull, 50MHz (MODE=11, CNF=10)
    GPIOA_CRH = crh;

    // PA10 = USART1_RX: Input floating (default) — no change needed

    // Baud rate: 72MHz / 115200 = 625 = 0x271
    USART1_BRR = 625;  // 72000000 / 115200 = 625.0

    // Enable USART: TX + RX + UE
    USART1_CR1 = USART_CR1_UE | USART_CR1_TE | USART_CR1_RE;

    uart_initialized = true;
}

/**
 * Send one byte over USART1 (blocking)
 */
static inline void uart_putc(char c) {
    while (!(USART1_SR & USART_SR_TXE)) {}  // Wait for TXE
    USART1_DR = static_cast<uint32_t>(c);
}

// ============================================================================
// Newlib syscall stubs (redirect stdio to USART1)
// ============================================================================

extern "C" {

// Redirect _write to USART1
int _write(int fd, const char* buf, int len) {
    (void)fd;
    if (!uart_initialized) uart_init();
    for (int i = 0; i < len; i++) {
        if (buf[i] == '\n') uart_putc('\r');  // CR+LF
        uart_putc(buf[i]);
    }
    return len;
}

// Heap management
extern uint32_t end;     // Defined by linker (_end)
extern uint32_t _estack;

static uint8_t* heap_ptr = nullptr;

void* _sbrk(ptrdiff_t incr) {
    if (heap_ptr == nullptr) {
        heap_ptr = reinterpret_cast<uint8_t*>(&end);
    }

    uint8_t* prev = heap_ptr;
    uint8_t* new_ptr = heap_ptr + incr;

    // Check against stack (leave 1KB margin)
    if (new_ptr >= reinterpret_cast<uint8_t*>(&_estack) - 1024) {
        errno = ENOMEM;
        return reinterpret_cast<void*>(-1);
    }

    heap_ptr = new_ptr;
    return prev;
}

// No-op stubs
int _close(int fd) { (void)fd; return -1; }
int _fstat(int fd, struct stat* st) { (void)fd; st->st_mode = S_IFCHR; return 0; }
int _isatty(int fd) { (void)fd; return 1; }
int _lseek(int fd, int ptr, int dir) { (void)fd; (void)ptr; (void)dir; return 0; }
int _read(int fd, char* buf, int len) { (void)fd; (void)buf; (void)len; return 0; }
void _exit(int status) { (void)status; while (1) {} }
int _kill(int pid, int sig) { (void)pid; (void)sig; return -1; }
int _getpid() { return 1; }

} // extern "C"
