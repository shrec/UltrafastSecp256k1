#!/usr/bin/env python3
"""Fix all 211 code-scanning alerts across 13 files."""

import re
from pathlib import Path

BASE = Path(__file__).resolve().parent


def read(path):
    return (BASE / path).read_text().splitlines(keepends=True)


def save(path, lines):
    (BASE / path).write_text(''.join(lines))


# ============================================================================
# Algorithmic helpers
# ============================================================================

def add_braces(lines, alert_lines_1based, tag=''):
    """Add { } around single-statement bodies. Process bottom-to-top."""
    fixed = 0
    for lnum in sorted(alert_lines_1based, reverse=True):
        idx = lnum - 1
        if idx >= len(lines):
            print(f'  SKIP {tag}L{lnum}: out of range ({len(lines)} lines)')
            continue
        line = lines[idx]
        # Get indentation of the controlling statement
        indent = len(line) - len(line.lstrip())
        indent_str = line[:indent]
        stripped = line.rstrip('\n\r').rstrip()

        # Skip if already has brace at end
        if stripped.endswith('{'):
            print(f'  SKIP {tag}L{lnum}: already has {{')
            continue

        # Find next non-empty line (the body)
        body_idx = idx + 1
        while body_idx < len(lines) and lines[body_idx].strip() == '':
            body_idx += 1

        if body_idx >= len(lines):
            print(f'  SKIP {tag}L{lnum}: no body line found')
            continue

        body_line_stripped = lines[body_idx].lstrip()
        # Skip if body already starts with {
        if body_line_stripped.startswith('{'):
            print(f'  SKIP {tag}L{lnum}: body already has {{')
            continue

        # Apply fix
        lines[idx] = stripped + ' {\n'
        lines.insert(body_idx + 1, indent_str + '}\n')
        fixed += 1

    print(f'  -> {tag}braces fixed: {fixed}')
    return lines


def add_const_to_lines(lines, alert_lines_1based, tag=''):
    """Prepend const to variable declarations, handling range-for loops."""
    fixed = 0
    for lnum in sorted(alert_lines_1based, reverse=True):
        idx = lnum - 1
        if idx >= len(lines):
            continue
        line = lines[idx]
        stripped = line.lstrip()
        leading = line[:len(line) - len(stripped)]

        if stripped.startswith('const '):
            print(f'  SKIP {tag}L{lnum}: already const')
            continue

        # Range-based for loop: for (TYPE var : container) -> for (const TYPE var : container)
        if stripped.startswith('for (') and ':' in stripped:
            # Match: for (TYPE var : ...
            m = re.match(r'(for \()(\w[^:]+: .+)', stripped)
            if m:
                lines[idx] = leading + m.group(1) + 'const ' + m.group(2)
                fixed += 1
                continue

        # Regular declaration
        lines[idx] = leading + 'const ' + stripped
        fixed += 1

    print(f'  -> {tag}const fixed: {fixed}')
    return lines


# ============================================================================
# File: include/ufsecp/ufsecp_impl.cpp
# ============================================================================

def fix_ufsecp_impl():
    path = 'include/ufsecp/ufsecp_impl.cpp'
    print(f'\n=== {path} ===')
    lines = read(path)

    # --- readability-braces-around-statements (59 alerts) ---
    brace_lines = [
        1242, 1245, 1248, 1260, 1274, 1277, 1281, 1294, 1297,
        1300, 1314, 1318, 1322, 1340, 1343, 1345, 1355, 1368,
        1412, 1415, 1431, 1435, 1438, 1457, 1462, 1477, 1481,
        1486, 1514, 1516, 1519, 1522, 1525, 1542, 1545, 1549,
        1567, 1577, 1594, 1691, 1695, 1699, 1701, 1749, 1753,
        1787, 1801, 1831, 1834, 1844, 1856, 1974, 2047, 2068,
        2071, 2076, 2138, 2832, 2834,
    ]
    lines = add_braces(lines, brace_lines, 'ufsecp_impl/')

    # --- misc-const-correctness ---
    const_lines = [1366, 1855, 1905, 2075, 3147, 3167, 3172]
    lines = add_const_to_lines(lines, const_lines, 'ufsecp_impl/')

    # --- modernize-use-auto ---
    # L1573: uint32_t nk = static_cast<uint32_t>(...) -> auto nk = ...
    # L1846: uint32_t cc32 = static_cast<uint32_t>(...) -> auto cc32 = ...
    for lnum in [1573, 1846]:
        idx = lnum - 1
        line = lines[idx]
        m = re.match(r'(\s*)uint32_t (\w+) = (static_cast<uint32_t>\(.+)', line)
        if m:
            lines[idx] = f'{m.group(1)}auto {m.group(2)} = {m.group(3)}'
            print(f'  AUTO: L{lnum}')

    # --- cppcoreguidelines-init-variables ---
    # L1655: uint32_t nk; -> uint32_t nk = 0;
    idx = 1655 - 1
    if '    uint32_t nk;' in lines[idx]:
        lines[idx] = lines[idx].replace('uint32_t nk;', 'uint32_t nk = 0;')
        print('  INIT: L1655')

    # L1706: { uint32_t nk; -> { uint32_t nk = 0;
    idx = 1706 - 1
    if 'uint32_t nk;' in lines[idx]:
        lines[idx] = lines[idx].replace('uint32_t nk;', 'uint32_t nk = 0;')
        print('  INIT: L1706')

    # L1761: same pattern
    idx = 1761 - 1
    if 'uint32_t nk;' in lines[idx]:
        lines[idx] = lines[idx].replace('uint32_t nk;', 'uint32_t nk = 0;')
        print('  INIT: L1761')

    # --- bugprone-implicit-widening-of-multiplication-result ---
    # L1578: keyagg_out + 38 + i * 32 -> keyagg_out + 38 + static_cast<size_t>(i) * 32
    idx = 1578 - 1
    if 'i * 32' in lines[idx] and 'static_cast<size_t>(i)' not in lines[idx]:
        lines[idx] = lines[idx].replace(
            'keyagg_out + 38 + i * 32',
            'keyagg_out + 38 + static_cast<size_t>(i) * 32'
        )
        print('  WIDENING: L1578')

    save(path, lines)
    print(f'  Saved {path} ({len(lines)} lines)')


# ============================================================================
# File: cpu/src/bip39.cpp
# ============================================================================

def fix_bip39():
    path = 'cpu/src/bip39.cpp'
    print(f'\n=== {path} ===')
    lines = read(path)

    brace_lines = [49, 50, 93, 110, 117, 138, 140, 150, 171, 196, 200, 223,
                   246, 269, 273]
    lines = add_braces(lines, brace_lines, 'bip39/')

    const_lines = [33, 46, 47, 97, 126, 127, 128, 129, 136, 145,
                   182, 183, 184, 185, 191, 193, 194, 199,
                   255, 256, 257, 258, 264, 266, 267, 272]
    lines = add_const_to_lines(lines, const_lines, 'bip39/')

    # --- cppcoreguidelines-init-variables ---
    # L137: some variable, need to find it
    idx = 137 - 1
    line = lines[idx]
    # Pattern: TYPE var; (uninitialized) - add = 0 or = {} or = nullptr
    m = re.match(r'(\s*)((?:int|uint\w*|size_t|bool|char|float|double)\s+\w+);(\s*(?://.*)?)\n', line)
    if m:
        type_and_var = m.group(2).rstrip()
        # Determine default value
        if 'bool' in type_and_var:
            default = 'false'
        elif 'float' in type_and_var or 'double' in type_and_var:
            default = '0.0'
        elif 'char*' in type_and_var or 'uint8_t*' in type_and_var:
            default = 'nullptr'
        else:
            default = '0'
        lines[idx] = f'{m.group(1)}{type_and_var} = {default};{m.group(3)}\n'
        print(f'  INIT: L137 -> added = {default}')
    else:
        print(f'  INIT_SKIP: L137 pattern not matched: {repr(line[:60])}')

    # --- modernize-use-auto ---
    # L191 and L264: iterator/auto type replacement
    for lnum in [191, 264]:
        idx = lnum - 1
        line = lines[idx]
        # Pattern: SomeType::iterator it = or std::vector<...>::iterator it =
        m = re.match(r'(\s*)(\w[\w:<>, *]+::iterator)(\s+\w+\s*=.+)', line)
        if m:
            lines[idx] = f'{m.group(1)}auto{m.group(3)}'
            print(f'  AUTO: L{lnum}')
        else:
            # Try: SomeType it = container.begin()
            m2 = re.match(r'(\s*)(\w[\w:<>, *]+\*?)(\s+\w+\s*=\s*\w.+\.begin\(\).+)', line)
            if m2:
                lines[idx] = f'{m2.group(1)}auto{m2.group(3)}'
                print(f'  AUTO: L{lnum}')
            else:
                print(f'  AUTO_SKIP: L{lnum}: {repr(line[:60])}')

    # --- cert-err33-c (unchecked fclose return) ---
    # L34: std::fclose(f); -> (void)std::fclose(f);
    idx = 34 - 1
    line = lines[idx]
    if 'std::fclose' in line and '(void)' not in line:
        lines[idx] = line.replace('std::fclose', '(void)std::fclose')
        print('  ERR33: L34 fclose')

    save(path, lines)
    print(f'  Saved {path} ({len(lines)} lines)')


# ============================================================================
# File: cpu/src/zk.cpp
# ============================================================================

def fix_zk():
    path = 'cpu/src/zk.cpp'
    print(f'\n=== {path} ===')
    lines = read(path)

    brace_lines = [45, 68, 381, 415, 423, 481, 503, 610, 615, 619, 623,
                   664, 668, 675, 686, 688, 720, 785]
    lines = add_braces(lines, brace_lines, 'zk/')

    const_lines = [359, 363, 446, 448, 500, 642, 661]
    lines = add_const_to_lines(lines, const_lines, 'zk/')

    save(path, lines)
    print(f'  Saved {path} ({len(lines)} lines)')


# ============================================================================
# File: cpu/src/message_signing.cpp
# ============================================================================

def fix_message_signing():
    path = 'cpu/src/message_signing.cpp'
    print(f'\n=== {path} ===')
    lines = read(path)

    brace_lines = [30, 35]
    lines = add_braces(lines, brace_lines, 'msg_signing/')

    const_lines = [65, 152, 153, 154, 155, 159, 193, 196]
    lines = add_const_to_lines(lines, const_lines, 'msg_signing/')

    save(path, lines)
    print(f'  Saved {path} ({len(lines)} lines)')


# ============================================================================
# File: cpu/src/eth_signing.cpp
# ============================================================================

def fix_eth_signing():
    path = 'cpu/src/eth_signing.cpp'
    print(f'\n=== {path} ===')
    lines = read(path)

    # --- misc-unused-using-decls: L16 'using fast::Point;' ---
    idx = 16 - 1
    if 'using fast::Point' in lines[idx]:
        lines[idx] = ''  # Remove the line (keep blank to preserve line numbers)
        # Actually remove the line entirely
        lines[idx] = '\n'
        # Better: just delete and shift
        del lines[idx]
        # Now const_lines will shift by -1
        print('  UNUSED-USING: L16 removed')
        # After removal, adjust const lines
        const_lines = [95, 96]  # shifted from [96, 97]
    else:
        print(f'  UNUSED-USING SKIP: L16: {repr(lines[idx][:50])}')
        const_lines = [96, 97]

    lines = add_const_to_lines(lines, const_lines, 'eth_signing/')

    save(path, lines)
    print(f'  Saved {path} ({len(lines)} lines)')


# ============================================================================
# File: cpu/src/address.cpp
# ============================================================================

def fix_address():
    path = 'cpu/src/address.cpp'
    print(f'\n=== {path} ===')
    lines = read(path)

    # L516: for (char c : prefix) -> for (const char c : prefix)
    # L527: std::uint8_t version_byte = ... -> const std::uint8_t version_byte = ...
    # L527: also modernize-use-auto -> auto version_byte = ...
    const_lines = [516, 527]
    lines = add_const_to_lines(lines, const_lines, 'address/')

    # L527: modernize-use-auto: const std::uint8_t version_byte = static_cast<...>
    # -> const auto version_byte = static_cast<...>
    # This is handled by add_const adding 'const', but we also need to change the type
    # Actually the modernize-use-auto wants: 'auto version_byte = static_cast<std::uint8_t>(...)'
    # And const-correctness wants: 'const ... version_byte = ...'
    # Combined: 'const auto version_byte = static_cast<std::uint8_t>(...)'
    # Let's check what add_const_to_lines did for L527:
    # Line 527 was: std::uint8_t version_byte = static_cast<std::uint8_t>(type << 3);
    # After add_const: const std::uint8_t version_byte = ...
    # But we also want to replace std::uint8_t with auto for modernize-use-auto:
    # Find current state of L527 (0-indexed: 526, but const_lines processed in reverse,
    # so L516 was processed first (higher reverse order), then L527)
    # Actually both were processed with const_lines = [516, 527], processed in reverse: 527, 516
    # After const processing, L527 has 'const std::uint8_t version_byte = ...'
    # Now apply modernize-use-auto: replace 'const std::uint8_t' with 'const auto'
    idx = 527 - 1
    if idx < len(lines):
        line = lines[idx]
        if 'const std::uint8_t version_byte' in line:
            lines[idx] = line.replace('const std::uint8_t version_byte',
                                       'const auto version_byte')
            print('  AUTO: L527')
        elif 'const auto version_byte' in line:
            print('  AUTO: L527 already auto')
        else:
            print(f'  AUTO_SKIP: L527: {repr(line[:60])}')

    save(path, lines)
    print(f'  Saved {path} ({len(lines)} lines)')


# ============================================================================
# File: cpu/src/wallet.cpp
# ============================================================================

def fix_wallet():
    path = 'cpu/src/wallet.cpp'
    print(f'\n=== {path} ===')
    lines = read(path)

    # L150, L171: bugprone-misplaced-widening-cast
    # Pattern: static_cast<std::uint64_t>(27 + rsig.recid)
    # Fix: static_cast<std::uint64_t>(27) + static_cast<std::uint64_t>(rsig.recid)
    for lnum in [150, 171]:
        idx = lnum - 1
        if idx >= len(lines):
            continue
        line = lines[idx]
        if 'static_cast<std::uint64_t>(27 + rsig.recid)' in line:
            lines[idx] = line.replace(
                'static_cast<std::uint64_t>(27 + rsig.recid)',
                'static_cast<std::uint64_t>(27) + static_cast<std::uint64_t>(rsig.recid)'
            )
            print(f'  WIDEN: L{lnum}')
        else:
            print(f'  WIDEN_SKIP: L{lnum}: {repr(line[:60])}')

    save(path, lines)
    print(f'  Saved {path} ({len(lines)} lines)')


# ============================================================================
# File: cpu/src/coin_address.cpp
# ============================================================================

def fix_coin_address():
    path = 'cpu/src/coin_address.cpp'
    print(f'\n=== {path} ===')
    lines = read(path)

    # L170: std::string prefix = testnet ? ... -> const std::string prefix = ...
    const_lines = [170]
    lines = add_const_to_lines(lines, const_lines, 'coin_address/')

    save(path, lines)
    print(f'  Saved {path} ({len(lines)} lines)')


# ============================================================================
# File: cpu/tests/test_bip39.cpp
# ============================================================================

# Helper function for replacing sscanf with strtoul in hex_to_bytes
HEX_TO_BYTES_SSCANF_BIP39 = '''\
static void hex_to_bytes(const char* hex, uint8_t* out, size_t len) {
    for (size_t i = 0; i < len; ++i) {
        unsigned int byte = 0;
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif
        std::sscanf(hex + 2 * i, "%02x", &byte);
#ifdef __clang__
#pragma clang diagnostic pop
#endif
        out[i] = static_cast<uint8_t>(byte);
    }
}'''

HEX_TO_BYTES_STRTOUL_BIP39 = '''\
static void hex_to_bytes(const char* hex, uint8_t* out, size_t len) {
    for (size_t i = 0; i < len; ++i) {
        char pair[3] = { hex[2 * i], hex[2 * i + 1], '\\0' };
        char* endptr = nullptr;
        const unsigned long val = std::strtoul(pair, &endptr, 16);
        out[i] = (endptr == pair + 2) ? static_cast<uint8_t>(val) : 0;
    }
}'''

BYTES_TO_HEX_OLD = '''\
static std::string bytes_to_hex(const uint8_t* data, size_t len) {
    std::string result;
    result.reserve(len * 2);
    for (size_t i = 0; i < len; ++i) {
        char buf[3];
        std::snprintf(buf, sizeof(buf), "%02x", data[i]);
        result += buf;
    }
    return result;
}'''

BYTES_TO_HEX_NEW = '''\
static std::string bytes_to_hex(const uint8_t* data, size_t len) {
    std::string result;
    result.reserve(len * 2);
    for (size_t i = 0; i < len; ++i) {
        char buf[3];
        (void)std::snprintf(buf, sizeof(buf), "%02x", data[i]);
        result += buf;
    }
    return result;
}'''


def fix_test_bip39():
    path = 'cpu/tests/test_bip39.cpp'
    print(f'\n=== {path} ===')
    content = (BASE / path).read_text()

    # cert-err33-c + cert-err34-c: replace sscanf with strtoul
    if HEX_TO_BYTES_SSCANF_BIP39 in content:
        content = content.replace(HEX_TO_BYTES_SSCANF_BIP39, HEX_TO_BYTES_STRTOUL_BIP39)
        print('  ERR34: hex_to_bytes sscanf -> strtoul')
    else:
        print('  ERR34_SKIP: hex_to_bytes sscanf pattern not found')

    # cert-err33-c: snprintf return unchecked
    if BYTES_TO_HEX_OLD in content:
        content = content.replace(BYTES_TO_HEX_OLD, BYTES_TO_HEX_NEW)
        print('  ERR33: bytes_to_hex snprintf -> (void)snprintf')
    else:
        print('  ERR33_SKIP: bytes_to_hex pattern not found')

    # clang-analyzer-core.NullDereference at L99
    # CHECK(wl != nullptr, ...) then wl[0] - add explicit if
    old_null = '    CHECK(wl != nullptr, "wordlist not null");\n    CHECK(std::strcmp(wl[0]'
    new_null = '    CHECK(wl != nullptr, "wordlist not null");\n    if (!wl) { return; }\n    CHECK(std::strcmp(wl[0]'
    if old_null in content:
        content = content.replace(old_null, new_null)
        print('  NULL_DEREF: L99 added null guard')
    else:
        print('  NULL_DEREF_SKIP: pattern not found')

    (BASE / path).write_text(content)
    # Now add const to specific lines
    lines = read(path)

    # After the sscanf->strtoul replacement, L32 changes. The line numbers may shift.
    # The original file had 393 lines. After replacing 14-line block with 7-line block
    # and 9-line block with 9-line block (same), the const lines may shift.
    # Let's handle const by string pattern instead.
    # L238, L252, L264: std::string hex = bytes_to_hex(...) -> const std::string hex = ...
    for idx in range(len(lines)):
        line = lines[idx]
        stripped = line.lstrip()
        if stripped.startswith('std::string hex = bytes_to_hex('):
            leading = line[:len(line) - len(stripped)]
            lines[idx] = leading + 'const ' + stripped
            print(f'  CONST: L{idx+1} std::string hex')

    # L340, L352, L365: for (char c : mnemonic) -> for (const char c : mnemonic)
    for idx in range(len(lines)):
        line = lines[idx]
        if 'for (char c : mnemonic)' in line:
            lines[idx] = line.replace('for (char c : mnemonic)',
                                       'for (const char c : mnemonic)')
            print(f'  CONST: L{idx+1} for (char c : mnemonic)')

    save(path, lines)
    print(f'  Saved {path} ({len(lines)} lines)')


# ============================================================================
# File: cpu/tests/test_ethereum.cpp
# ============================================================================

HEX_TO_BYTES_SSCANF_ETH = '''\
static void hex_to_bytes(const char* hex, uint8_t* out, size_t len) {
    for (size_t i = 0; i < len; ++i) {
        unsigned int byte = 0;
        if (std::sscanf(hex + i * 2, "%02x", &byte) != 1) byte = 0;
        out[i] = static_cast<uint8_t>(byte);
    }
}'''

HEX_TO_BYTES_STRTOUL_ETH = '''\
static void hex_to_bytes(const char* hex, uint8_t* out, size_t len) {
    for (size_t i = 0; i < len; ++i) {
        char pair[3] = { hex[i * 2], hex[i * 2 + 1], '\\0' };
        char* endptr = nullptr;
        const unsigned long val = std::strtoul(pair, &endptr, 16);
        out[i] = (endptr == pair + 2) ? static_cast<uint8_t>(val) : 0;
    }
}'''

SNPRINTF_ETH_OLD = '        std::snprintf(buf, sizeof(buf), "Round-trip chain_id=%lu (%s)",'
SNPRINTF_ETH_NEW = '        (void)std::snprintf(buf, sizeof(buf), "Round-trip chain_id=%lu (%s)",'


def fix_test_ethereum():
    path = 'cpu/tests/test_ethereum.cpp'
    print(f'\n=== {path} ===')
    content = (BASE / path).read_text()

    # cert-err34-c: sscanf -> strtoul
    if HEX_TO_BYTES_SSCANF_ETH in content:
        content = content.replace(HEX_TO_BYTES_SSCANF_ETH, HEX_TO_BYTES_STRTOUL_ETH)
        print('  ERR34: hex_to_bytes sscanf -> strtoul')
    else:
        print('  ERR34_SKIP: hex_to_bytes pattern not found')

    # cert-err33-c at L352: snprintf return unchecked
    if SNPRINTF_ETH_OLD in content:
        content = content.replace(SNPRINTF_ETH_OLD, SNPRINTF_ETH_NEW)
        print('  ERR33: snprintf -> (void)snprintf')
    else:
        print('  ERR33_SKIP: snprintf pattern not found')

    # readability-simplify-boolean-expr: extract conditions to named bools
    # L189: ASSERT_TRUE(sig.v == 27 || sig.v == 28, "legacy v should be 27 or 28");
    # Fix: const bool v_ok = (sig.v == 27 || sig.v == 28); ASSERT_TRUE(v_ok, ...);
    content = content.replace(
        '    ASSERT_TRUE(sig.v == 27 || sig.v == 28, "legacy v should be 27 or 28");',
        '    {\n        const bool v_ok = (sig.v == 27 || sig.v == 28);\n        ASSERT_TRUE(v_ok, "legacy v should be 27 or 28");\n    }'
    )
    content = content.replace(
        '    ASSERT_TRUE(sig2.v == 37 || sig2.v == 38, "EIP-155 v should be 37 or 38");',
        '    {\n        const bool v2_ok = (sig2.v == 37 || sig2.v == 38);\n        ASSERT_TRUE(v2_ok, "EIP-155 v should be 37 or 38");\n    }'
    )
    content = content.replace(
        '    ASSERT_TRUE(sig.v == 27 || sig.v == 28, "v should be 27 or 28");',
        '    {\n        const bool v_ok2 = (sig.v == 27 || sig.v == 28);\n        ASSERT_TRUE(v_ok2, "v should be 27 or 28");\n    }'
    )
    print('  SIMPLIFY-BOOL: test_ethereum sig.v checks')

    (BASE / path).write_text(content)

    # Add const to variable declarations (by pattern)
    lines = read(path)

    # Find and fix const alerts: Point pk = ..., Scalar sk = ..., auto vars, etc.
    # L226: Point pk = ... -> const Point pk
    # L264: std::array<...> zero{} - this is const alert? Let me check
    # Actually the const alerts at L226, L264, L287, L302, L309, L317, L333
    # are all variable declarations that should be const
    const_patterns = [
        'Point pk = ',
        'Point pk2 = ',
        'auto expected_addr = ',
        'auto addr = ',
        'auto addr2 = ',
        'std::array<uint8_t, 32> hash{};',
        'std::array<uint8_t, 32> wrong_hash{};',
        'bool wrong = ',
        'bool wrong2 = ',
    ]
    # Instead, use line numbers after adjusting for line-number shifts from replacements
    # The simplify-bool fix added 3 blocks (each +4 lines = 3 lines inserted per block = +9 total)
    # But let's use pattern matching instead of line numbers

    # Pattern: find lines with variable declarations that are const-alerting
    # Based on the alert line context I read:
    # L226: Point pk = Point::generator().scalar_mul(sk);
    # L264: std::array<uint8_t, 32> zero{};
    # L287: Point pk = ...
    # L302: bool valid = ...
    # L309: bool wrong = ...
    # L317: bool wrong2 = ...
    # L333: Point pk = ...

    for idx in range(len(lines)):
        line = lines[idx]
        stripped = line.lstrip()
        leading = line[:len(line) - len(stripped)]

        if stripped.startswith('const '):
            continue

        # Point pk = ... (not already const)
        if re.match(r'Point pk\d? = ', stripped) and not stripped.startswith('const '):
            lines[idx] = leading + 'const ' + stripped
            print(f'  CONST: L{idx+1} Point pk')
        elif re.match(r'(bool (valid|wrong\d?|r_zero|s_zero|all_zero)) = ', stripped):
            lines[idx] = leading + 'const ' + stripped
            print(f'  CONST: L{idx+1} bool')
        elif re.match(r'std::array<uint8_t, 32> (hash|wrong_hash|zero)\{\}', stripped):
            lines[idx] = leading + 'const ' + stripped
            print(f'  CONST: L{idx+1} array')
        elif re.match(r'auto expected_addr = ethernet_address_bytes', stripped) or \
             re.match(r'auto expected_addr = ethereum_address_bytes', stripped):
            lines[idx] = leading + 'const ' + stripped
            print(f'  CONST: L{idx+1} auto expected_addr')

    save(path, lines)
    print(f'  Saved {path} ({len(lines)} lines)')


# ============================================================================
# File: cpu/tests/test_wallet.cpp
# ============================================================================

HEX_TO_BYTES_SSCANF_WALLET = '''\
static void hex_to_bytes(const char* hex, uint8_t* out, size_t len) {
    for (size_t i = 0; i < len; ++i) {
        unsigned int byte = 0;
        if (std::sscanf(hex + i * 2, "%02x", &byte) != 1) byte = 0;
        out[i] = static_cast<uint8_t>(byte);
    }
}'''

HEX_TO_BYTES_STRTOUL_WALLET = '''\
static void hex_to_bytes(const char* hex, uint8_t* out, size_t len) {
    for (size_t i = 0; i < len; ++i) {
        char pair[3] = { hex[i * 2], hex[i * 2 + 1], '\\0' };
        char* endptr = nullptr;
        const unsigned long val = std::strtoul(pair, &endptr, 16);
        out[i] = (endptr == pair + 2) ? static_cast<uint8_t>(val) : 0;
    }
}'''


def fix_test_wallet():
    path = 'cpu/tests/test_wallet.cpp'
    print(f'\n=== {path} ===')
    content = (BASE / path).read_text()

    # misc-unused-using-decls: L45 'using fast::Point;'
    if 'using fast::Point;\n' in content:
        content = content.replace('using fast::Point;\n', '')
        print('  UNUSED-USING: removed using fast::Point')
    else:
        print('  UNUSED-USING SKIP: using fast::Point not found')

    # cert-err34-c: sscanf -> strtoul
    if HEX_TO_BYTES_SSCANF_WALLET in content:
        content = content.replace(HEX_TO_BYTES_SSCANF_WALLET, HEX_TO_BYTES_STRTOUL_WALLET)
        print('  ERR34: hex_to_bytes sscanf -> strtoul')
    else:
        print('  ERR34_SKIP: hex_to_bytes sscanf pattern not found')

    # readability-simplify-boolean-expr: extract to named bools
    # L197: ASSERT_TRUE(wif[0] == 'K' || wif[0] == 'L', "WIF starts with K or L");
    content = content.replace(
        '    ASSERT_TRUE(wif[0] == \'K\' || wif[0] == \'L\', "WIF starts with K or L");',
        '    {\n        const bool wif_prefix_ok = (wif[0] == \'K\' || wif[0] == \'L\');\n        ASSERT_TRUE(wif_prefix_ok, "WIF starts with K or L");\n    }'
    )
    # L397: ASSERT_TRUE(sig.recid >= 0 && sig.recid <= 3, "valid recid");
    content = content.replace(
        '    ASSERT_TRUE(sig.recid >= 0 && sig.recid <= 3, "valid recid");',
        '    {\n        const bool recid_ok = (sig.recid >= 0 && sig.recid <= 3);\n        ASSERT_TRUE(recid_ok, "valid recid");\n    }'
    )
    # L505: ASSERT_TRUE(!btc.empty() && !ltc.empty() && !doge.empty(), "all non-empty");
    content = content.replace(
        '    ASSERT_TRUE(!btc.empty() && !ltc.empty() && !doge.empty(), "all non-empty");',
        '    {\n        const bool coins_non_empty = !btc.empty() && !ltc.empty() && !doge.empty();\n        ASSERT_TRUE(coins_non_empty, "all non-empty");\n    }'
    )
    # L602: multi-line ASSERT_TRUE
    content = content.replace(
        '    ASSERT_TRUE(!p2pkh.empty() && !p2wpkh.empty() && !p2sh.empty() && !p2tr.empty(),\n                "all non-empty");',
        '    {\n        const bool addrs_non_empty = !p2pkh.empty() && !p2wpkh.empty() && !p2sh.empty() && !p2tr.empty();\n        ASSERT_TRUE(addrs_non_empty, "all non-empty");\n    }'
    )
    print('  SIMPLIFY-BOOL: 4 bool expressions extracted')

    (BASE / path).write_text(content)

    # Add const to variable declarations (by pattern matching)
    lines = read(path)
    for idx in range(len(lines)):
        line = lines[idx]
        stripped = line.lstrip()
        leading = line[:len(line) - len(stripped)]

        if stripped.startswith('const '):
            continue

        # L290: size_t msg_len = sizeof(msg) - 1;
        # L293: bool ok = bitcoin_verify_message(...)
        # L298: bool bad = bitcoin_verify_message(...)
        # L314: size_t msg_len = sizeof(msg) - 1;
        # L336: size_t msg_len = sizeof(msg) - 1;
        # L366: size_t msg_len = sizeof(msg) - 1;
        # L369: bool verified = verify_message(...)
        # L418: size_t msg_len = sizeof(msg) - 1;
        # L437: size_t msg_len = sizeof(msg) - 1;
        if re.match(r'size_t msg_len = sizeof\(msg\) - 1;', stripped):
            lines[idx] = leading + 'const ' + stripped
            print(f'  CONST: L{idx+1} size_t msg_len')
        elif re.match(r'bool ok = bitcoin_verify_message\(', stripped):
            lines[idx] = leading + 'const ' + stripped
            print(f'  CONST: L{idx+1} bool ok')
        elif re.match(r'bool bad = bitcoin_verify_message\(', stripped):
            lines[idx] = leading + 'const ' + stripped
            print(f'  CONST: L{idx+1} bool bad')
        elif re.match(r'bool verified = verify_message\(', stripped):
            lines[idx] = leading + 'const ' + stripped
            print(f'  CONST: L{idx+1} bool verified')

    save(path, lines)
    print(f'  Saved {path} ({len(lines)} lines)')


# ============================================================================
# File: cpu/tests/test_zk.cpp
# ============================================================================

def fix_test_zk():
    path = 'cpu/tests/test_zk.cpp'
    print(f'\n=== {path} ===')
    lines = read(path)

    # All 10 alerts are misc-const-correctness at:
    # L60, L95, L103, L117, L134, L267, L281, L295, L309, L325
    const_lines = [60, 95, 103, 117, 134, 267, 281, 295, 309, 325]
    lines = add_const_to_lines(lines, const_lines, 'test_zk/')

    save(path, lines)
    print(f'  Saved {path} ({len(lines)} lines)')


# ============================================================================
# File: audit/test_ffi_round_trip.cpp
# ============================================================================

def fix_test_ffi():
    path = 'audit/test_ffi_round_trip.cpp'
    print(f'\n=== {path} ===')
    content = (BASE / path).read_text()

    # L1055: misc-redundant-expression (tautological check)
    # Fix: remove the first redundant half of the OR expression
    old_check = (
        'CHECK(ufsecp_bip39_validate(ctx, "abandon abandon abandon abandon abandon abandon '
        'abandon abandon abandon abandon abandon abandon") != UFSECP_OK\n'
        '          || ufsecp_bip39_validate(ctx, "abandon abandon abandon abandon '
        'abandon abandon abandon abandon abandon abandon abandon abandon") == UFSECP_OK,\n'
        '          "bip39_validate accepts or rejects known mnemonic");'
    )
    new_check = (
        'CHECK(ufsecp_bip39_validate(ctx, "abandon abandon abandon abandon abandon abandon '
        'abandon abandon abandon abandon abandon abandon") == UFSECP_OK,\n'
        '          "bip39_validate accepts valid 12-word mnemonic");'
    )
    if old_check in content:
        content = content.replace(old_check, new_check)
        print('  REDUNDANT: L1055 tautological check fixed')
    else:
        print('  REDUNDANT_SKIP: L1055 exact pattern not found, trying partial match')
        # Try a partial match
        old_pattern = 'bip39_validate accepts or rejects known mnemonic'
        if old_pattern in content:
            # Need to find and replace the surrounding context
            # Use regex for multi-line replacement
            pattern = re.compile(
                r'CHECK\(ufsecp_bip39_validate\(ctx,\s*"abandon[^"]+"\)\s*!=\s*UFSECP_OK\s*\n'
                r'\s*\|\|\s*ufsecp_bip39_validate\(ctx,\s*"abandon[^"]+"\)\s*==\s*UFSECP_OK,\s*\n'
                r'\s*"bip39_validate accepts or rejects known mnemonic"\)',
                re.MULTILINE
            )
            replacement = (
                'CHECK(ufsecp_bip39_validate(ctx, "abandon abandon abandon abandon abandon abandon '
                'abandon abandon abandon abandon abandon abandon") == UFSECP_OK,\n'
                '          "bip39_validate accepts valid 12-word mnemonic")'
            )
            content, n = pattern.subn(replacement, content)
            if n:
                print(f'  REDUNDANT: L1055 fixed via regex ({n} replacement)')
            else:
                print('  REDUNDANT_FAIL: could not fix L1055')

    (BASE / path).write_text(content)
    lines = read(path)

    # L1317: size_t msg_len = 15; -> const size_t msg_len = 15;
    # L1538: bool match = ... -> const bool match = ...
    # Use pattern matching since line numbers may have shifted
    for idx in range(len(lines)):
        line = lines[idx]
        stripped = line.lstrip()
        leading = line[:len(line) - len(stripped)]

        if stripped.startswith('const '):
            continue

        if stripped == 'size_t msg_len = 15;\n':
            lines[idx] = leading + 'const ' + stripped
            print(f'  CONST: L{idx+1} size_t msg_len = 15')
        elif stripped.startswith('bool match = (std::memcmp('):
            lines[idx] = leading + 'const ' + stripped
            print(f'  CONST: L{idx+1} bool match')

    save(path, lines)
    print(f'  Saved {path} ({len(lines)} lines)')


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    print('Fix Round 4: resolving 211 code-scanning alerts')
    print('=' * 60)

    fix_ufsecp_impl()
    fix_bip39()
    fix_zk()
    fix_message_signing()
    fix_eth_signing()
    fix_address()
    fix_wallet()
    fix_coin_address()
    fix_test_bip39()
    fix_test_ethereum()
    fix_test_wallet()
    fix_test_zk()
    fix_test_ffi()

    print('\n' + '=' * 60)
    print('Done. Check brace balance:')
    files = [
        'include/ufsecp/ufsecp_impl.cpp',
        'cpu/src/bip39.cpp',
        'cpu/src/zk.cpp',
        'cpu/src/message_signing.cpp',
        'cpu/src/eth_signing.cpp',
        'cpu/src/address.cpp',
        'cpu/src/wallet.cpp',
        'cpu/src/coin_address.cpp',
        'cpu/tests/test_bip39.cpp',
        'cpu/tests/test_ethereum.cpp',
        'cpu/tests/test_wallet.cpp',
        'cpu/tests/test_zk.cpp',
        'audit/test_ffi_round_trip.cpp',
    ]
    all_ok = True
    for f in files:
        try:
            text = (BASE / f).read_text()
            opens = text.count('{')
            closes = text.count('}')
            ok = opens == closes
            status = 'OK' if ok else f'MISMATCH ({opens} vs {closes})'
            print(f'  {f}: {status}')
            if not ok:
                all_ok = False
        except Exception as e:
            print(f'  {f}: ERROR {e}')
            all_ok = False

    if all_ok:
        print('\nAll brace counts balanced.')
    else:
        print('\nWARNING: Some files have mismatched braces!')
