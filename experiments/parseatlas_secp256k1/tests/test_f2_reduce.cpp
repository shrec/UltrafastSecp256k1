#include "../probes/f2_reduce.hpp"
#include "secp256k1/scalar.hpp"

#include <boost/multiprecision/cpp_int.hpp>

#include <array>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

using pa_f2::Columns;
using pa_f2::Limbs;
using pa_f2::UInt128;
using pa_f2::Wide5;
using pa_f2::Word;
using boost::multiprecision::cpp_int;
using secp256k1::fast::Scalar;
using Bytes32 = std::array<std::uint8_t, 32>;

constexpr Word seed = 20260905;
constexpr Word random_reducer_cases = 100000;
constexpr Word random_column_cases = 10000;
constexpr Word random_array_cases = 2048;
constexpr Word maximum = ~Word{0};
const cpp_int modulus{"0xfffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd0364141"};
const cpp_int word_base = cpp_int{1} << 64;
const cpp_int width = cpp_int{1} << 256;
const cpp_int wide_width = cpp_int{1} << 320;
const cpp_int complement = width - modulus;

void require(bool condition, const char* message) {
    // These gates intentionally remain active in -DNDEBUG builds.
    if (!condition) throw std::runtime_error(message);
}

template<std::size_t N>
cpp_int to_bigint(const std::array<Word, N>& value) {
    cpp_int result = 0;
    for (std::size_t i = N; i != 0; --i) {
        result *= word_base;
        result += value[i - 1];
    }
    return result;
}

template<std::size_t N>
std::array<Word, N> to_words(cpp_int value) {
    const cpp_int bound = N == 4 ? width : wide_width;
    require((N == 4 || N == 5) && value >= 0 && value < bound,
            "test word conversion outside declared domain");
    const cpp_int mask = word_base - 1;
    std::array<Word, N> result{};
    for (auto& limb : result) {
        limb = (value & mask).template convert_to<Word>();
        value >>= 64;
    }
    return result;
}

cpp_int column_bigint(UInt128 value) {
    cpp_int result = static_cast<Word>(value >> 64);
    result *= word_base;
    result += static_cast<Word>(value);
    return result;
}

UInt128 to_column(const cpp_int& value) {
    require(value >= 0 && value < word_base * word_base,
            "test column conversion outside uint128 domain");
    const Word low = (value & (word_base - 1)).convert_to<Word>();
    const Word high = (value >> 64).convert_to<Word>();
    return (UInt128{high} << 64) | UInt128{low};
}

cpp_int columns_bigint(const Columns& columns) {
    cpp_int result = 0;
    for (unsigned i = 4; i != 0; --i) {
        result *= word_base;
        result += column_bigint(columns[i - 1]);
    }
    return result;
}

template<std::size_t N>
std::array<std::uint8_t, 8 * N> independent_le(const std::array<Word, N>& value) {
    std::array<std::uint8_t, 8 * N> result{};
    for (std::size_t byte = 0; byte < result.size(); ++byte) {
        result[byte] = static_cast<std::uint8_t>(value[byte / 8] >> (8 * (byte % 8)));
    }
    return result;
}

template<std::size_t N>
bool same_bytes(const std::array<Word, N>& a, const std::array<Word, N>& b) {
    return independent_le(a) == independent_le(b);
}

bool same_columns(const Columns& a, const Columns& b) {
    for (unsigned i = 0; i < 4; ++i) if (a[i] != b[i]) return false;
    return true;
}

Word next_random(Word& state) {
    state += UINT64_C(0x9e3779b97f4a7c15);
    Word z = state;
    z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
    return z ^ (z >> 31);
}

Limbs random_canonical(Word& state) {
    Limbs result{};
    do {
        for (auto& limb : result) limb = next_random(state);
    } while (to_bigint(result) >= modulus);
    return result;
}

struct Counts {
    Word reducer_cases = 0;
    Word reducer_boundary_cases = 0;
    Word deliberate_fold_boundary_cases = 0;
    Word reducer_first_fold_carry_cases = 0;
    Word reducer_library_subset_cases = 0;
    Word library_bit_replay_additions = 0;
    Word column_cases = 0;
    Word column_boundary_cases = 0;
    Word synthetic_repeated_array_witnesses = 0;
    Word synthetic_cap_witnesses = 0;
    Word array_cases = 0;
    Word array_boundary_cases = 0;
    Word array_rhs_values = 0;
    Word array_empty_cases = 0;
    Word library_array_replay_additions = 0;
    Word wide_intermediate_checks = 0;
    Word column_component_checks = 0;
    Word normalized_wide_checks = 0;
    Word pipeline_reduce_checks = 0;
    Word wide_final_checks = 0;
    Word columns_final_checks = 0;
    Word canonical_result_checks = 0;
    Word encoder_checks = 0;
    Word public_be_checks = 0;
    Word preserved_input_values = 0;
    Word checked_error_cases = 0;
    Word empty_identity_checks = 0;
    Word negative_wide_bit_checks = 0;
    Word negative_final_bit_checks = 0;
    Word negative_column_bit_checks = 0;
    Word lossy_high_controls = 0;
    Word overflow_loss_controls = 0;
    Word checksum = UINT64_C(14695981039346656037);
};

template<std::size_t N>
void digest_words(const std::array<Word, N>& value, unsigned kind, Counts& counts) {
    counts.checksum ^= kind;
    counts.checksum *= UINT64_C(1099511628211);
    for (const auto byte : independent_le(value)) {
        counts.checksum ^= byte;
        counts.checksum *= UINT64_C(1099511628211);
    }
}

void check_final(const Limbs& actual, const Limbs& expected, Counts& counts) {
    require(same_bytes(actual, expected), "complete final 32 bytes differ from independent oracle");
    require(to_bigint(actual) < modulus && pa_modn::canonical(actual),
            "final residue is not canonical");
    ++counts.canonical_result_checks;
    require(pa_modn::encode_le(actual) == independent_le(expected),
            "candidate encoder differs from independent LE bytes");
    ++counts.encoder_checks;
}

void check_public_be(const Scalar& reference, const Limbs& expected, Counts& counts) {
    Bytes32 expected_be{};
    const auto little = independent_le(expected);
    for (unsigned byte = 0; byte < 32; ++byte) expected_be[byte] = little[31 - byte];
    require(reference.to_bytes() == expected_be, "public Scalar BE bytes differ from oracle");
    ++counts.public_be_checks;
}

Scalar library_reduce_wide(const Wide5& value, Counts& counts) {
    // Independent public-library replay of ALL 320 bits; no truncating
    // Scalar::from_limbs call can silently discard the fifth limb here.
    Scalar result = Scalar::zero();
    const Scalar one = Scalar::one();
    for (unsigned bit = 320; bit != 0; --bit) {
        result += result;
        ++counts.library_bit_replay_additions;
        if ((value[(bit - 1) / 64] >> ((bit - 1) % 64)) & 1) {
            result += one;
            ++counts.library_bit_replay_additions;
        }
    }
    return result;
}

void check_reducer(const Wide5& value, bool library_subset, Counts& counts) {
    const cpp_int integer = to_bigint(value);
    const Limbs expected = to_words<4>(integer % modulus);
    const Wide5 saved = value;
    const Limbs actual = pa_f2::reduce320(value);
    check_final(actual, expected, counts);
    require(same_bytes(value, saved), "reduce320 mutated its input");
    ++counts.preserved_input_values;
    const cpp_int first_fold = (integer % width) + cpp_int{value[4]} * complement;
    if (first_fold >= width) ++counts.reducer_first_fold_carry_cases;
    if (library_subset) {
        const Scalar reference = library_reduce_wide(value, counts);
        check_final(reference.limbs(), expected, counts);
        require(same_bytes(actual, reference.limbs()), "reduce320 differs from real library bit replay");
        check_public_be(reference, expected, counts);
        ++counts.reducer_library_subset_cases;
    }
    digest_words(value, 0, counts);
    digest_words(actual, 1, counts);
    ++counts.reducer_cases;
}

void check_reducer_boundaries(Counts& counts) {
    const auto check = [&](const cpp_int& value) {
        if (value < 0 || value >= wide_width) return;
        check_reducer(to_words<5>(value), true, counts);
        ++counts.reducer_boundary_cases;
    };
    for (int delta = -2; delta <= 2; ++delta) {
        check(cpp_int{delta});
        check(modulus + delta);
        check(width + delta);
        check(wide_width + delta);
    }
    const std::array<cpp_int, 8> multiples{{cpp_int{0}, cpp_int{1}, cpp_int{2}, cpp_int{3},
        cpp_int{1000000000}, word_base / 2, word_base - 1, word_base}};
    for (const auto& multiple : multiples) {
        for (int delta = -2; delta <= 2; ++delta) check(multiple * modulus + delta);
    }
    const std::array<cpp_int, 8> lows{{cpp_int{0}, cpp_int{1}, modulus - 1, modulus,
        modulus + 1, width - 2, width - 1, complement}};
    for (const Word high : std::array<Word, 6>{{0, 1, 2, maximum / 2, maximum - 1, maximum}}) {
        for (const auto& low : lows) check(cpp_int{high} * width + low);
    }
    cpp_int power = 1;
    for (unsigned bit = 0; bit < 320; ++bit) {
        check(power - 1);
        check(power);
        check(power + 1);
        power *= 2;
    }
    Word random = seed ^ UINT64_C(0x8ebc6af09c88c6e3);
    // H*D < 2^193 for every 64-bit H. Choose L so that L+H*D
    // lies immediately below/at/above 2^256; uniform random rarely hits this.
    for (unsigned fixture = 0; fixture < 128; ++fixture) {
        Word high = fixture == 0 ? 1 : (fixture == 1 ? maximum : next_random(random));
        if (high == 0) high = 1;
        for (int delta = -1; delta <= 1; ++delta) {
            const cpp_int low = width - cpp_int{high} * complement + delta;
            require(low >= 0 && low < width, "first-fold boundary fixture outside low256");
            check_reducer(to_words<5>(cpp_int{high} * width + low), true, counts);
            ++counts.deliberate_fold_boundary_cases;
        }
    }
}

void check_column_case(const Columns& columns, Counts& counts) {
    const Columns saved = columns;
    for (unsigned i = 0; i < 4; ++i) {
        require(column_bigint(columns[i]) <= column_bigint(pa_f2::max_column_sum),
                "generated out-of-bound column fixture");
        ++counts.column_component_checks;
    }
    const cpp_int total = columns_bigint(columns);
    const Wide5 expected = to_words<5>(total);
    const Wide5 actual = pa_f2::normalize_columns(columns);
    require(same_bytes(actual, expected) && to_bigint(actual) == total,
            "column normalization lost part of full integer");
    require(same_columns(columns, saved), "normalize_columns mutated input columns");
    ++counts.preserved_input_values;
    ++counts.normalized_wide_checks;
    check_final(pa_f2::reduce320(actual), to_words<4>(total % modulus), counts);
    ++counts.pipeline_reduce_checks;
    digest_words(actual, 2, counts);
    ++counts.column_cases;
}

void check_column_boundaries(Counts& counts) {
    const UInt128 bound = pa_f2::max_column_sum;
    const std::array<UInt128, 8> values{{0, 1, UInt128{maximum}, UInt128{maximum} + 1,
                                       bound / 2, bound - 2, bound - 1, bound}};
    for (const auto value : values) {
        Columns columns{};
        columns.fill(value);
        check_column_case(columns, counts);
        ++counts.column_boundary_cases;
        for (unsigned index = 0; index < 4; ++index) {
            columns = {};
            columns[index] = value;
            check_column_case(columns, counts);
            ++counts.column_boundary_cases;
        }
    }
    // All 3^4 boundary-column combinations. These are normalization-domain
    // witnesses, not a claim that four independently maximal limbs can arise
    // simultaneously from a repeated canonical scalar array.
    for (unsigned code = 0; code < 81; ++code) {
        unsigned remaining = code;
        Columns columns{};
        for (auto& column : columns) {
            const unsigned digit = remaining % 3;
            remaining /= 3;
            column = digit == 0 ? 0 : (digit == 1 ? bound - 1 : bound);
        }
        check_column_case(columns, counts);
        ++counts.column_boundary_cases;
    }
    const Limbs n_minus_one = to_words<4>(modulus - 1);
    const std::array<Limbs, 4> repeated{{n_minus_one, Limbs{maximum, maximum, maximum, 0},
        Limbs{maximum, maximum, maximum - 2, maximum}, Limbs{0, 0, 0, maximum}}};
    const std::array<std::size_t, 8> counts_to_model{{0, 1, 3, 255, 256, 257,
        pa_f2::max_count - 1, pa_f2::max_count}};
    for (const auto count : counts_to_model) for (const auto& value : repeated) {
        require(to_bigint(value) < modulus, "synthetic repeated value is not canonical");
        Columns columns{};
        for (unsigned i = 0; i < 4; ++i) {
            const cpp_int exact = cpp_int{n_minus_one[i]} + cpp_int{count} * value[i];
            columns[i] = to_column(exact);
            require(column_bigint(columns[i]) == exact, "synthetic column construction lost bits");
        }
        const cpp_int exact_total = to_bigint(n_minus_one) + cpp_int{count} * to_bigint(value);
        require(columns_bigint(columns) == exact_total, "synthetic array/columns integer mismatch");
        check_column_case(columns, counts);
        ++counts.synthetic_repeated_array_witnesses;
        if (count == pa_f2::max_count) ++counts.synthetic_cap_witnesses;
    }
}

void check_preservation(const Limbs& x0, const Limbs& saved_x0,
                        const std::vector<Limbs>& rhs,
                        const std::vector<Limbs>& saved_rhs, Counts& counts) {
    require(same_bytes(x0, saved_x0) && rhs.size() == saved_rhs.size(),
            "array kernel changed x0/input size");
    ++counts.preserved_input_values;
    for (std::size_t i = 0; i < rhs.size(); ++i) {
        require(same_bytes(rhs[i], saved_rhs[i]), "array kernel changed RHS");
        ++counts.preserved_input_values;
    }
}

void check_array(Limbs x0, std::vector<Limbs> rhs, Counts& counts) {
    require(to_bigint(x0) < modulus, "array x0 not canonical");
    cpp_int total = to_bigint(x0);
    std::array<cpp_int, 4> expected_columns{};
    for (unsigned i = 0; i < 4; ++i) expected_columns[i] = x0[i];
    Scalar reference = Scalar::from_limbs(x0);
    for (const auto& value : rhs) {
        require(to_bigint(value) < modulus, "array RHS not canonical");
        total += to_bigint(value);
        for (unsigned i = 0; i < 4; ++i) expected_columns[i] += value[i];
        reference += Scalar::from_limbs(value);
        ++counts.library_array_replay_additions;
    }
    const Limbs expected = to_words<4>(total % modulus);
    check_final(reference.limbs(), expected, counts);
    check_public_be(reference, expected, counts);
    const Limbs saved_x0 = x0;
    const auto saved_rhs = rhs;
    const Limbs* const input = rhs.empty() ? nullptr : rhs.data();

    const Wide5 wide = pa_f2::accumulate_wide(x0, input, rhs.size());
    require(same_bytes(wide, to_words<5>(total)) && to_bigint(wide) == total,
            "wide accumulator lost intermediate integer or included x0 incorrectly");
    ++counts.wide_intermediate_checks;
    check_preservation(x0, saved_x0, rhs, saved_rhs, counts);
    const Columns columns = pa_f2::accumulate_columns(x0, input, rhs.size());
    for (unsigned i = 0; i < 4; ++i) {
        require(column_bigint(columns[i]) == expected_columns[i],
                "column accumulator differs from independent per-limb sum");
        ++counts.column_component_checks;
    }
    require(columns_bigint(columns) == total, "weighted columns differ from full input sum");
    check_preservation(x0, saved_x0, rhs, saved_rhs, counts);
    const Columns saved_columns = columns;
    const Wide5 normalized = pa_f2::normalize_columns(columns);
    require(same_bytes(normalized, wide) && same_bytes(normalized, to_words<5>(total)),
            "normalized columns disagree with full wide integer");
    require(same_columns(columns, saved_columns), "normalization changed columns");
    ++counts.preserved_input_values;
    ++counts.normalized_wide_checks;
    check_final(pa_f2::reduce320(wide), expected, counts);
    check_final(pa_f2::reduce320(normalized), reference.limbs(), counts);
    counts.pipeline_reduce_checks += 2;

    const Limbs wide_final = pa_f2::wide_final(x0, input, rhs.size());
    check_final(wide_final, expected, counts);
    require(same_bytes(wide_final, reference.limbs()), "wide final differs from real Scalar replay");
    ++counts.wide_final_checks;
    check_preservation(x0, saved_x0, rhs, saved_rhs, counts);
    const Limbs columns_final = pa_f2::columns_final(x0, input, rhs.size());
    check_final(columns_final, expected, counts);
    require(same_bytes(columns_final, reference.limbs()), "columns final differs from Scalar replay");
    ++counts.columns_final_checks;
    check_preservation(x0, saved_x0, rhs, saved_rhs, counts);
    digest_words(wide, 3, counts);
    digest_words(normalized, 4, counts);
    digest_words(wide_final, 5, counts);
    digest_words(columns_final, 6, counts);
    ++counts.array_cases;
    counts.array_rhs_values += rhs.size();
    if (rhs.empty()) ++counts.array_empty_cases;
}

void check_array_boundaries(Counts& counts) {
    constexpr std::array<std::size_t, 19> sizes{{0, 1, 2, 3, 4, 5, 7, 8, 9,
        15, 16, 17, 63, 64, 65, 255, 256, 257, 4096}};
    const Limbs n_minus_one = to_words<4>(modulus - 1);
    const std::array<Limbs, 6> values{{Limbs{}, Limbs{1, 0, 0, 0}, n_minus_one,
        Limbs{maximum, maximum, maximum, 0},
        Limbs{maximum, maximum, maximum - 2, maximum}, Limbs{0, 0, 0, maximum}}};
    for (const auto size : sizes) for (unsigned pattern = 0; pattern < values.size(); ++pattern) {
        std::vector<Limbs> rhs(size, values[pattern]);
        if (pattern == 1) {
            for (std::size_t i = 0; i < size; i += 2) rhs[i] = n_minus_one;
        }
        check_array(Limbs{}, rhs, counts);
        check_array(n_minus_one, rhs, counts);
        counts.array_boundary_cases += 2;
    }
}

template<class Exception, class Function>
void expect_error(Function function, Counts& counts) {
    bool caught = false;
    try {
        function();
    } catch (const Exception&) {
        caught = true;
    } catch (...) {
        throw std::runtime_error("checked F2 API threw wrong exception class");
    }
    require(caught, "checked F2 API accepted invalid input");
    ++counts.checked_error_cases;
}

void check_errors_and_contract(Counts& counts) {
    require(pa_f2::max_count == 1000000000, "unexpected F2 element cap");
    require(to_bigint(pa_modn::order) == modulus, "candidate order differs from independent n");
    require(column_bigint(pa_f2::max_column_sum) ==
            cpp_int{pa_f2::max_count + 1} * (word_base - 1), "incorrect column cap");
    const Limbs x0 = to_words<4>(modulus - 1);
    for (const auto count : std::array<std::size_t, 2>{{pa_f2::max_count + 1,
                                                      std::numeric_limits<std::size_t>::max()}}) {
        expect_error<std::length_error>([&] { (void)pa_f2::accumulate_wide(x0, nullptr, count); }, counts);
        expect_error<std::length_error>([&] { (void)pa_f2::accumulate_columns(x0, nullptr, count); }, counts);
        expect_error<std::length_error>([&] { (void)pa_f2::wide_final(x0, nullptr, count); }, counts);
        expect_error<std::length_error>([&] { (void)pa_f2::columns_final(x0, nullptr, count); }, counts);
    }
    expect_error<std::invalid_argument>([&] { (void)pa_f2::accumulate_wide(x0, nullptr, 1); }, counts);
    expect_error<std::invalid_argument>([&] { (void)pa_f2::accumulate_columns(x0, nullptr, 1); }, counts);
    expect_error<std::invalid_argument>([&] { (void)pa_f2::wide_final(x0, nullptr, 1); }, counts);
    expect_error<std::invalid_argument>([&] { (void)pa_f2::columns_final(x0, nullptr, 1); }, counts);
    for (unsigned index = 0; index < 4; ++index) {
        for (const UInt128 invalid : std::array<UInt128, 2>{{pa_f2::max_column_sum + 1, ~UInt128{0}}}) {
            Columns columns{};
            columns[index] = invalid;
            const Columns saved = columns;
            expect_error<std::invalid_argument>([&] { (void)pa_f2::normalize_columns(columns); }, counts);
            require(same_columns(columns, saved), "invalid column call mutated input");
            ++counts.preserved_input_values;
        }
    }
    require(same_bytes(pa_f2::accumulate_wide(x0, nullptr, 0), to_words<5>(to_bigint(x0))),
            "empty wide accumulator does not encode x0 once");
    ++counts.empty_identity_checks;
    const Columns empty = pa_f2::accumulate_columns(x0, nullptr, 0);
    for (unsigned i = 0; i < 4; ++i) {
        require(column_bigint(empty[i]) == x0[i], "empty columns do not encode x0 once");
    }
    ++counts.empty_identity_checks;
    require(same_bytes(pa_f2::wide_final(x0, nullptr, 0), x0), "empty wide final not identity");
    require(same_bytes(pa_f2::columns_final(x0, nullptr, 0), x0), "empty columns final not identity");
    counts.empty_identity_checks += 2;
    // Pointer extent and canonical input validity are external preconditions.
    // Oversize counts are checked before any pointer access; do not allocate
    // a billion elements or call unchecked arithmetic with noncanonical data.
}

void check_negative_controls(Counts& counts) {
    const Wide5 wide_zero{};
    const Limbs final_zero{};
    const Columns column_zero{};
    require(same_bytes(wide_zero, wide_zero) && same_bytes(final_zero, final_zero) &&
            same_columns(column_zero, column_zero), "comparator rejects equal inputs");
    for (unsigned bit = 0; bit < 320; ++bit) {
        Wide5 changed{};
        changed[bit / 64] = Word{1} << (bit % 64);
        require(!same_bytes(wide_zero, changed) && !same_bytes(changed, wide_zero),
                "wide comparator missed deliberate intermediate corruption");
        ++counts.negative_wide_bit_checks;
    }
    for (unsigned bit = 0; bit < 256; ++bit) {
        Limbs changed{};
        changed[bit / 64] = Word{1} << (bit % 64);
        require(!same_bytes(final_zero, changed) && !same_bytes(changed, final_zero),
                "final comparator missed deliberate corruption");
        ++counts.negative_final_bit_checks;
    }
    for (unsigned bit = 0; bit < 512; ++bit) {
        Columns changed{};
        changed[bit / 128] = UInt128{1} << (bit % 128);
        require(!same_columns(column_zero, changed) && !same_columns(changed, column_zero),
                "column comparator missed deliberate uint128 corruption");
        ++counts.negative_column_bit_checks;
    }
    for (const cpp_int& value : std::array<cpp_int, 4>{{width, width + 1,
                                                       wide_width - 1, 2 * modulus + 5}}) {
        const Wide5 original = to_words<5>(value);
        Wide5 lossy = original;
        lossy[4] = 0;
        const Limbs correct = to_words<4>(value % modulus);
        const Limbs wrong = pa_f2::reduce320(lossy);
        require(!same_bytes(correct, wrong), "lossy-high negative control unexpectedly agrees");
        check_final(pa_f2::reduce320(original), correct, counts);
        ++counts.lossy_high_controls;
    }
    for (const cpp_int& column : std::array<cpp_int, 2>{{word_base, column_bigint(pa_f2::max_column_sum)}}) {
        Columns exact{};
        exact[0] = to_column(column);
        const Wide5 normalized = pa_f2::normalize_columns(exact);
        Wide5 lossy = normalized;
        lossy[1] = 0;
        require(!same_bytes(normalized, lossy) && to_bigint(normalized) == column,
                "column carry-loss negative control failed");
        ++counts.overflow_loss_controls;
    }
    Limbs pattern{};
    Bytes32 expected{};
    for (unsigned byte = 0; byte < 32; ++byte) {
        expected[byte] = static_cast<std::uint8_t>(byte + 1);
        pattern[byte / 8] |= Word{byte + 1} << (8 * (byte % 8));
    }
    require(independent_le(pattern) == expected && pa_modn::encode_le(pattern) == expected,
            "explicit LE byte-order fixture failed");
}

} // namespace

int main(int argc, char**) {
    if (argc != 1) {
        std::cerr << "usage: test_f2_reduce (no arguments)\n";
        return 2;
    }
    try {
        Counts c;
        check_errors_and_contract(c);
        check_negative_controls(c);
        check_reducer_boundaries(c);
        Word random = seed;
        for (Word i = 0; i < random_reducer_cases; ++i) {
            Wide5 value{};
            for (auto& limb : value) limb = next_random(random);
            check_reducer(value, i % 128 == 0, c);
        }
        check_column_boundaries(c);
        for (Word i = 0; i < random_column_cases; ++i) {
            Columns columns{};
            for (auto& column : columns) {
                const Word high = next_random(random);
                const Word low = next_random(random);
                column = (UInt128{high} << 64) | UInt128{low};
                column %= pa_f2::max_column_sum + 1;
            }
            check_column_case(columns, c);
        }
        check_array_boundaries(c);
        for (Word i = 0; i < random_array_cases; ++i) {
            std::size_t count = static_cast<std::size_t>(next_random(random) % 514);
            if (i % 128 == 0) count = 4096;
            Limbs x0 = random_canonical(random);
            if (i % 16 == 0) x0 = {};
            if (i % 16 == 1) x0 = to_words<4>(modulus - 1);
            std::vector<Limbs> rhs(count);
            for (auto& value : rhs) value = random_canonical(random);
            check_array(x0, rhs, c);
        }
        require(c.reducer_cases == c.reducer_boundary_cases + c.deliberate_fold_boundary_cases +
                random_reducer_cases, "reducer family counts do not reconcile");
        require(c.column_cases == c.column_boundary_cases + c.synthetic_repeated_array_witnesses +
                random_column_cases, "column family counts do not reconcile");
        require(c.array_cases == c.array_boundary_cases + random_array_cases &&
                c.array_rhs_values == c.library_array_replay_additions,
                "array/replay counts do not reconcile");
        require(c.reducer_first_fold_carry_cases > 0 && c.synthetic_cap_witnesses > 0,
                "missing first-fold carry or cap witness coverage");
        std::cout << "{\n"
                  << "  \"schema\": \"pa_f2_reduce_correctness_v1\",\n"
                  << "  \"status\": \"pass\",\n"
                  << "  \"seed\": " << seed << ",\n"
                  << "  \"fixture_counts_are_unique_inputs\": false,\n"
                  << "  \"reducer_full_domain_bits\": 320,\n"
                  << "  \"reducer_cases\": " << c.reducer_cases << ",\n"
                  << "  \"reducer_random_cases\": " << random_reducer_cases << ",\n"
                  << "  \"reducer_boundary_cases\": " << c.reducer_boundary_cases << ",\n"
                  << "  \"deliberate_fold_boundary_cases\": " << c.deliberate_fold_boundary_cases << ",\n"
                  << "  \"reducer_first_fold_carry_cases\": " << c.reducer_first_fold_carry_cases << ",\n"
                  << "  \"reducer_library_subset_cases\": " << c.reducer_library_subset_cases << ",\n"
                  << "  \"library_bit_replay_additions\": " << c.library_bit_replay_additions << ",\n"
                  << "  \"column_cases\": " << c.column_cases << ",\n"
                  << "  \"normalization_domain_includes_non_array_reachable_tuples\": true,\n"
                  << "  \"column_random_cases\": " << random_column_cases << ",\n"
                  << "  \"column_boundary_cases\": " << c.column_boundary_cases << ",\n"
                  << "  \"synthetic_repeated_array_witnesses\": " << c.synthetic_repeated_array_witnesses << ",\n"
                  << "  \"synthetic_cap_witnesses\": " << c.synthetic_cap_witnesses << ",\n"
                  << "  \"cap_witnesses_are_synthetic_not_executed_arrays\": true,\n"
                  << "  \"billion_element_array_allocated\": false,\n"
                  << "  \"array_cases\": " << c.array_cases << ",\n"
                  << "  \"array_boundary_cases\": " << c.array_boundary_cases << ",\n"
                  << "  \"array_random_cases\": " << random_array_cases << ",\n"
                  << "  \"array_rhs_values\": " << c.array_rhs_values << ",\n"
                  << "  \"array_empty_cases\": " << c.array_empty_cases << ",\n"
                  << "  \"library_array_replay_additions\": " << c.library_array_replay_additions << ",\n"
                  << "  \"wide_intermediate_checks\": " << c.wide_intermediate_checks << ",\n"
                  << "  \"column_component_checks\": " << c.column_component_checks << ",\n"
                  << "  \"normalized_wide_checks\": " << c.normalized_wide_checks << ",\n"
                  << "  \"pipeline_reduce_checks\": " << c.pipeline_reduce_checks << ",\n"
                  << "  \"wide_final_checks\": " << c.wide_final_checks << ",\n"
                  << "  \"columns_final_checks\": " << c.columns_final_checks << ",\n"
                  << "  \"canonical_result_checks\": " << c.canonical_result_checks << ",\n"
                  << "  \"encoder_checks\": " << c.encoder_checks << ",\n"
                  << "  \"public_be_checks\": " << c.public_be_checks << ",\n"
                  << "  \"preserved_input_values\": " << c.preserved_input_values << ",\n"
                  << "  \"checked_error_cases\": " << c.checked_error_cases << ",\n"
                  << "  \"empty_identity_checks\": " << c.empty_identity_checks << ",\n"
                  << "  \"negative_wide_bit_checks\": " << c.negative_wide_bit_checks << ",\n"
                  << "  \"negative_final_bit_checks\": " << c.negative_final_bit_checks << ",\n"
                  << "  \"negative_column_bit_checks\": " << c.negative_column_bit_checks << ",\n"
                  << "  \"lossy_high_controls\": " << c.lossy_high_controls << ",\n"
                  << "  \"overflow_loss_controls\": " << c.overflow_loss_controls << ",\n"
                  << "  \"oracle\": \"boost::multiprecision::cpp_int_independent_n_and_D\",\n"
                  << "  \"reference\": \"linked_unchanged_Scalar_array_replay_and_320bit_replay_subset\",\n"
                  << "  \"comparison\": \"all40_intermediate_bytes_all128bits_per_column_all32_final_bytes\",\n"
                  << "  \"new_arithmetic_requires_uint128\": true,\n"
                  << "  \"no_int128_macro_scope\": \"reference_Wide5_and_reducer_second_fold_carry_helpers_only\",\n"
#ifdef SECP256K1_NO_INT128
                  << "  \"reference_carry_portable_macro\": true,\n"
#else
                  << "  \"reference_carry_portable_macro\": false,\n"
#endif
                  << "  \"result_checksum_fnv1a64\": \"" << std::hex << std::setw(16)
                  << std::setfill('0') << c.checksum << std::dec << "\",\n"
                  << "  \"mismatches\": 0,\n"
                  << "  \"negative_controls\": \"pass\",\n"
                  << "  \"timing_claim\": false,\n"
                  << "  \"constant_time_claim\": false\n}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "F2 correctness failure: " << error.what() << '\n';
        return 1;
    }
}
