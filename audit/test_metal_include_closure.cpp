#include <cctype>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

struct CMakeCall {
    std::string name;
    std::string body;
};

std::string read_file(const char* path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error(std::string("cannot open ") + path);
    }
    return {std::istreambuf_iterator<char>(input),
            std::istreambuf_iterator<char>()};
}

std::vector<CMakeCall> parse_calls(std::string_view source) {
    std::vector<CMakeCall> calls;
    for (std::size_t pos = 0; pos < source.size();) {
        if (source[pos] == '#') {
            pos = source.find('\n', pos);
            if (pos == std::string_view::npos) break;
            continue;
        }
        if (!std::isalpha(static_cast<unsigned char>(source[pos])) && source[pos] != '_') {
            ++pos;
            continue;
        }
        const auto name_begin = pos++;
        while (pos < source.size() &&
               (std::isalnum(static_cast<unsigned char>(source[pos])) || source[pos] == '_')) {
            ++pos;
        }
        std::string name(source.substr(name_begin, pos - name_begin));
        for (char& ch : name) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
        while (pos < source.size() && std::isspace(static_cast<unsigned char>(source[pos]))) ++pos;
        if (pos == source.size() || source[pos] != '(') continue;

        const auto body_begin = ++pos;
        int depth = 1;
        bool quoted = false;
        for (; pos < source.size() && depth != 0; ++pos) {
            if (source[pos] == '"' && (pos == 0 || source[pos - 1] != '\\')) quoted = !quoted;
            if (quoted) continue;
            if (source[pos] == '(') ++depth;
            if (source[pos] == ')') --depth;
        }
        if (depth != 0) throw std::runtime_error("unterminated CMake command: " + name);
        calls.push_back({name, std::string(source.substr(body_begin, pos - body_begin - 1))});
    }
    return calls;
}

std::vector<std::string> words(std::string body) {
    bool comment = false;
    bool quoted = false;
    for (char& ch : body) {
        if (ch == '\n') comment = false;
        if (!quoted && ch == '#') comment = true;
        if (!comment && ch == '"') quoted = !quoted;
        if (comment || std::isspace(static_cast<unsigned char>(ch)) || ch == '"') ch = ' ';
    }
    std::istringstream input(body);
    std::vector<std::string> result;
    for (std::string word; input >> word;) result.push_back(std::move(word));
    return result;
}

bool verify_target(const std::vector<CMakeCall>& calls, std::string_view target) {
    constexpr std::string_view required = "${CMAKE_CURRENT_SOURCE_DIR}/../cpu/include";
    int matching_blocks = 0;
    for (const auto& call : calls) {
        if (call.name != "target_include_directories") continue;
        const auto tokens = words(call.body);
        if (tokens.empty() || tokens.front() != target) continue;
        ++matching_blocks;
        if (tokens.size() < 3 || tokens[1] != "PRIVATE") {
            std::cerr << target << ": include block is not PRIVATE\n";
            return false;
        }
        bool found = false;
        for (const auto& token : tokens) {
            if (token == required) found = true;
            if (token.find("/../../cpu/include") != std::string::npos) {
                std::cerr << target << ": cpu include uses the wrong ../.. root\n";
                return false;
            }
        }
        if (!found) {
            std::cerr << target << ": missing " << required << '\n';
            return false;
        }
    }
    if (matching_blocks != 1) {
        std::cerr << target << ": expected exactly one include block, found "
                  << matching_blocks << '\n';
        return false;
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    const char* cmake_path = argc > 1 ? argv[1] : "src/metal/CMakeLists.txt";
    try {
        const auto calls = parse_calls(read_file(cmake_path));
        for (const auto& call : calls) {
            if (call.name == "include_directories") {
                std::cerr << "global include_directories workaround is forbidden\n";
                return 1;
            }
        }
        const bool test_ok = verify_target(calls, "metal_secp256k1_test");
        const bool bench_ok = verify_target(calls, "metal_secp256k1_bench_full");
        return test_ok && bench_ok ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
