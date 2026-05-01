#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
workspace_root="$(cd "$repo_root/../.." && pwd)"
build_dir="${1:-$workspace_root/build_rel}"

csharp_test_project="$repo_root/bindings/csharp/tests/SmokeTest.csproj"
swift_dir="$repo_root/bindings/swift"
java_src_dir="$repo_root/bindings/java/src"
java_test_dir="$repo_root/bindings/java/tests"
java_out_dir="/tmp/ufsecp-java-smoke"
java_jni_dir="$build_dir/libs/UltrafastSecp256k1/bindings/java"
ufsecp_lib_dir="$build_dir/libs/UltrafastSecp256k1/include/ufsecp"

red='\033[0;31m'
green='\033[0;32m'
cyan='\033[0;36m'
yellow='\033[0;33m'
nc='\033[0m'

step() {
    printf "${cyan}==>${nc} %s\n" "$1"
}

require_cmd() {
    if ! command -v "$1" >/dev/null 2>&1; then
        printf "${red}missing command:${nc} %s\n" "$1" >&2
        exit 1
    fi
}

have_cmd() {
    command -v "$1" >/dev/null 2>&1
}

skip() {
    printf "${yellow}skip:${nc} %s\n" "$1"
}

if [ -f "$HOME/.local/share/swiftly/env.sh" ]; then
    # shellcheck disable=SC1091
    source "$HOME/.local/share/swiftly/env.sh"
    hash -r
fi

if [ -x "$HOME/.local/dart-sdk/bin/dart" ]; then
    export PATH="$HOME/.local/dart-sdk/bin:$PATH"
    hash -r
fi

require_cmd dotnet
require_cmd javac
require_cmd java
require_cmd cmake
require_cmd pkg-config
require_cmd swift

step "Verifying Swift pkg-config dependency"
pkg-config --exists ufsecp

if [ -d "/usr/local/lib/pkgconfig" ]; then
    export PKG_CONFIG_PATH="/usr/local/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
fi

export UFSECP_LIB_DIR="${UFSECP_LIB_DIR:-/usr/local/lib}"
export UFSECP_LIB="${UFSECP_LIB:-/usr/local/lib/libufsecp.so}"
export LD_LIBRARY_PATH="/usr/local/lib:${LD_LIBRARY_PATH:-}"

step "Building Java JNI bridge"
cmake --build "$build_dir" --target ufsecp_jni -j"$(nproc)"

step "Building C# smoke runner"
dotnet build "$csharp_test_project" -nologo

step "Running C# smoke suite"
export LD_LIBRARY_PATH="$ufsecp_lib_dir:${LD_LIBRARY_PATH:-}"
dotnet run --project "$csharp_test_project" -nologo

step "Compiling Java smoke suite"
mkdir -p "$java_out_dir"
javac -d "$java_out_dir" \
    $(find "$java_src_dir" -name '*.java') \
    $(find "$java_test_dir" -name '*.java')

step "Running Java smoke suite"
export LD_LIBRARY_PATH="$java_jni_dir:$ufsecp_lib_dir:${LD_LIBRARY_PATH:-}"
java -cp "$java_out_dir" \
    -Djava.library.path="$java_jni_dir" \
    com.ultrafast.ufsecp.tests.SmokeTest

step "Running Swift smoke suite"
(cd "$swift_dir" && swift test)

if have_cmd python3; then
    step "Running Python smoke suite"
    (cd "$repo_root/bindings/python/tests" && python3 smoke_test.py)
else
    skip "Python smoke suite (missing python3)"
fi

if have_cmd go; then
    step "Running Go smoke suite"
    export CGO_LDFLAGS="-L/usr/local/lib -L$ufsecp_lib_dir${CGO_LDFLAGS:+ $CGO_LDFLAGS}"
    (cd "$repo_root/bindings/go" && go test -run TestSmoke -v)
else
    skip "Go smoke suite (missing go)"
fi

if have_cmd cargo; then
    step "Running Rust smoke suite"
    (cd "$repo_root/bindings/rust" && cargo test -p ufsecp smoke -- --nocapture)
else
    skip "Rust smoke suite (missing cargo)"
fi

if have_cmd node; then
    step "Running Node.js smoke suite"
    (cd "$repo_root/bindings/nodejs" && node tests/smoke_test.js)
else
    skip "Node.js smoke suite (missing node)"
fi

if have_cmd php; then
    step "Running PHP smoke suite"
    (cd "$repo_root/bindings/php" && php tests/smoke_test.php)
else
    skip "PHP smoke suite (missing php)"
fi

if have_cmd ruby; then
    step "Running Ruby smoke suite"
    (cd "$repo_root/bindings/ruby" && ruby tests/smoke_test.rb)
else
    skip "Ruby smoke suite (missing ruby)"
fi

if have_cmd dart; then
    step "Running Dart smoke suite"
    (cd "$repo_root/bindings/dart" && dart pub get >/dev/null && dart run tool/smoke_runner.dart)
else
    skip "Dart smoke suite (missing dart SDK)"
fi

if have_cmd node; then
    step "Running React Native contract smoke suite"
    (cd "$repo_root/bindings/react-native" && node tests/mock_bridge_smoke.cjs)
else
    skip "React Native contract smoke suite (missing node)"
fi

if [ "${UFSECP_VALIDATE_REACT_NATIVE:-0}" = "1" ]; then
    require_cmd npm
    step "Running React Native native smoke suite"
    (cd "$repo_root/bindings/react-native" && npx jest tests/smoke_test.js)
else
    skip "React Native native smoke suite (set UFSECP_VALIDATE_REACT_NATIVE=1 and provide a RN/Jest test harness)"
fi

printf "${green}All available binding smoke suites passed.${nc}\n"