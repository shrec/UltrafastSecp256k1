from conan import ConanFile
from conan.tools.cmake import CMake, CMakeToolchain, cmake_layout


class UltrafastSecp256k1Conan(ConanFile):
    name = "ultrafastsecp256k1"
    version = "3.2.0"
    license = "MIT"
    author = "shrec"
    url = "https://github.com/shrec/UltrafastSecp256k1"
    description = ("High-performance secp256k1 elliptic curve library — "
                   "ECDSA, Schnorr (BIP-340), ECDH, Taproot (BIP-341), "
                   "MuSig2 (BIP-327), BIP-32 HD derivation")
    topics = ("secp256k1", "elliptic-curve", "ecdsa", "schnorr", "ecdh",
              "taproot", "bitcoin", "cryptography")

    settings = "os", "compiler", "build_type", "arch"
    options = {
        "shared": [True, False],
        "fPIC": [True, False],
        "with_asm": [True, False],
        "with_lto": [True, False],
    }
    default_options = {
        "shared": False,
        "fPIC": True,
        "with_asm": True,
        "with_lto": False,
    }

    exports_sources = (
        "CMakeLists.txt",
        "cpu/*",
        "cuda/*",
        "include/*",
        "cmake/*",
    )

    def config_options(self):
        if self.settings.os == "Windows":
            del self.options.fPIC

    def layout(self):
        cmake_layout(self)

    def generate(self):
        tc = CMakeToolchain(self)
        tc.variables["SECP256K1_USE_ASM"] = self.options.with_asm
        tc.variables["SECP256K1_USE_LTO"] = self.options.with_lto
        tc.variables["BUILD_TESTING"] = False
        tc.variables["SECP256K1_INSTALL"] = True
        tc.generate()

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

    def package(self):
        cmake = CMake(self)
        cmake.install()

    def package_info(self):
        self.cpp_info.libs = ["fastsecp256k1"]
        self.cpp_info.set_property("cmake_file_name", "secp256k1-fast")
        self.cpp_info.set_property("cmake_target_name", "secp256k1::fast")

        if self.settings.os in ("Linux", "FreeBSD"):
            self.cpp_info.system_libs = ["pthread"]
