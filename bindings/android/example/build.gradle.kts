// ============================================================================
// UltrafastSecp256k1 — Android Example App (build.gradle.kts)
// ============================================================================
// This shows how to integrate the native library into an Android project
// using CMake + Gradle.
// ============================================================================

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.secp256k1.example"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.secp256k1.example"
        minSdk = 24
        targetSdk = 34
        versionCode = 1
        versionName = "1.0"

        // Native build configuration
        externalNativeBuild {
            cmake {
                // Build Release for performance
                arguments += "-DCMAKE_BUILD_TYPE=Release"
                // Target ABIs (arm64 primary, x86_64 for emulator)
                abiFilters += listOf("arm64-v8a", "x86_64")
                // Use c++_static STL
                arguments += "-DANDROID_STL=c++_static"
            }
        }
    }

    externalNativeBuild {
        cmake {
            // Point to the android CMakeLists.txt
            // Adjust this path relative to your app module
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.12.0")
    implementation("androidx.appcompat:appcompat:1.6.1")
}
