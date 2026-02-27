Pod::Spec.new do |s|
  s.name         = "react-native-ufsecp"
  s.version      = "3.4.0"
  s.summary      = "React Native bindings for UltrafastSecp256k1 (ufsecp C ABI v1)"
  s.homepage     = "https://github.com/shrec/UltrafastSecp256k1"
  s.license      = "MIT"
  s.author       = "UltrafastSecp256k1"
  s.source       = { :git => "https://github.com/shrec/UltrafastSecp256k1.git", :tag => s.version }
  s.platforms    = { :ios => "15.0" }
  s.source_files = "ios/**/*.{h,m,mm}"
  s.dependency "React-Core"
  s.libraries = "ufsecp"
  s.xcconfig  = {
    "HEADER_SEARCH_PATHS" => "\"$(PODS_ROOT)/../../../include/ufsecp\"",
    "LIBRARY_SEARCH_PATHS" => "\"$(PODS_ROOT)/../../../build\""
  }
end
