Gem::Specification.new do |s|
  s.name        = 'ufsecp'
  s.version     = '0.0.0.dev'
  s.summary     = 'Ruby FFI bindings for UltrafastSecp256k1 (ufsecp C ABI v1)'
  s.description = 'High-performance secp256k1 ECC: ECDSA, Schnorr, ECDH, recovery, BIP-32, Taproot, addresses. Context-based, dual-layer constant-time.'
  s.authors     = ['UltrafastSecp256k1']
  s.license     = 'MIT'
  s.homepage    = 'https://github.com/shrec/UltrafastSecp256k1'
  s.required_ruby_version = '>= 3.0'
  s.add_dependency 'ffi', '~> 1.15'
  s.files = Dir['lib/**/*'] + Dir['lib/native/*']
  s.metadata = {
    'source_code_uri' => 'https://github.com/shrec/UltrafastSecp256k1',
    'bug_tracker_uri' => 'https://github.com/shrec/UltrafastSecp256k1/issues'
  }
end
