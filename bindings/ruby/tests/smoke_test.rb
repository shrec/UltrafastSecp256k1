# ============================================================================
# UltrafastSecp256k1 — Ruby Binding Smoke Test (Golden Vectors)
# ============================================================================
# Verifies FFI boundary correctness using deterministic known-answer tests.
# Runs in <2 seconds.  Requires `ffi` gem.
#
# Usage:
#   ruby tests/smoke_test.rb
# ============================================================================

require_relative '../lib/ufsecp'

PASSED = []
FAILED = []

# ── Golden Vectors ───────────────────────────────────────────────────────

KNOWN_PRIVKEY = ['0000000000000000000000000000000000000000000000000000000000000001'].pack('H*')
KNOWN_PUBKEY  = ['0279BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798'].pack('H*')
KNOWN_XONLY   = ['79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798'].pack('H*')
SHA256_EMPTY  = ['E3B0C44298FC1C149AFBF4C8996FB92427AE41E4649B934CA495991B7852B855'].pack('H*')
MSG32         = "\x00" * 32
AUX32         = "\x00" * 32

# ── Helper ───────────────────────────────────────────────────────────────

def check(name)
  yield
  puts "  [OK] #{name}"
  PASSED << name
rescue => e
  puts "  [FAIL] #{name}: #{e.message}"
  FAILED << name
end

def assert_eq(a, b, label)
  raise "#{label}: #{a.unpack1('H*')} != #{b.unpack1('H*')}" unless a == b
end

def assert_true(cond, msg)
  raise "Assertion failed: #{msg}" unless cond
end

# ── Tests ────────────────────────────────────────────────────────────────

puts "UltrafastSecp256k1 Ruby Smoke Test"
puts '=' * 60

ctx = Ufsecp::Context.new

check('ctx_create_abi') do
  abi = ctx.abi_version
  assert_true(abi >= 1, "ABI #{abi} < 1")
end

check('pubkey_create_golden') do
  pub = ctx.pubkey_create(KNOWN_PRIVKEY)
  assert_eq(pub, KNOWN_PUBKEY, 'compressed pubkey')
end

check('pubkey_xonly_golden') do
  xonly = ctx.pubkey_xonly(KNOWN_PRIVKEY)
  assert_eq(xonly, KNOWN_XONLY, 'x-only pubkey')
end

check('ecdsa_sign_verify') do
  sig = ctx.ecdsa_sign(MSG32, KNOWN_PRIVKEY)
  assert_true(sig.bytesize == 64, 'sig length')
  assert_true(ctx.ecdsa_verify(MSG32, sig, KNOWN_PUBKEY), 'valid sig rejected')

  # Mutated → fail
  bad = sig.dup
  bad.setbyte(0, bad.getbyte(0) ^ 0x01)
  assert_true(!ctx.ecdsa_verify(MSG32, bad, KNOWN_PUBKEY), 'mutated sig accepted')
end

check('schnorr_sign_verify') do
  sig = ctx.schnorr_sign(MSG32, KNOWN_PRIVKEY, AUX32)
  assert_true(sig.bytesize == 64, 'schnorr sig length')
  assert_true(ctx.schnorr_verify(MSG32, sig, KNOWN_XONLY), 'valid schnorr sig rejected')
end

check('ecdsa_recover') do
  rec = ctx.ecdsa_sign_recoverable(MSG32, KNOWN_PRIVKEY)
  assert_true(rec[:recovery_id] >= 0 && rec[:recovery_id] <= 3, 'recid range')
  pub = ctx.ecdsa_recover(MSG32, rec[:signature], rec[:recovery_id])
  assert_eq(pub, KNOWN_PUBKEY, 'recovered pubkey')
end

check('sha256_golden') do
  digest = ctx.sha256('')
  assert_eq(digest, SHA256_EMPTY, 'SHA-256 empty')
end

check('addr_p2wpkh') do
  addr = ctx.addr_p2wpkh(KNOWN_PUBKEY, 0)
  assert_true(addr.start_with?('bc1q'), "Expected bc1q..., got #{addr}")
end

check('wif_roundtrip') do
  wif = ctx.wif_encode(KNOWN_PRIVKEY, true, 0)
  decoded = ctx.wif_decode(wif)
  assert_eq(decoded[:privkey], KNOWN_PRIVKEY, 'WIF privkey')
  assert_true(decoded[:compressed], 'WIF compressed')
  assert_true(decoded[:network] == 0, 'WIF mainnet')
end

check('ecdh_symmetric') do
  k2 = ['0000000000000000000000000000000000000000000000000000000000000002'].pack('H*')
  pub1 = ctx.pubkey_create(KNOWN_PRIVKEY)
  pub2 = ctx.pubkey_create(k2)
  s12 = ctx.ecdh(KNOWN_PRIVKEY, pub2)
  s21 = ctx.ecdh(k2, pub1)
  assert_eq(s12, s21, 'ECDH symmetric')
end

check('error_path') do
  threw = false
  begin
    ctx.pubkey_create("\x00" * 32)
  rescue => e
    threw = true
  end
  assert_true(threw, 'zero key should throw')
end

check('ecdsa_deterministic') do
  sig1 = ctx.ecdsa_sign(MSG32, KNOWN_PRIVKEY)
  sig2 = ctx.ecdsa_sign(MSG32, KNOWN_PRIVKEY)
  assert_eq(sig1, sig2, 'RFC 6979 deterministic')
end

ctx.destroy

puts '=' * 60
puts "  Ruby smoke test: #{PASSED.size} passed, #{FAILED.size} failed"
puts '=' * 60
exit(FAILED.any? ? 1 : 0)
