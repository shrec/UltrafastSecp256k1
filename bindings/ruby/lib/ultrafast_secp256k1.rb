# frozen_string_literal: true

require 'ffi'

# UltrafastSecp256k1 — Ruby FFI bindings.
#
# High-performance secp256k1 elliptic curve cryptography.
#
# Usage:
#   require 'ultrafast_secp256k1'
#   lib = UltrafastSecp256k1.new
#   pubkey = lib.ec_pubkey_create("\x00" * 31 + "\x01")
#
module UltrafastSecp256k1
  # FFI function signatures
  module Native
    extend FFI::Library

    LIB_NAMES = case RbConfig::CONFIG['host_os']
                when /mswin|mingw/ then %w[ultrafast_secp256k1.dll libultrafast_secp256k1.dll]
                when /darwin/      then %w[libultrafast_secp256k1.dylib]
                else                    %w[libultrafast_secp256k1.so]
                end

    def self.find_library
      # 1. Env var
      if (env = ENV['ULTRAFAST_SECP256K1_LIB']) && File.exist?(env)
        return env
      end
      # 2. Common build dirs relative to this file
      base = File.expand_path('../../..', __dir__)
      dirs = [
        File.join(base, 'c_api', 'build'),
        File.join(base, 'c_api', 'build', 'Release'),
        File.expand_path('build_rel', base),
      ]
      dirs.each do |dir|
        LIB_NAMES.each do |name|
          path = File.join(dir, name)
          return path if File.exist?(path)
        end
      end
      # 3. System
      'ultrafast_secp256k1'
    end

    ffi_lib find_library

    # BIP-32 key struct
    class Bip32Key < FFI::Struct
      layout :data, [:uint8, 78],
             :is_private, :uint8
    end

    # Version / Init
    attach_function :secp256k1_version, [], :string
    attach_function :secp256k1_init, [], :int

    # Key ops
    attach_function :secp256k1_ec_pubkey_create, [:pointer, :pointer], :int
    attach_function :secp256k1_ec_pubkey_create_uncompressed, [:pointer, :pointer], :int
    attach_function :secp256k1_ec_pubkey_parse, [:pointer, :size_t, :pointer], :int
    attach_function :secp256k1_ec_seckey_verify, [:pointer], :int
    attach_function :secp256k1_ec_privkey_negate, [:pointer], :int
    attach_function :secp256k1_ec_privkey_tweak_add, [:pointer, :pointer], :int
    attach_function :secp256k1_ec_privkey_tweak_mul, [:pointer, :pointer], :int

    # ECDSA
    attach_function :secp256k1_ecdsa_sign, [:pointer, :pointer, :pointer], :int
    attach_function :secp256k1_ecdsa_verify, [:pointer, :pointer, :pointer], :int
    attach_function :secp256k1_ecdsa_signature_serialize_der, [:pointer, :pointer, :pointer], :int

    # Recovery
    attach_function :secp256k1_ecdsa_sign_recoverable, [:pointer, :pointer, :pointer, :pointer], :int
    attach_function :secp256k1_ecdsa_recover, [:pointer, :pointer, :int, :pointer], :int

    # Schnorr
    attach_function :secp256k1_schnorr_sign, [:pointer, :pointer, :pointer, :pointer], :int
    attach_function :secp256k1_schnorr_verify, [:pointer, :pointer, :pointer], :int
    attach_function :secp256k1_schnorr_pubkey, [:pointer, :pointer], :int

    # ECDH
    attach_function :secp256k1_ecdh, [:pointer, :pointer, :pointer], :int
    attach_function :secp256k1_ecdh_xonly, [:pointer, :pointer, :pointer], :int
    attach_function :secp256k1_ecdh_raw, [:pointer, :pointer, :pointer], :int

    # Hashing
    attach_function :secp256k1_sha256, [:pointer, :size_t, :pointer], :void
    attach_function :secp256k1_hash160, [:pointer, :size_t, :pointer], :void
    attach_function :secp256k1_tagged_hash, [:string, :pointer, :size_t, :pointer], :void

    # Addresses
    attach_function :secp256k1_address_p2pkh, [:pointer, :int, :pointer, :pointer], :int
    attach_function :secp256k1_address_p2wpkh, [:pointer, :int, :pointer, :pointer], :int
    attach_function :secp256k1_address_p2tr, [:pointer, :int, :pointer, :pointer], :int

    # WIF
    attach_function :secp256k1_wif_encode, [:pointer, :int, :int, :pointer, :pointer], :int
    attach_function :secp256k1_wif_decode, [:string, :pointer, :pointer, :pointer], :int

    # BIP-32
    attach_function :secp256k1_bip32_master_key, [:pointer, :size_t, :pointer], :int
    attach_function :secp256k1_bip32_derive_child, [:pointer, :uint32, :pointer], :int
    attach_function :secp256k1_bip32_derive_path, [:pointer, :string, :pointer], :int
    attach_function :secp256k1_bip32_get_privkey, [:pointer, :pointer], :int
    attach_function :secp256k1_bip32_get_pubkey, [:pointer, :pointer], :int

    # Taproot
    attach_function :secp256k1_taproot_output_key, [:pointer, :pointer, :pointer, :pointer], :int
    attach_function :secp256k1_taproot_tweak_privkey, [:pointer, :pointer, :pointer], :int
    attach_function :secp256k1_taproot_verify_commitment, [:pointer, :int, :pointer, :pointer, :size_t], :int
  end

  NETWORK_MAINNET = 0
  NETWORK_TESTNET = 1

  class Error < StandardError; end

  # Main library wrapper
  class Secp256k1
    def initialize
      rc = Native.secp256k1_init
      raise Error, 'Library selftest failed' unless rc.zero?
    end

    def version
      Native.secp256k1_version
    end

    # ── Key Operations ─────────────────────────────────────────────────

    def ec_pubkey_create(privkey)
      check_bytes(privkey, 32, 'privkey')
      out = FFI::MemoryPointer.new(:uint8, 33)
      rc = Native.secp256k1_ec_pubkey_create(to_ptr(privkey), out)
      raise Error, 'Invalid private key' unless rc.zero?
      out.read_bytes(33)
    end

    def ec_pubkey_create_uncompressed(privkey)
      check_bytes(privkey, 32, 'privkey')
      out = FFI::MemoryPointer.new(:uint8, 65)
      rc = Native.secp256k1_ec_pubkey_create_uncompressed(to_ptr(privkey), out)
      raise Error, 'Invalid private key' unless rc.zero?
      out.read_bytes(65)
    end

    def ec_pubkey_parse(input)
      out = FFI::MemoryPointer.new(:uint8, 33)
      rc = Native.secp256k1_ec_pubkey_parse(to_ptr(input), input.bytesize, out)
      raise Error, 'Invalid public key' unless rc.zero?
      out.read_bytes(33)
    end

    def ec_seckey_verify(privkey)
      check_bytes(privkey, 32, 'privkey')
      Native.secp256k1_ec_seckey_verify(to_ptr(privkey)) == 1
    end

    def ec_privkey_negate(privkey)
      check_bytes(privkey, 32, 'privkey')
      buf = to_ptr_mut(privkey, 32)
      rc = Native.secp256k1_ec_privkey_negate(buf)
      raise Error, 'Negate failed: invalid (zero) key' unless rc.zero?
      buf.read_bytes(32)
    end

    def ec_privkey_tweak_add(privkey, tweak)
      check_bytes(privkey, 32, 'privkey')
      check_bytes(tweak, 32, 'tweak')
      buf = to_ptr_mut(privkey, 32)
      rc = Native.secp256k1_ec_privkey_tweak_add(buf, to_ptr(tweak))
      raise Error, 'Tweak add failed' unless rc.zero?
      buf.read_bytes(32)
    end

    def ec_privkey_tweak_mul(privkey, tweak)
      check_bytes(privkey, 32, 'privkey')
      check_bytes(tweak, 32, 'tweak')
      buf = to_ptr_mut(privkey, 32)
      rc = Native.secp256k1_ec_privkey_tweak_mul(buf, to_ptr(tweak))
      raise Error, 'Tweak mul failed' unless rc.zero?
      buf.read_bytes(32)
    end

    # ── ECDSA ──────────────────────────────────────────────────────────

    def ecdsa_sign(msg_hash, privkey)
      check_bytes(msg_hash, 32, 'msg_hash')
      check_bytes(privkey, 32, 'privkey')
      sig = FFI::MemoryPointer.new(:uint8, 64)
      rc = Native.secp256k1_ecdsa_sign(to_ptr(msg_hash), to_ptr(privkey), sig)
      raise Error, 'ECDSA signing failed' unless rc.zero?
      sig.read_bytes(64)
    end

    def ecdsa_verify(msg_hash, sig, pubkey)
      check_bytes(msg_hash, 32, 'msg_hash')
      check_bytes(sig, 64, 'sig')
      check_bytes(pubkey, 33, 'pubkey')
      Native.secp256k1_ecdsa_verify(to_ptr(msg_hash), to_ptr(sig), to_ptr(pubkey)) == 1
    end

    def ecdsa_serialize_der(sig)
      check_bytes(sig, 64, 'sig')
      der = FFI::MemoryPointer.new(:uint8, 72)
      der_len = FFI::MemoryPointer.new(:size_t)
      der_len.write(:size_t, 72)
      rc = Native.secp256k1_ecdsa_signature_serialize_der(to_ptr(sig), der, der_len)
      raise Error, 'DER serialization failed' unless rc.zero?
      der.read_bytes(der_len.read(:size_t))
    end

    # ── Recovery ────────────────────────────────────────────────────────

    def ecdsa_sign_recoverable(msg_hash, privkey)
      check_bytes(msg_hash, 32, 'msg_hash')
      check_bytes(privkey, 32, 'privkey')
      sig = FFI::MemoryPointer.new(:uint8, 64)
      recid = FFI::MemoryPointer.new(:int)
      rc = Native.secp256k1_ecdsa_sign_recoverable(to_ptr(msg_hash), to_ptr(privkey), sig, recid)
      raise Error, 'Recoverable signing failed' unless rc.zero?
      [sig.read_bytes(64), recid.read_int]
    end

    def ecdsa_recover(msg_hash, sig, recid)
      check_bytes(msg_hash, 32, 'msg_hash')
      check_bytes(sig, 64, 'sig')
      pubkey = FFI::MemoryPointer.new(:uint8, 33)
      rc = Native.secp256k1_ecdsa_recover(to_ptr(msg_hash), to_ptr(sig), recid, pubkey)
      raise Error, 'Recovery failed' unless rc.zero?
      pubkey.read_bytes(33)
    end

    # ── Schnorr ─────────────────────────────────────────────────────────

    def schnorr_sign(msg, privkey, aux_rand)
      check_bytes(msg, 32, 'msg')
      check_bytes(privkey, 32, 'privkey')
      check_bytes(aux_rand, 32, 'aux_rand')
      sig = FFI::MemoryPointer.new(:uint8, 64)
      rc = Native.secp256k1_schnorr_sign(to_ptr(msg), to_ptr(privkey), to_ptr(aux_rand), sig)
      raise Error, 'Schnorr signing failed' unless rc.zero?
      sig.read_bytes(64)
    end

    def schnorr_verify(msg, sig, pubkey_x)
      check_bytes(msg, 32, 'msg')
      check_bytes(sig, 64, 'sig')
      check_bytes(pubkey_x, 32, 'pubkey_x')
      Native.secp256k1_schnorr_verify(to_ptr(msg), to_ptr(sig), to_ptr(pubkey_x)) == 1
    end

    def schnorr_pubkey(privkey)
      check_bytes(privkey, 32, 'privkey')
      out = FFI::MemoryPointer.new(:uint8, 32)
      rc = Native.secp256k1_schnorr_pubkey(to_ptr(privkey), out)
      raise Error, 'Invalid private key' unless rc.zero?
      out.read_bytes(32)
    end

    # ── ECDH ────────────────────────────────────────────────────────────

    def ecdh(privkey, pubkey)
      check_bytes(privkey, 32, 'privkey')
      check_bytes(pubkey, 33, 'pubkey')
      out = FFI::MemoryPointer.new(:uint8, 32)
      rc = Native.secp256k1_ecdh(to_ptr(privkey), to_ptr(pubkey), out)
      raise Error, 'ECDH failed' unless rc.zero?
      out.read_bytes(32)
    end

    def ecdh_xonly(privkey, pubkey)
      check_bytes(privkey, 32, 'privkey')
      check_bytes(pubkey, 33, 'pubkey')
      out = FFI::MemoryPointer.new(:uint8, 32)
      rc = Native.secp256k1_ecdh_xonly(to_ptr(privkey), to_ptr(pubkey), out)
      raise Error, 'ECDH xonly failed' unless rc.zero?
      out.read_bytes(32)
    end

    def ecdh_raw(privkey, pubkey)
      check_bytes(privkey, 32, 'privkey')
      check_bytes(pubkey, 33, 'pubkey')
      out = FFI::MemoryPointer.new(:uint8, 32)
      rc = Native.secp256k1_ecdh_raw(to_ptr(privkey), to_ptr(pubkey), out)
      raise Error, 'ECDH raw failed' unless rc.zero?
      out.read_bytes(32)
    end

    # ── Hashing ─────────────────────────────────────────────────────────

    def sha256(data)
      out = FFI::MemoryPointer.new(:uint8, 32)
      Native.secp256k1_sha256(data.empty? ? nil : to_ptr(data), data.bytesize, out)
      out.read_bytes(32)
    end

    def hash160(data)
      out = FFI::MemoryPointer.new(:uint8, 20)
      Native.secp256k1_hash160(data.empty? ? nil : to_ptr(data), data.bytesize, out)
      out.read_bytes(20)
    end

    def tagged_hash(tag, data)
      out = FFI::MemoryPointer.new(:uint8, 32)
      Native.secp256k1_tagged_hash(tag, data.empty? ? nil : to_ptr(data), data.bytesize, out)
      out.read_bytes(32)
    end

    # ── Addresses ───────────────────────────────────────────────────────

    def address_p2pkh(pubkey, network: NETWORK_MAINNET)
      check_bytes(pubkey, 33, 'pubkey')
      get_address { |buf, len| Native.secp256k1_address_p2pkh(to_ptr(pubkey), network, buf, len) }
    end

    def address_p2wpkh(pubkey, network: NETWORK_MAINNET)
      check_bytes(pubkey, 33, 'pubkey')
      get_address { |buf, len| Native.secp256k1_address_p2wpkh(to_ptr(pubkey), network, buf, len) }
    end

    def address_p2tr(internal_key_x, network: NETWORK_MAINNET)
      check_bytes(internal_key_x, 32, 'internal_key_x')
      get_address { |buf, len| Native.secp256k1_address_p2tr(to_ptr(internal_key_x), network, buf, len) }
    end

    # ── WIF ─────────────────────────────────────────────────────────────

    def wif_encode(privkey, compressed: true, network: NETWORK_MAINNET)
      check_bytes(privkey, 32, 'privkey')
      get_address do |buf, len|
        Native.secp256k1_wif_encode(to_ptr(privkey), compressed ? 1 : 0, network, buf, len)
      end
    end

    def wif_decode(wif)
      pk = FFI::MemoryPointer.new(:uint8, 32)
      comp = FFI::MemoryPointer.new(:int)
      net = FFI::MemoryPointer.new(:int)
      rc = Native.secp256k1_wif_decode(wif, pk, comp, net)
      raise Error, 'Invalid WIF' unless rc.zero?
      { privkey: pk.read_bytes(32), compressed: comp.read_int == 1, network: net.read_int }
    end

    # ── BIP-32 ──────────────────────────────────────────────────────────

    def bip32_master_key(seed)
      raise Error, 'Seed must be 16-64 bytes' unless seed.bytesize.between?(16, 64)
      key = FFI::MemoryPointer.new(:uint8, 79)
      rc = Native.secp256k1_bip32_master_key(to_ptr(seed), seed.bytesize, key)
      raise Error, 'Master key failed' unless rc.zero?
      key.read_bytes(79)
    end

    def bip32_derive_child(parent, index)
      check_bytes(parent, 79, 'parent')
      child = FFI::MemoryPointer.new(:uint8, 79)
      rc = Native.secp256k1_bip32_derive_child(to_ptr(parent), index, child)
      raise Error, 'Derive child failed' unless rc.zero?
      child.read_bytes(79)
    end

    def bip32_derive_path(master, path)
      check_bytes(master, 79, 'master')
      key = FFI::MemoryPointer.new(:uint8, 79)
      rc = Native.secp256k1_bip32_derive_path(to_ptr(master), path, key)
      raise Error, "Path derivation failed: #{path}" unless rc.zero?
      key.read_bytes(79)
    end

    def bip32_get_privkey(key)
      check_bytes(key, 79, 'key')
      pk = FFI::MemoryPointer.new(:uint8, 32)
      rc = Native.secp256k1_bip32_get_privkey(to_ptr(key), pk)
      raise Error, 'Not a private key' unless rc.zero?
      pk.read_bytes(32)
    end

    def bip32_get_pubkey(key)
      check_bytes(key, 79, 'key')
      pub = FFI::MemoryPointer.new(:uint8, 33)
      rc = Native.secp256k1_bip32_get_pubkey(to_ptr(key), pub)
      raise Error, 'Pubkey extraction failed' unless rc.zero?
      pub.read_bytes(33)
    end

    # ── Taproot ─────────────────────────────────────────────────────────

    def taproot_output_key(internal_key_x, merkle_root: nil)
      check_bytes(internal_key_x, 32, 'internal_key_x')
      out = FFI::MemoryPointer.new(:uint8, 32)
      parity = FFI::MemoryPointer.new(:int)
      mr = merkle_root ? to_ptr(merkle_root) : nil
      rc = Native.secp256k1_taproot_output_key(to_ptr(internal_key_x), mr, out, parity)
      raise Error, 'Taproot output key failed' unless rc.zero?
      { output_key_x: out.read_bytes(32), parity: parity.read_int }
    end

    def taproot_tweak_privkey(privkey, merkle_root: nil)
      check_bytes(privkey, 32, 'privkey')
      out = FFI::MemoryPointer.new(:uint8, 32)
      mr = merkle_root ? to_ptr(merkle_root) : nil
      rc = Native.secp256k1_taproot_tweak_privkey(to_ptr(privkey), mr, out)
      raise Error, 'Taproot tweak failed' unless rc.zero?
      out.read_bytes(32)
    end

    def taproot_verify_commitment(output_key_x, parity, internal_key_x, merkle_root: nil)
      check_bytes(output_key_x, 32, 'output_key_x')
      check_bytes(internal_key_x, 32, 'internal_key_x')
      mr = merkle_root ? to_ptr(merkle_root) : nil
      mr_len = merkle_root ? merkle_root.bytesize : 0
      Native.secp256k1_taproot_verify_commitment(
        to_ptr(output_key_x), parity, to_ptr(internal_key_x), mr, mr_len
      ) == 1
    end

    private

    def check_bytes(data, expected, name)
      raise ArgumentError, "#{name} must be #{expected} bytes, got #{data.bytesize}" unless data.bytesize == expected
    end

    def to_ptr(data)
      ptr = FFI::MemoryPointer.new(:uint8, data.bytesize)
      ptr.put_bytes(0, data)
      ptr
    end

    def to_ptr_mut(data, size)
      ptr = FFI::MemoryPointer.new(:uint8, size)
      ptr.put_bytes(0, data)
      ptr
    end

    def get_address
      buf = FFI::MemoryPointer.new(:char, 128)
      len = FFI::MemoryPointer.new(:size_t)
      len.write(:size_t, 128)
      rc = yield(buf, len)
      raise Error, 'Address generation failed' unless rc.zero?
      buf.read_bytes(len.read(:size_t))
    end
  end
end
