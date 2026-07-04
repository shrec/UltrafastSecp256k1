# UltrafastSecp256k1 — Ruby FFI binding (ufsecp stable C ABI v1).
#
# High-performance secp256k1 ECC with dual-layer constant-time architecture.
# Context-based API.
#
# Usage:
#   require 'ufsecp'
#   ctx = Ufsecp::Context.new
#   pubkey = ctx.pubkey_create("\x00" * 31 + "\x01")
#   ctx.destroy

require 'ffi'

module Ufsecp
  extend FFI::Library

  # ── Library resolution ──────────────────────────────────────────────

  LIB_NAMES = case FFI::Platform::OS
              when /darwin/i then ['libufsecp.dylib']
              when /windows|mingw/i then ['ufsecp.dll']
              else ['libufsecp.so']
              end

  def self.find_library
    # 1. UFSECP_LIB env var
    env = ENV['UFSECP_LIB']
    return env if env && File.exist?(env)

    # 2. Next to this file
    here = File.dirname(__FILE__)
    LIB_NAMES.each do |n|
      p = File.join(here, n)
      return p if File.exist?(p)
    end

    # 3. System default
    LIB_NAMES.first
  end

  ffi_lib find_library

  # ── Error codes ─────────────────────────────────────────────────────

  OK              = 0
  ERR_NULL_ARG    = 1
  ERR_BAD_KEY     = 2
  ERR_BAD_PUBKEY  = 3
  ERR_BAD_SIG     = 4
  ERR_BAD_INPUT   = 5
  ERR_VERIFY_FAIL = 6
  ERR_ARITH       = 7
  ERR_SELFTEST    = 8
  ERR_INTERNAL    = 9
  ERR_BUF_SMALL   = 10

  ERROR_NAMES = {
    1 => 'null argument', 2 => 'invalid private key',
    3 => 'invalid public key', 4 => 'invalid signature',
    5 => 'bad input', 6 => 'verification failed',
    7 => 'arithmetic error', 8 => 'selftest failed',
    9 => 'internal error', 10 => 'buffer too small',
  }.freeze

  NET_MAINNET = 0
  NET_TESTNET = 1
  EXPECTED_ABI = 4

  class Error < StandardError
    attr_reader :code, :operation
    def initialize(op, code)
      @operation = op; @code = code
      super("ufsecp #{op} failed: #{ERROR_NAMES[code] || "unknown (#{code})"}")
    end
  end

  # ── FFI attach ──────────────────────────────────────────────────────

  # Context
  attach_function :ufsecp_ctx_create,  [:pointer], :int
  attach_function :ufsecp_ctx_destroy, [:pointer], :void
  attach_function :ufsecp_ctx_clone,   [:pointer, :pointer], :int

  # Version
  attach_function :ufsecp_version,        [], :uint
  attach_function :ufsecp_abi_version,    [], :uint
  attach_function :ufsecp_version_string, [], :string
  attach_function :ufsecp_error_str,      [:int], :string
  attach_function :ufsecp_last_error,     [:pointer], :int
  attach_function :ufsecp_last_error_msg, [:pointer], :string

  # Key ops
  attach_function :ufsecp_pubkey_create,              [:pointer, :pointer, :pointer], :int
  attach_function :ufsecp_pubkey_create_uncompressed, [:pointer, :pointer, :pointer], :int
  attach_function :ufsecp_pubkey_parse,               [:pointer, :pointer, :size_t, :pointer], :int
  attach_function :ufsecp_pubkey_xonly,               [:pointer, :pointer, :pointer], :int
  attach_function :ufsecp_seckey_verify,              [:pointer, :pointer], :int
  attach_function :ufsecp_seckey_negate,              [:pointer, :pointer], :int
  attach_function :ufsecp_seckey_tweak_add,           [:pointer, :pointer, :pointer], :int
  attach_function :ufsecp_seckey_tweak_mul,           [:pointer, :pointer, :pointer], :int

  # ECDSA
  attach_function :ufsecp_ecdsa_sign,       [:pointer, :pointer, :pointer, :pointer], :int
  attach_function :ufsecp_ecdsa_verify,     [:pointer, :pointer, :pointer, :pointer], :int
  attach_function :ufsecp_ecdsa_sig_to_der, [:pointer, :pointer, :pointer, :pointer], :int
  attach_function :ufsecp_ecdsa_sig_from_der, [:pointer, :pointer, :size_t, :pointer], :int

  # Recovery
  attach_function :ufsecp_ecdsa_sign_recoverable, [:pointer, :pointer, :pointer, :pointer, :pointer], :int
  attach_function :ufsecp_ecdsa_recover,           [:pointer, :pointer, :pointer, :int, :pointer], :int

  # Schnorr
  attach_function :ufsecp_schnorr_sign,   [:pointer, :pointer, :pointer, :pointer, :pointer], :int
  attach_function :ufsecp_schnorr_verify, [:pointer, :pointer, :pointer, :pointer], :int

  # ECDH
  attach_function :ufsecp_ecdh,       [:pointer, :pointer, :pointer, :pointer], :int
  attach_function :ufsecp_ecdh_xonly, [:pointer, :pointer, :pointer, :pointer], :int
  attach_function :ufsecp_ecdh_raw,   [:pointer, :pointer, :pointer, :pointer], :int

  # Hashing
  attach_function :ufsecp_sha256,      [:pointer, :size_t, :pointer], :int
  attach_function :ufsecp_hash160,     [:pointer, :size_t, :pointer], :int
  attach_function :ufsecp_tagged_hash, [:string, :pointer, :size_t, :pointer], :int

  # Addresses
  attach_function :ufsecp_addr_p2pkh,  [:pointer, :pointer, :int, :pointer, :pointer], :int
  attach_function :ufsecp_addr_p2wpkh, [:pointer, :pointer, :int, :pointer, :pointer], :int
  attach_function :ufsecp_addr_p2tr,   [:pointer, :pointer, :int, :pointer, :pointer], :int

  # WIF
  attach_function :ufsecp_wif_encode, [:pointer, :pointer, :int, :int, :pointer, :pointer], :int
  attach_function :ufsecp_wif_decode, [:pointer, :string, :pointer, :pointer, :pointer], :int

  # BIP-32
  attach_function :ufsecp_bip32_master,      [:pointer, :pointer, :size_t, :pointer], :int
  attach_function :ufsecp_bip32_derive,      [:pointer, :pointer, :uint, :pointer], :int
  attach_function :ufsecp_bip32_derive_path, [:pointer, :pointer, :string, :pointer], :int
  attach_function :ufsecp_bip32_privkey,     [:pointer, :pointer, :pointer], :int
  attach_function :ufsecp_bip32_pubkey,      [:pointer, :pointer, :pointer], :int

  # Taproot
  attach_function :ufsecp_taproot_output_key,   [:pointer, :pointer, :pointer, :pointer, :pointer], :int
  attach_function :ufsecp_taproot_tweak_seckey, [:pointer, :pointer, :pointer, :pointer], :int
  attach_function :ufsecp_taproot_verify,       [:pointer, :pointer, :int, :pointer, :pointer, :size_t], :int

  # ── Context class ────────────────────────────────────────────────────

  class Context
    attr_reader :ptr

    def self.finalizer(ptr)
      proc do
        Ufsecp.ufsecp_ctx_destroy(ptr) if ptr && !ptr.null?
      end
    end

    def initialize(lib_path: nil)
      abi = Ufsecp.ufsecp_abi_version
      if abi != EXPECTED_ABI
        raise Ufsecp::Error.new('init', ERR_INTERNAL), "ABI mismatch: wrapper expects ABI #{EXPECTED_ABI}, lib reports ABI #{abi}."
      end
      ctx_ptr = FFI::MemoryPointer.new(:pointer)
      rc = Ufsecp.ufsecp_ctx_create(ctx_ptr)
      raise Ufsecp::Error.new('ctx_create', rc) unless rc == OK
      @ptr = ctx_ptr.read_pointer
      @destroyed = false
      @finalizer = self.class.finalizer(@ptr)
      ObjectSpace.define_finalizer(self, @finalizer)
    end

    def destroy
      unless @destroyed
        ObjectSpace.undefine_finalizer(self)
        Ufsecp.ufsecp_ctx_destroy(@ptr) if @ptr && !@ptr.null?
        @ptr = nil; @destroyed = true
      end
    end

    alias close destroy

    # ── Version ────────────────────────────────────────────────────────

    def version;        Ufsecp.ufsecp_version; end
    def abi_version;    Ufsecp.ufsecp_abi_version; end
    def version_string; Ufsecp.ufsecp_version_string; end
    def last_error;     alive!; Ufsecp.ufsecp_last_error(@ptr); end
    def last_error_msg; alive!; Ufsecp.ufsecp_last_error_msg(@ptr); end

    # ── Key ops ────────────────────────────────────────────────────────

    def pubkey_create(privkey)
      chk!(privkey, 32, 'privkey'); alive!
      out = FFI::MemoryPointer.new(:uint8, 33)
      throw!(Ufsecp.ufsecp_pubkey_create(@ptr, buf(privkey), out), 'pubkey_create')
      out.read_bytes(33)
    end

    def pubkey_create_uncompressed(privkey)
      chk!(privkey, 32, 'privkey'); alive!
      out = FFI::MemoryPointer.new(:uint8, 65)
      throw!(Ufsecp.ufsecp_pubkey_create_uncompressed(@ptr, buf(privkey), out), 'pubkey_create_uncompressed')
      out.read_bytes(65)
    end

    def pubkey_parse(pubkey)
      alive!
      out = FFI::MemoryPointer.new(:uint8, 33)
      throw!(Ufsecp.ufsecp_pubkey_parse(@ptr, buf(pubkey), pubkey.bytesize, out), 'pubkey_parse')
      out.read_bytes(33)
    end

    def pubkey_xonly(privkey)
      chk!(privkey, 32, 'privkey'); alive!
      out = FFI::MemoryPointer.new(:uint8, 32)
      throw!(Ufsecp.ufsecp_pubkey_xonly(@ptr, buf(privkey), out), 'pubkey_xonly')
      out.read_bytes(32)
    end

    def seckey_verify(privkey)
      chk!(privkey, 32, 'privkey'); alive!
      Ufsecp.ufsecp_seckey_verify(@ptr, buf(privkey)) == OK
    end

    def seckey_negate(privkey)
      chk!(privkey, 32, 'privkey'); alive!
      b = buf_copy(privkey)
      throw!(Ufsecp.ufsecp_seckey_negate(@ptr, b), 'seckey_negate')
      b.read_bytes(32)
    end

    def seckey_tweak_add(privkey, tweak)
      chk!(privkey, 32, 'privkey'); chk!(tweak, 32, 'tweak'); alive!
      b = buf_copy(privkey)
      throw!(Ufsecp.ufsecp_seckey_tweak_add(@ptr, b, buf(tweak)), 'seckey_tweak_add')
      b.read_bytes(32)
    end

    def seckey_tweak_mul(privkey, tweak)
      chk!(privkey, 32, 'privkey'); chk!(tweak, 32, 'tweak'); alive!
      b = buf_copy(privkey)
      throw!(Ufsecp.ufsecp_seckey_tweak_mul(@ptr, b, buf(tweak)), 'seckey_tweak_mul')
      b.read_bytes(32)
    end

    # ── ECDSA ──────────────────────────────────────────────────────────

    def ecdsa_sign(msg_hash, privkey)
      chk!(msg_hash, 32, 'msg_hash'); chk!(privkey, 32, 'privkey'); alive!
      sig = FFI::MemoryPointer.new(:uint8, 64)
      throw!(Ufsecp.ufsecp_ecdsa_sign(@ptr, buf(msg_hash), buf(privkey), sig), 'ecdsa_sign')
      sig.read_bytes(64)
    end

    def ecdsa_verify(msg_hash, sig, pubkey)
      chk!(msg_hash, 32, 'msg_hash'); chk!(sig, 64, 'sig'); chk!(pubkey, 33, 'pubkey'); alive!
      Ufsecp.ufsecp_ecdsa_verify(@ptr, buf(msg_hash), buf(sig), buf(pubkey)) == OK
    end

    def ecdsa_sig_to_der(sig)
      chk!(sig, 64, 'sig'); alive!
      der = FFI::MemoryPointer.new(:uint8, 72)
      len = FFI::MemoryPointer.new(:size_t); len.write(:size_t, 72)
      throw!(Ufsecp.ufsecp_ecdsa_sig_to_der(@ptr, buf(sig), der, len), 'ecdsa_sig_to_der')
      der.read_bytes(len.read(:size_t))
    end

    def ecdsa_sig_from_der(der)
      alive!
      sig = FFI::MemoryPointer.new(:uint8, 64)
      throw!(Ufsecp.ufsecp_ecdsa_sig_from_der(@ptr, buf(der), der.bytesize, sig), 'ecdsa_sig_from_der')
      sig.read_bytes(64)
    end

    # ── Recovery ───────────────────────────────────────────────────────

    def ecdsa_sign_recoverable(msg_hash, privkey)
      chk!(msg_hash, 32, 'msg_hash'); chk!(privkey, 32, 'privkey'); alive!
      sig = FFI::MemoryPointer.new(:uint8, 64)
      recid = FFI::MemoryPointer.new(:int)
      throw!(Ufsecp.ufsecp_ecdsa_sign_recoverable(@ptr, buf(msg_hash), buf(privkey), sig, recid), 'ecdsa_sign_recoverable')
      { signature: sig.read_bytes(64), recovery_id: recid.read_int }
    end

    def ecdsa_recover(msg_hash, sig, recid)
      chk!(msg_hash, 32, 'msg_hash'); chk!(sig, 64, 'sig'); alive!
      pub = FFI::MemoryPointer.new(:uint8, 33)
      throw!(Ufsecp.ufsecp_ecdsa_recover(@ptr, buf(msg_hash), buf(sig), recid, pub), 'ecdsa_recover')
      pub.read_bytes(33)
    end

    # ── Schnorr ────────────────────────────────────────────────────────

    def schnorr_sign(msg, privkey, aux_rand)
      chk!(msg, 32, 'msg'); chk!(privkey, 32, 'privkey'); chk!(aux_rand, 32, 'aux_rand'); alive!
      sig = FFI::MemoryPointer.new(:uint8, 64)
      throw!(Ufsecp.ufsecp_schnorr_sign(@ptr, buf(msg), buf(privkey), buf(aux_rand), sig), 'schnorr_sign')
      sig.read_bytes(64)
    end

    def schnorr_verify(msg, sig, pubkey_x)
      chk!(msg, 32, 'msg'); chk!(sig, 64, 'sig'); chk!(pubkey_x, 32, 'pubkey_x'); alive!
      Ufsecp.ufsecp_schnorr_verify(@ptr, buf(msg), buf(sig), buf(pubkey_x)) == OK
    end

    # ── ECDH ──────────────────────────────────────────────────────────

    def ecdh(privkey, pubkey)
      chk!(privkey, 32, 'privkey'); chk!(pubkey, 33, 'pubkey'); alive!
      out = FFI::MemoryPointer.new(:uint8, 32)
      throw!(Ufsecp.ufsecp_ecdh(@ptr, buf(privkey), buf(pubkey), out), 'ecdh')
      out.read_bytes(32)
    end

    def ecdh_xonly(privkey, pubkey)
      chk!(privkey, 32, 'privkey'); chk!(pubkey, 33, 'pubkey'); alive!
      out = FFI::MemoryPointer.new(:uint8, 32)
      throw!(Ufsecp.ufsecp_ecdh_xonly(@ptr, buf(privkey), buf(pubkey), out), 'ecdh_xonly')
      out.read_bytes(32)
    end

    def ecdh_raw(privkey, pubkey)
      chk!(privkey, 32, 'privkey'); chk!(pubkey, 33, 'pubkey'); alive!
      out = FFI::MemoryPointer.new(:uint8, 32)
      throw!(Ufsecp.ufsecp_ecdh_raw(@ptr, buf(privkey), buf(pubkey), out), 'ecdh_raw')
      out.read_bytes(32)
    end

    # ── Hashing ────────────────────────────────────────────────────────

    def sha256(data)
      out = FFI::MemoryPointer.new(:uint8, 32)
      throw!(Ufsecp.ufsecp_sha256(buf(data), data.bytesize, out), 'sha256')
      out.read_bytes(32)
    end

    def hash160(data)
      out = FFI::MemoryPointer.new(:uint8, 20)
      throw!(Ufsecp.ufsecp_hash160(buf(data), data.bytesize, out), 'hash160')
      out.read_bytes(20)
    end

    def tagged_hash(tag, data)
      out = FFI::MemoryPointer.new(:uint8, 32)
      throw!(Ufsecp.ufsecp_tagged_hash(tag, buf(data), data.bytesize, out), 'tagged_hash')
      out.read_bytes(32)
    end

    # ── Addresses ──────────────────────────────────────────────────────

    def addr_p2pkh(pubkey, network: NET_MAINNET)
      chk!(pubkey, 33, 'pubkey'); get_addr(:ufsecp_addr_p2pkh, pubkey, network)
    end

    def addr_p2wpkh(pubkey, network: NET_MAINNET)
      chk!(pubkey, 33, 'pubkey'); get_addr(:ufsecp_addr_p2wpkh, pubkey, network)
    end

    def addr_p2tr(xonly_key, network: NET_MAINNET)
      chk!(xonly_key, 32, 'xonly_key'); get_addr(:ufsecp_addr_p2tr, xonly_key, network)
    end

    # ── WIF ────────────────────────────────────────────────────────────

    def wif_encode(privkey, compressed: true, network: NET_MAINNET)
      chk!(privkey, 32, 'privkey'); alive!
      b = FFI::MemoryPointer.new(:uint8, 128)
      len = FFI::MemoryPointer.new(:size_t); len.write(:size_t, 128)
      throw!(Ufsecp.ufsecp_wif_encode(@ptr, buf(privkey), compressed ? 1 : 0, network, b, len), 'wif_encode')
      b.read_bytes(len.read(:size_t))
    end

    def wif_decode(wif)
      alive!
      key = FFI::MemoryPointer.new(:uint8, 32)
      comp = FFI::MemoryPointer.new(:int)
      net = FFI::MemoryPointer.new(:int)
      throw!(Ufsecp.ufsecp_wif_decode(@ptr, wif, key, comp, net), 'wif_decode')
      { privkey: key.read_bytes(32), compressed: comp.read_int == 1, network: net.read_int }
    end

    # ── BIP-32 ─────────────────────────────────────────────────────────

    def bip32_master(seed)
      raise ArgumentError, "Seed must be 16-64 bytes" unless (16..64).include?(seed.bytesize)
      alive!
      key = FFI::MemoryPointer.new(:uint8, 82)
      throw!(Ufsecp.ufsecp_bip32_master(@ptr, buf(seed), seed.bytesize, key), 'bip32_master')
      key.read_bytes(82)
    end

    def bip32_derive(parent, index)
      chk!(parent, 82, 'parent'); alive!
      child = FFI::MemoryPointer.new(:uint8, 82)
      throw!(Ufsecp.ufsecp_bip32_derive(@ptr, buf(parent), index, child), 'bip32_derive')
      child.read_bytes(82)
    end

    def bip32_derive_path(master, path)
      chk!(master, 82, 'master'); alive!
      key = FFI::MemoryPointer.new(:uint8, 82)
      throw!(Ufsecp.ufsecp_bip32_derive_path(@ptr, buf(master), path, key), 'bip32_derive_path')
      key.read_bytes(82)
    end

    def bip32_privkey(key)
      chk!(key, 82, 'key'); alive!
      priv = FFI::MemoryPointer.new(:uint8, 32)
      throw!(Ufsecp.ufsecp_bip32_privkey(@ptr, buf(key), priv), 'bip32_privkey')
      priv.read_bytes(32)
    end

    def bip32_pubkey(key)
      chk!(key, 82, 'key'); alive!
      pub = FFI::MemoryPointer.new(:uint8, 33)
      throw!(Ufsecp.ufsecp_bip32_pubkey(@ptr, buf(key), pub), 'bip32_pubkey')
      pub.read_bytes(33)
    end

    # ── Taproot ────────────────────────────────────────────────────────

    def taproot_output_key(internal_key_x, merkle_root: nil)
      chk!(internal_key_x, 32, 'internal_key_x'); alive!
      out = FFI::MemoryPointer.new(:uint8, 32)
      parity = FFI::MemoryPointer.new(:int)
      mr = merkle_root ? buf(merkle_root) : nil
      throw!(Ufsecp.ufsecp_taproot_output_key(@ptr, buf(internal_key_x), mr, out, parity), 'taproot_output_key')
      { output_key_x: out.read_bytes(32), parity: parity.read_int }
    end

    def taproot_tweak_seckey(privkey, merkle_root: nil)
      chk!(privkey, 32, 'privkey'); alive!
      out = FFI::MemoryPointer.new(:uint8, 32)
      mr = merkle_root ? buf(merkle_root) : nil
      throw!(Ufsecp.ufsecp_taproot_tweak_seckey(@ptr, buf(privkey), mr, out), 'taproot_tweak_seckey')
      out.read_bytes(32)
    end

    def taproot_verify(output_key_x, parity, internal_key_x, merkle_root: nil)
      chk!(output_key_x, 32, 'output_key_x'); chk!(internal_key_x, 32, 'internal_key_x'); alive!
      mr = merkle_root ? buf(merkle_root) : nil
      mr_len = merkle_root ? merkle_root.bytesize : 0
      Ufsecp.ufsecp_taproot_verify(@ptr, buf(output_key_x), parity, buf(internal_key_x), mr, mr_len) == OK
    end

    private

    def alive!
      raise 'UfsecpContext already destroyed' if @destroyed
    end

    def throw!(rc, op)
      raise Ufsecp::Error.new(op, rc) unless rc == OK
    end

    def chk!(data, expected, name)
      raise ArgumentError, "#{name} must be #{expected} bytes, got #{data.bytesize}" unless data.bytesize == expected
    end

    def buf(str)
      p = FFI::MemoryPointer.new(:uint8, str.bytesize)
      p.put_bytes(0, str)
      p
    end

    def buf_copy(str)
      buf(str)
    end

    def get_addr(fn, key, network)
      alive!
      ab = FFI::MemoryPointer.new(:uint8, 128)
      len = FFI::MemoryPointer.new(:size_t); len.write(:size_t, 128)
      throw!(Ufsecp.send(fn, @ptr, buf(key), network, ab, len), 'address')
      ab.read_bytes(len.read(:size_t))
    end
  end
end
