/**
 * UltrafastSecp256k1 — React Native binding (ufsecp stable C ABI v1).
 *
 * Bridges to native via NativeModules.Ufsecp (Android: Java JNI, iOS: ObjC bridge).
 *
 * Usage:
 *   import { Ufsecp } from 'react-native-ufsecp';
 *   const ctx = await Ufsecp.create();
 *   const pubkey = await ctx.pubkeyCreate(privkeyHex);
 *   ctx.destroy();
 */

import { NativeModules, Platform } from 'react-native';

const { Ufsecp: NativeUfsecp } = NativeModules;

if (!NativeUfsecp) {
  throw new Error(
    'react-native-ufsecp: NativeModule not linked. Run `npx react-native link react-native-ufsecp` or rebuild.',
  );
}

const hex2buf = (hex) => {
  const bytes = [];
  for (let i = 0; i < hex.length; i += 2) bytes.push(parseInt(hex.substr(i, 2), 16));
  return bytes;
};

const buf2hex = (arr) =>
  (Array.isArray(arr) ? arr : [...arr]).map((b) => ('0' + (b & 0xff).toString(16)).slice(-2)).join('');

class UfsecpError extends Error {
  constructor(op, msg) {
    super(`ufsecp ${op}: ${msg}`);
    this.name = 'UfsecpError';
    this.operation = op;
  }
}

class UfsecpContext {
  constructor(handle) {
    this._h = handle;
  }

  static async create() {
    const h = await NativeUfsecp.create();
    return new UfsecpContext(h);
  }

  async destroy() {
    if (this._h != null) {
      await NativeUfsecp.destroy(this._h);
      this._h = null;
    }
  }

  // ── Version ──────────────────────────────────────────────────────

  static version()       { return NativeUfsecp.version(); }
  static abiVersion()    { return NativeUfsecp.abiVersion(); }
  static versionString() { return NativeUfsecp.versionString(); }

  // ── Key ops ──────────────────────────────────────────────────────

  /** @param {string} privkeyHex - 32-byte hex */
  async pubkeyCreate(privkeyHex) {
    this._alive();
    return NativeUfsecp.pubkeyCreate(this._h, privkeyHex);
  }

  async pubkeyCreateUncompressed(privkeyHex) {
    this._alive();
    return NativeUfsecp.pubkeyCreateUncompressed(this._h, privkeyHex);
  }

  async seckeyVerify(privkeyHex) {
    this._alive();
    return NativeUfsecp.seckeyVerify(this._h, privkeyHex);
  }

  async seckeyNegate(privkeyHex) {
    this._alive();
    return NativeUfsecp.seckeyNegate(this._h, privkeyHex);
  }

  async seckeyTweakAdd(privkeyHex, tweakHex) {
    this._alive();
    return NativeUfsecp.seckeyTweakAdd(this._h, privkeyHex, tweakHex);
  }

  async seckeyTweakMul(privkeyHex, tweakHex) {
    this._alive();
    return NativeUfsecp.seckeyTweakMul(this._h, privkeyHex, tweakHex);
  }

  async pubkeyParse(pubkeyHex) {
    this._alive();
    return NativeUfsecp.pubkeyParse(this._h, pubkeyHex);
  }

  async pubkeyXonly(privkeyHex) {
    this._alive();
    return NativeUfsecp.pubkeyXonly(this._h, privkeyHex);
  }

  // ── ECDSA ────────────────────────────────────────────────────────

  async ecdsaSign(msgHashHex, privkeyHex) {
    this._alive();
    return NativeUfsecp.ecdsaSign(this._h, msgHashHex, privkeyHex);
  }

  async ecdsaVerify(msgHashHex, sigHex, pubkeyHex) {
    this._alive();
    return NativeUfsecp.ecdsaVerify(this._h, msgHashHex, sigHex, pubkeyHex);
  }

  async ecdsaSignRecoverable(msgHashHex, privkeyHex) {
    this._alive();
    return NativeUfsecp.ecdsaSignRecoverable(this._h, msgHashHex, privkeyHex);
  }

  async ecdsaRecover(msgHashHex, sigHex, recid) {
    this._alive();
    return NativeUfsecp.ecdsaRecover(this._h, msgHashHex, sigHex, recid);
  }

  async ecdsaSigToDer(sigHex) {
    this._alive();
    return NativeUfsecp.ecdsaSigToDer(this._h, sigHex);
  }

  async ecdsaSigFromDer(derHex) {
    this._alive();
    return NativeUfsecp.ecdsaSigFromDer(this._h, derHex);
  }

  // ── Schnorr ──────────────────────────────────────────────────────

  async schnorrSign(msgHex, privkeyHex, auxRandHex) {
    this._alive();
    return NativeUfsecp.schnorrSign(this._h, msgHex, privkeyHex, auxRandHex);
  }

  async schnorrVerify(msgHex, sigHex, pubkeyXHex) {
    this._alive();
    return NativeUfsecp.schnorrVerify(this._h, msgHex, sigHex, pubkeyXHex);
  }

  // ── ECDH ─────────────────────────────────────────────────────────

  async ecdh(privkeyHex, pubkeyHex) {
    this._alive();
    return NativeUfsecp.ecdh(this._h, privkeyHex, pubkeyHex);
  }

  async ecdhXonly(privkeyHex, pubkeyHex) {
    this._alive();
    return NativeUfsecp.ecdhXonly(this._h, privkeyHex, pubkeyHex);
  }

  async ecdhRaw(privkeyHex, pubkeyHex) {
    this._alive();
    return NativeUfsecp.ecdhRaw(this._h, privkeyHex, pubkeyHex);
  }

  // ── Hashing ──────────────────────────────────────────────────────

  static sha256(dataHex)  { return NativeUfsecp.sha256(dataHex); }
  static hash160(dataHex) { return NativeUfsecp.hash160(dataHex); }
  static taggedHash(tag, dataHex) { return NativeUfsecp.taggedHash(tag, dataHex); }

  // ── Addresses ────────────────────────────────────────────────────

  async addrP2pkh(pubkeyHex, network = 0)  { this._alive(); return NativeUfsecp.addrP2pkh(this._h, pubkeyHex, network); }
  async addrP2wpkh(pubkeyHex, network = 0) { this._alive(); return NativeUfsecp.addrP2wpkh(this._h, pubkeyHex, network); }
  async addrP2tr(xonlyHex, network = 0)    { this._alive(); return NativeUfsecp.addrP2tr(this._h, xonlyHex, network); }

  // ── WIF ──────────────────────────────────────────────────────────

  async wifEncode(privkeyHex, compressed = true, network = 0) {
    this._alive();
    return NativeUfsecp.wifEncode(this._h, privkeyHex, compressed, network);
  }

  async wifDecode(wifStr) {
    this._alive();
    return NativeUfsecp.wifDecode(this._h, wifStr);
  }

  // ── BIP-32 ───────────────────────────────────────────────────────

  async bip32Master(seedHex) {
    this._alive();
    return NativeUfsecp.bip32Master(this._h, seedHex);
  }

  async bip32Derive(parentHex, index) {
    this._alive();
    return NativeUfsecp.bip32Derive(this._h, parentHex, index);
  }

  async bip32DerivePath(masterHex, path) {
    this._alive();
    return NativeUfsecp.bip32DerivePath(this._h, masterHex, path);
  }

  async bip32Privkey(keyHex) {
    this._alive();
    return NativeUfsecp.bip32Privkey(this._h, keyHex);
  }

  async bip32Pubkey(keyHex) {
    this._alive();
    return NativeUfsecp.bip32Pubkey(this._h, keyHex);
  }

  // ── Taproot ──────────────────────────────────────────────────────

  async taprootOutputKey(internalXHex, merkleRootHex = null) {
    this._alive();
    return NativeUfsecp.taprootOutputKey(this._h, internalXHex, merkleRootHex);
  }

  async taprootTweakSeckey(privkeyHex, merkleRootHex = null) {
    this._alive();
    return NativeUfsecp.taprootTweakSeckey(this._h, privkeyHex, merkleRootHex);
  }

  async taprootVerify(outputXHex, parity, internalXHex, merkleRootHex = null) {
    this._alive();
    return NativeUfsecp.taprootVerify(this._h, outputXHex, parity, internalXHex, merkleRootHex);
  }

  // ── Internal ─────────────────────────────────────────────────────

  _alive() {
    if (this._h == null) throw new UfsecpError('context', 'already destroyed');
  }
}

export { UfsecpContext, UfsecpError };
export default UfsecpContext;
