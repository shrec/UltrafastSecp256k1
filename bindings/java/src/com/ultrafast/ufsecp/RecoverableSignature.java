package com.ultrafast.ufsecp;

/** ECDSA recoverable signature: 64-byte compact sig + recovery id (0-3). */
public final class RecoverableSignature {
    private final byte[] signature;
    private final int    recid;

    public RecoverableSignature(byte[] signature, int recid) {
        this.signature = signature;
        this.recid     = recid;
    }

    public byte[] getSignature() { return signature; }
    public int    getRecid()     { return recid; }
}
