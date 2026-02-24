package com.ultrafast.ufsecp;

/** Taproot output key: 32-byte x-only public key + parity flag. */
public final class TaprootOutputKeyResult {
    private final byte[] outputKey;
    private final int    parity;

    public TaprootOutputKeyResult(byte[] outputKey, int parity) {
        this.outputKey = outputKey;
        this.parity    = parity;
    }

    public byte[] getOutputKey() { return outputKey; }
    public int    getParity()    { return parity; }
}
