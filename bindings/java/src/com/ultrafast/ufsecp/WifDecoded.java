package com.ultrafast.ufsecp;

/** Decoded WIF private key: raw 32-byte key + compression flag + network id. */
public final class WifDecoded {
    private final byte[]  privkey;
    private final boolean compressed;
    private final int     network;

    public WifDecoded(byte[] privkey, boolean compressed, int network) {
        this.privkey    = privkey;
        this.compressed = compressed;
        this.network    = network;
    }

    public byte[]  getPrivkey()    { return privkey; }
    public boolean isCompressed()  { return compressed; }
    public int     getNetwork()    { return network; }
}
