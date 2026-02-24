<?php
// ============================================================================
// UltrafastSecp256k1 — PHP Binding Smoke Test (Golden Vectors)
// ============================================================================
// Verifies FFI boundary correctness using deterministic known-answer tests.
// Runs in <2 seconds.  Requires PHP 8.1+ with ext-ffi.
//
// Usage:
//   php tests/smoke_test.php
// ============================================================================

declare(strict_types=1);

require_once __DIR__ . '/../src/Ufsecp.php';

use Ultrafast\Ufsecp\Ufsecp;

$passed = 0;
$failed = 0;

// ── Golden Vectors ──────────────────────────────────────────────────────

$KNOWN_PRIVKEY = hex2bin('0000000000000000000000000000000000000000000000000000000000000001');
$KNOWN_PUBKEY  = hex2bin('0279BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798');
$KNOWN_XONLY   = hex2bin('79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798');
$SHA256_EMPTY  = hex2bin('E3B0C44298FC1C149AFBF4C8996FB92427AE41E4649B934CA495991B7852B855');
$MSG32         = str_repeat("\x00", 32);
$AUX32         = str_repeat("\x00", 32);

// ── Helper ──────────────────────────────────────────────────────────────

function check(string $name, callable $fn): void {
    global $passed, $failed;
    try {
        $fn();
        echo "  [OK] {$name}\n";
        $passed++;
    } catch (Throwable $e) {
        echo "  [FAIL] {$name}: {$e->getMessage()}\n";
        $failed++;
    }
}

function assertEq(string $a, string $b, string $label): void {
    if ($a !== $b) {
        throw new RuntimeException("{$label}: " . bin2hex($a) . " != " . bin2hex($b));
    }
}

function assertTrue(bool $cond, string $msg): void {
    if (!$cond) {
        throw new RuntimeException("Assertion failed: {$msg}");
    }
}

// ── Tests ───────────────────────────────────────────────────────────────

echo "UltrafastSecp256k1 PHP Smoke Test\n";
echo str_repeat('=', 60) . "\n";

$ctx = new Ufsecp();

check('ctx_create_abi', function () use ($ctx) {
    $abi = $ctx->abiVersion();
    assertTrue($abi >= 1, "ABI {$abi} < 1");
});

check('pubkey_create_golden', function () use ($ctx, $KNOWN_PRIVKEY, $KNOWN_PUBKEY) {
    $pub = $ctx->pubkeyCreate($KNOWN_PRIVKEY);
    assertEq($pub, $KNOWN_PUBKEY, 'compressed pubkey');
});

check('pubkey_xonly_golden', function () use ($ctx, $KNOWN_PRIVKEY, $KNOWN_XONLY) {
    $xonly = $ctx->pubkeyXonly($KNOWN_PRIVKEY);
    assertEq($xonly, $KNOWN_XONLY, 'x-only pubkey');
});

check('ecdsa_sign_verify', function () use ($ctx, $MSG32, $KNOWN_PRIVKEY, $KNOWN_PUBKEY) {
    $sig = $ctx->ecdsaSign($MSG32, $KNOWN_PRIVKEY);
    assertTrue(strlen($sig) === 64, 'sig length');
    assertTrue($ctx->ecdsaVerify($MSG32, $sig, $KNOWN_PUBKEY), 'valid sig rejected');

    // Mutated → fail
    $bad = $sig;
    $bad[0] = chr(ord($bad[0]) ^ 0x01);
    assertTrue(!$ctx->ecdsaVerify($MSG32, $bad, $KNOWN_PUBKEY), 'mutated sig accepted');
});

check('schnorr_sign_verify', function () use ($ctx, $MSG32, $KNOWN_PRIVKEY, $KNOWN_XONLY, $AUX32) {
    $sig = $ctx->schnorrSign($MSG32, $KNOWN_PRIVKEY, $AUX32);
    assertTrue(strlen($sig) === 64, 'schnorr sig length');
    assertTrue($ctx->schnorrVerify($MSG32, $sig, $KNOWN_XONLY), 'valid schnorr sig rejected');
});

check('ecdsa_recover', function () use ($ctx, $MSG32, $KNOWN_PRIVKEY, $KNOWN_PUBKEY) {
    $rec = $ctx->ecdsaSignRecoverable($MSG32, $KNOWN_PRIVKEY);
    assertTrue($rec['recid'] >= 0 && $rec['recid'] <= 3, 'recid range');
    $pub = $ctx->ecdsaRecover($MSG32, $rec['sig'], $rec['recid']);
    assertEq($pub, $KNOWN_PUBKEY, 'recovered pubkey');
});

check('sha256_golden', function () use ($ctx, $SHA256_EMPTY) {
    $digest = Ufsecp::sha256('');
    assertEq($digest, $SHA256_EMPTY, 'SHA-256 empty');
});

check('addr_p2wpkh', function () use ($ctx, $KNOWN_PUBKEY) {
    $addr = $ctx->addrP2wpkh($KNOWN_PUBKEY, 0);
    assertTrue(str_starts_with($addr, 'bc1q'), "Expected bc1q..., got {$addr}");
});

check('wif_roundtrip', function () use ($ctx, $KNOWN_PRIVKEY) {
    $wif = $ctx->wifEncode($KNOWN_PRIVKEY, true, 0);
    $decoded = $ctx->wifDecode($wif);
    assertEq($decoded['privkey'], $KNOWN_PRIVKEY, 'WIF privkey');
    assertTrue($decoded['compressed'] === true, 'WIF compressed');
    assertTrue($decoded['network'] === 0, 'WIF mainnet');
});

check('ecdh_symmetric', function () use ($ctx, $KNOWN_PRIVKEY) {
    $k2 = hex2bin('0000000000000000000000000000000000000000000000000000000000000002');
    $pub1 = $ctx->pubkeyCreate($KNOWN_PRIVKEY);
    $pub2 = $ctx->pubkeyCreate($k2);
    $s12 = $ctx->ecdh($KNOWN_PRIVKEY, $pub2);
    $s21 = $ctx->ecdh($k2, $pub1);
    assertEq($s12, $s21, 'ECDH symmetric');
});

check('error_path', function () use ($ctx) {
    $threw = false;
    try { $ctx->pubkeyCreate(str_repeat("\x00", 32)); }
    catch (Throwable $e) { $threw = true; }
    assertTrue($threw, 'zero key should throw');
});

check('ecdsa_deterministic', function () use ($ctx, $MSG32, $KNOWN_PRIVKEY) {
    $sig1 = $ctx->ecdsaSign($MSG32, $KNOWN_PRIVKEY);
    $sig2 = $ctx->ecdsaSign($MSG32, $KNOWN_PRIVKEY);
    assertEq($sig1, $sig2, 'RFC 6979 deterministic');
});

unset($ctx);

echo str_repeat('=', 60) . "\n";
echo "  PHP smoke test: {$passed} passed, {$failed} failed\n";
echo str_repeat('=', 60) . "\n";
exit($failed > 0 ? 1 : 0);
