use num_bigint::BigUint;
use ufsecp_core::{AffinePoint, FieldElement as F, JacobianPoint as J, Scalar as S};

fn big(a: [u64; 4]) -> BigUint {
    BigUint::from_bytes_le(&a.into_iter().flat_map(u64::to_le_bytes).collect::<Vec<_>>())
}
fn limbs(a: BigUint) -> [u64; 4] {
    let words = a.to_u64_digits();
    core::array::from_fn(|i| words.get(i).copied().unwrap_or(0))
}
fn vectors(modulus: [u64; 4]) -> Vec<[u64; 4]> {
    let mut out = vec![
        [0; 4],
        [1, 0, 0, 0],
        modulus,
        limbs(big(modulus) - 1u32),
        [u64::MAX; 4],
    ];
    for bit in 1..256 {
        out.push(limbs(BigUint::from(1u32) << bit));
        out.push(limbs((BigUint::from(1u32) << bit) - 1u32));
    }
    let mut state = 0x91bb5e41d779de99u64;
    for _ in 0..1024 {
        out.push(core::array::from_fn(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        }));
    }
    out
}

#[test]
fn field_against_bigint() {
    let p = big(F::MODULUS);
    let inputs = vectors(F::MODULUS);
    for (i, a) in inputs.iter().copied().enumerate() {
        let b = inputs[(i * 31 + 7) % inputs.len()];
        let aa = big(a) % &p;
        let bb = big(b) % &p;
        let x = F::from_limbs_reduced(a);
        let y = F::from_limbs_reduced(b);
        assert_eq!(F::from_limbs(a).is_some(), big(a) < p);
        assert_eq!(x.to_limbs(), limbs(aa.clone()));
        assert_eq!(x.add_mod(y).to_limbs(), limbs((&aa + &bb) % &p));
        assert_eq!(x.sub_mod(y).to_limbs(), limbs((&aa + &p - &bb) % &p));
        assert_eq!(x.negate().to_limbs(), limbs((&p - &aa) % &p));
        assert_eq!(x.mul_mod(y).to_limbs(), limbs((&aa * &bb) % &p));
        assert_eq!(x.square().to_limbs(), limbs((&aa * &aa) % &p));
        assert_eq!(x.inverse().to_limbs(), limbs(aa.modpow(&(&p - 2u32), &p)));
        let square =
            aa == BigUint::from(0u32) || aa.modpow(&((&p - 1u32) >> 1), &p) == BigUint::from(1u32);
        assert_eq!(x.is_square(), square);
        if let Some(root) = x.sqrt() {
            assert_eq!(root.square(), x);
        }
        assert_eq!(F::from_be_bytes(x.to_be_bytes()), Some(x));
    }
    eprintln!("{} field boundary/random vectors checked", inputs.len());
}

#[test]
fn scalar_and_point_against_independent_oracles() {
    let n = big(S::ORDER);
    let inputs = vectors(S::ORDER);
    for (i, a) in inputs.iter().copied().enumerate() {
        let b = inputs[(i * 13 + 1) % inputs.len()];
        let aa = big(a) % &n;
        let bb = big(b) % &n;
        let x = S::from_limbs_reduced(a);
        let y = S::from_limbs_reduced(b);
        assert_eq!(S::from_limbs(a).is_some(), big(a) < n);
        assert_eq!(x.to_limbs(), limbs(aa.clone()));
        assert_eq!(x.add_mod(y).to_limbs(), limbs((&aa + &bb) % &n));
        assert_eq!(x.sub_mod(y).to_limbs(), limbs((&aa + &n - &bb) % &n));
        assert_eq!(x.negate().to_limbs(), limbs((&n - &aa) % &n));
        assert_eq!(x.mul_mod(y).to_limbs(), limbs((&aa * &bb) % &n));
        if i < 20 {
            assert_eq!(x.inverse().to_limbs(), limbs(aa.modpow(&(&n - 2u32), &n)));
        }
        assert_eq!(S::from_be_bytes(x.to_be_bytes()), x);
        if i < 5 || i % 5 == 0 {
            let point = J::generator_mul(x).to_affine();
            match secp256k1::SecretKey::from_secret_bytes(x.to_be_bytes()) {
                Ok(key) => {
                    let expected =
                        secp256k1::PublicKey::from_secret_key(&key).serialize_uncompressed();
                    let point = point.unwrap();
                    assert_eq!(point.x.to_be_bytes(), expected[1..33]);
                    assert_eq!(point.y.to_be_bytes(), expected[33..]);
                    assert_eq!(AffinePoint::new(point.x, point.y), Some(point));
                }
                Err(_) => assert!(point.is_none()),
            }
        }
    }
    let g = AffinePoint::GENERATOR;
    let j = J::from_affine(g);
    assert_eq!(j.add_mixed(g).to_affine(), j.double().to_affine());
    assert!(j
        .add_mixed(AffinePoint {
            x: g.x,
            y: g.y.negate()
        })
        .to_affine()
        .is_none());
    assert_eq!(J::INFINITY.add_mixed(g).to_affine(), Some(g));
    assert!(AffinePoint::new(F::ZERO, F::ZERO).is_none());
    eprintln!(
        "{} scalar vectors and sampled full-width public points checked",
        inputs.len()
    );
}
