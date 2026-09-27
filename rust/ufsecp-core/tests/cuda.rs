//! Run only with exclusive GPU access and a freshly built cuda_probe.ptx.
use cudarc::{
    driver::{CudaContext, LaunchConfig, PushKernelArg},
    nvrtc::Ptx,
};
use ufsecp_core::{FieldElement as F, JacobianPoint as J, Scalar as S};

#[test]
#[ignore = "requires UFSECP_TEST_PTX and exclusive access to a CUDA GPU"]
fn cuda_matches_host_boundary_arithmetic() {
    let mut inputs = vec![[0; 4], [1, 0, 0, 0], F::MODULUS, S::ORDER, [u64::MAX; 4]];
    for modulus in [F::MODULUS, S::ORDER] {
        let mut below = modulus;
        below[0] -= 1;
        inputs.push(below);
    }
    for bit in 1..256 {
        let mut power = [0; 4];
        power[bit / 64] = 1 << (bit % 64);
        inputs.push(power);
        let mut minus = power;
        for limb in &mut minus {
            let (value, borrow) = limb.overflowing_sub(1);
            *limb = value;
            if !borrow {
                break;
            }
        }
        inputs.push(minus);
    }
    let mut seed = 0x18238a4838b447a1u64;
    for _ in 0..256 {
        inputs.push(core::array::from_fn(|_| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        }));
    }
    let mut packed = Vec::new();
    let mut expected = Vec::new();
    for (i, a) in inputs.iter().copied().enumerate() {
        let b = inputs[(i * 11 + 3) % inputs.len()];
        packed.extend(a);
        packed.extend(b);
        let x = F::from_limbs_reduced(a);
        let y = F::from_limbs_reduced(b);
        let s = S::from_limbs_reduced(a);
        let t = S::from_limbs_reduced(b);
        let point = J::generator_mul(s).to_affine();
        for value in [
            x.add_mod(y).to_limbs(),
            x.sub_mod(y).to_limbs(),
            x.mul_mod(y).to_limbs(),
            x.square().to_limbs(),
            x.inverse().to_limbs(),
            s.add_mod(t).to_limbs(),
            s.sub_mod(t).to_limbs(),
            s.mul_mod(t).to_limbs(),
            s.inverse().to_limbs(),
            point.map_or([0; 4], |p| p.x.to_limbs()),
            point.map_or([0; 4], |p| p.y.to_limbs()),
        ] {
            expected.extend(value);
        }
    }
    let count = inputs.len() as u32;
    let context = CudaContext::new(0).unwrap();
    let stream = context.default_stream();
    let source =
        std::fs::read_to_string(std::env::var("UFSECP_TEST_PTX").expect("set UFSECP_TEST_PTX"))
            .unwrap();
    let module = context.load_module(Ptx::from_src(source)).unwrap();
    let kernel = module.load_function("ufsecp_arithmetic_probe").unwrap();
    let input = stream.clone_htod(&packed).unwrap();
    let mut output = stream.alloc_zeros::<u64>(expected.len()).unwrap();
    unsafe {
        stream
            .launch_builder(&kernel)
            .arg(&input)
            .arg(&mut output)
            .arg(&count)
            // This diagnostic kernel retains all operation results; a small
            // block accommodates its register use on supported architectures.
            .launch(LaunchConfig {
                grid_dim: (count.div_ceil(64), 1, 1),
                block_dim: (64, 1, 1),
                shared_mem_bytes: 0,
            })
            .unwrap();
    }
    let actual = stream.clone_dtoh(&output).unwrap();
    for (i, (a, b)) in actual.iter().zip(&expected).enumerate() {
        assert_eq!(
            a,
            b,
            "vector {} operation {} limb {}",
            i / 44,
            (i % 44) / 4,
            i % 4
        );
    }
    eprintln!("{count} CUDA vectors: field/scalar arithmetic and full-width generator multiplication matched the host oracle");
}
