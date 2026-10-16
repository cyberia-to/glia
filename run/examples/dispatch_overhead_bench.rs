//! Decompose the per-dispatch cost of one Q8 matmul at a real GatedDeltaNet
//! shape into its parts, so the optimization targets the actual cost:
//!
//!   gpu       — MTLCommandBuffer gpuStartTime→gpuEndTime (pure kernel)
//!   submit+wait — wall clock around commit→waitUntilCompleted (kernel +
//!                 Metal scheduling latency both ways)
//!   alloc     — wall clock of dev.alloc() for the output buffer alone
//!   batch_raw — wall clock of the hot-path q8_matmul::dispatch() (alloc +
//!               autorelease pool + encode + commit + wait)
//!   trait     — wall clock of HoneycrispBackend::quant_matmul() (everything
//!               above + buf_ref + Tensor wrapping)
//!
//! Every number is a per-call average over `iters` warm calls.

use run::backend::honeycrisp::device::HoneycrispDevice;
use run::backend::honeycrisp::kernels::q8_matmul;
use run::backend::honeycrisp::HoneycrispBackend;
use run::backend::Backend;
use run::core::dtype::DType;
use run::core::tensor::Tensor;
use std::time::Instant;

#[repr(C)]
#[derive(Clone, Copy)]
struct Dims { batch: u32, n_rows: u32, n_blocks: u32, pad: u32 }

fn dims_bytes(d: &Dims) -> &[u8] {
    unsafe { std::slice::from_raw_parts(d as *const Dims as *const u8, std::mem::size_of::<Dims>()) }
}

fn main() {
    let iters = 300;
    // out_proj (GDN): N=5120, K=6144 — the shape whose live cost (233-1025us)
    // was 3-15x its isolated GPU time (61us).
    let shapes: &[(&str, usize, usize)] = &[
        ("out_proj (GDN)", 5120, 6144),
        ("in_proj_qkv (GDN)", 10240, 5120),
        ("in_proj_b/a (GDN)", 48, 5120),
    ];

    let dev = HoneycrispDevice::new().expect("device");
    let backend = HoneycrispBackend::new().expect("backend");

    println!(
        "{:<20} {:>9} {:>12} {:>9} {:>11} {:>9}",
        "shape", "gpu us", "submit+wait", "alloc us", "batch_raw", "trait us"
    );

    for &(name, n_rows, k) in shapes {
        let n_blocks = k / q8_matmul::BLOCK_SIZE;
        let pipe = dev.pipeline(&q8_matmul::msl_mb(q8_matmul::TG_MAX_BLOCKS)).expect("pipe");
        let x = dev.alloc(k * 4).expect("x");
        let w = dev.alloc(n_rows * n_blocks * q8_matmul::BLOCK_BYTES).expect("w");
        let dims = Dims { batch: 1, n_rows: n_rows as u32, n_blocks: n_blocks as u32, pad: 0 };
        let rows_per_tg = q8_matmul::SIMDS_PER_GROUP as usize;
        let groups = (n_rows + rows_per_tg - 1) / rows_per_tg;

        // warm
        for _ in 0..10 {
            let _ = q8_matmul::dispatch(&dev, &pipe, &x, &w, 1, n_rows as u32, n_blocks as u32);
        }

        // 1+2: gpu time and submit+wait wall clock, same command buffers
        let mut gpu_us = 0f64;
        let mut sw_us = 0f64;
        for _ in 0..iters {
            let out = dev.alloc(n_rows * 4).expect("out");
            let cmd = dev.queue.commands().expect("cmd");
            let enc = cmd.encoder().expect("enc");
            enc.bind(&pipe);
            enc.bind_buffer(&x, 0, 0);
            enc.bind_buffer(&w, 0, 1);
            enc.bind_buffer(&out, 0, 2);
            enc.push(dims_bytes(&dims), 3);
            enc.launch_groups((groups, 1, 1), (rows_per_tg * 32, 1, 1));
            enc.finish();
            let t = Instant::now();
            cmd.submit();
            cmd.wait();
            sw_us += t.elapsed().as_secs_f64() * 1e6;
            gpu_us += cmd.gpu_time() * 1e6;
        }

        // 3: alloc alone (keep them alive until the end of the loop so the
        // allocator can't trivially recycle the same block)
        let t = Instant::now();
        let mut keep = Vec::with_capacity(iters);
        for _ in 0..iters {
            keep.push(dev.alloc(n_rows * 4).expect("out"));
        }
        let alloc_us = t.elapsed().as_secs_f64() * 1e6 / iters as f64;
        drop(keep);

        // 4: hot-path dispatch (what quant_matmul calls)
        let t = Instant::now();
        for _ in 0..iters {
            let _ = q8_matmul::dispatch(&dev, &pipe, &x, &w, 1, n_rows as u32, n_blocks as u32)
                .expect("dispatch");
        }
        let batch_raw_us = t.elapsed().as_secs_f64() * 1e6 / iters as f64;

        // 5: full trait path with GPU-resident tensors (as in the live model)
        let x_t = backend
            .upload(&vec![0u8; k * 4], vec![1, k], DType::F32)
            .expect("upload x");
        let w_t = backend
            .upload(&vec![0u8; n_rows * n_blocks * q8_matmul::BLOCK_BYTES], vec![n_rows, k], DType::Q8)
            .expect("upload w");
        for _ in 0..10 {
            let _ = backend.quant_matmul(&x_t, &w_t).expect("qm");
        }
        let t = Instant::now();
        for _ in 0..iters {
            let _: Tensor = backend.quant_matmul(&x_t, &w_t).expect("qm");
        }
        let trait_us = t.elapsed().as_secs_f64() * 1e6 / iters as f64;

        println!(
            "{:<20} {:>9.1} {:>12.1} {:>9.1} {:>11.1} {:>9.1}",
            name,
            gpu_us / iters as f64,
            sw_us / iters as f64,
            alloc_us,
            batch_raw_us,
            trait_us
        );
    }
}
