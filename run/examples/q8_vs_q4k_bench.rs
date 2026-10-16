//! Real GPU-only timing (MTLCommandBuffer gpuStartTime/gpuEndTime) for the
//! Q8 vs Q4K "LARGE" matmul kernels at shapes matching this repo's actual
//! GatedDeltaNet + FFN projections on the 27B qwen3_5 model. No Instruments
//! needed — Metal's own command-buffer timestamps are exact and don't
//! include CPU-side submit/encode overhead, isolating pure kernel compute
//! + memory time. Answers: is Q4_K's LARGE kernel actually more expensive
//! per output element than Q8's, or is something else going on?

use run::backend::honeycrisp::device::HoneycrispDevice;
use run::backend::honeycrisp::kernels::{q4k_matmul, q8_matmul};

fn bench_q8(dev: &HoneycrispDevice, n_rows: usize, k: usize, iters: usize) -> f64 {
    let pipe = dev.pipeline(&q8_matmul::msl_mb(q8_matmul::TG_MAX_BLOCKS)).expect("compile q8");
    let n_blocks = k / q8_matmul::BLOCK_SIZE;
    let x = dev.alloc(k * 4).expect("alloc x");
    let w = dev.alloc(n_rows * n_blocks * q8_matmul::BLOCK_BYTES).expect("alloc w");

    // Warmup
    for _ in 0..5 {
        let _ = q8_matmul::dispatch(dev, &pipe, &x, &w, 1, n_rows as u32, n_blocks as u32);
    }

    let mut total_us = 0f64;
    for _ in 0..iters {
        let cmd = dev.queue.commands().expect("commands");
        let enc = cmd.encoder().expect("encoder");
        enc.bind(&pipe);
        enc.bind_buffer(&x, 0, 0);
        enc.bind_buffer(&w, 0, 1);
        let out = dev.alloc((n_rows * 4) as usize).expect("alloc out");
        enc.bind_buffer(&out, 0, 2);
        #[repr(C)]
        struct Dims { batch: u32, n_rows: u32, n_blocks: u32, pad: u32 }
        let dims = Dims { batch: 1, n_rows: n_rows as u32, n_blocks: n_blocks as u32, pad: 0 };
        let bytes = unsafe {
            std::slice::from_raw_parts(&dims as *const Dims as *const u8, std::mem::size_of::<Dims>())
        };
        enc.push(bytes, 3);
        let rows_per_tg = q8_matmul::SIMDS_PER_GROUP as usize;
        let groups = (n_rows + rows_per_tg - 1) / rows_per_tg;
        enc.launch_groups((groups, 1, 1), (rows_per_tg * 32, 1, 1));
        enc.finish();
        cmd.submit();
        cmd.wait();
        total_us += cmd.gpu_time() * 1e6;
    }
    total_us / iters as f64
}

fn bench_q4k(dev: &HoneycrispDevice, n_rows: usize, k: usize, iters: usize) -> f64 {
    let pipe = dev.pipeline(q4k_matmul::MSL_LARGE).expect("compile q4k");
    let n_blocks = k / 256;
    let x = dev.alloc(k * 4).expect("alloc x");
    let w = dev.alloc(n_rows * n_blocks * 144).expect("alloc w");

    for _ in 0..5 {
        let _ = q4k_matmul::dispatch_large(dev, &pipe, &x, &w, 1, n_rows as u32, n_blocks as u32);
    }

    let mut total_us = 0f64;
    for _ in 0..iters {
        let cmd = dev.queue.commands().expect("commands");
        let enc = cmd.encoder().expect("encoder");
        enc.bind(&pipe);
        enc.bind_buffer(&x, 0, 0);
        enc.bind_buffer(&w, 0, 1);
        let out = dev.alloc((n_rows * 4) as usize).expect("alloc out");
        enc.bind_buffer(&out, 0, 2);
        #[repr(C)]
        struct Dims { batch: u32, n_rows: u32, n_blocks: u32, pad: u32 }
        let dims = Dims { batch: 1, n_rows: n_rows as u32, n_blocks: n_blocks as u32, pad: 0 };
        let bytes = unsafe {
            std::slice::from_raw_parts(&dims as *const Dims as *const u8, std::mem::size_of::<Dims>())
        };
        enc.push(bytes, 3);
        let rows_per_tg = q4k_matmul::SIMDS_PER_GROUP as usize;
        let groups = (n_rows + rows_per_tg - 1) / rows_per_tg;
        enc.launch_groups((groups, 1, 1), (rows_per_tg * 32, 1, 1));
        enc.finish();
        cmd.submit();
        cmd.wait();
        total_us += cmd.gpu_time() * 1e6;
    }
    total_us / iters as f64
}

fn main() {
    let dev = HoneycrispDevice::new().expect("device");
    let iters = 200;

    // Shapes from the real 27B qwen3_5 model (hidden=5120):
    // in_proj_qkv: N=10240 K=5120 (linear_attn) ; FFN gate/up: N~14336 K=5120
    let shapes: &[(&str, usize, usize)] = &[
        ("in_proj_qkv (GDN)", 10240, 5120),
        ("out_proj (GDN)",     5120, 6144),
        ("ffn gate/up",       14336, 5120),
        ("ffn down",           5120, 14336),
    ];

    println!("{:<20} {:>10} {:>10} {:>12} {:>12} {:>10}", "shape", "N", "K", "Q8 (us)", "Q4K (us)", "Q4K/Q8");
    for &(name, n, k) in shapes {
        if k % 256 != 0 {
            println!("{name}: K={k} not 256-aligned, skipping Q4K");
            continue;
        }
        let q8_us = bench_q8(&dev, n, k, iters);
        let q4k_us = bench_q4k(&dev, n, k, iters);
        println!("{:<20} {:>10} {:>10} {:>12.2} {:>12.2} {:>10.2}x", name, n, k, q8_us, q4k_us, q4k_us / q8_us);
    }
}
