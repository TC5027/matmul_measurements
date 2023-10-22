#![feature(portable_simd)]
use clap::{Parser, Subcommand};
use partial_callgrind::{start, stop};
use std::ops::{Index, IndexMut};
use std::simd::SimdUint;
use std::time::{Duration, Instant};

struct Matrix<T> {
    pub width: usize,
    pub height: usize,
    data: Vec<T>,
}

impl<T> Matrix<T> {
    fn new(width: usize, height: usize, data: Vec<T>) -> Self {
        Matrix {
            width,
            height,
            data,
        }
    }
}

impl<T> Index<(usize, usize)> for Matrix<T> {
    type Output = T;
    fn index(&self, index: (usize, usize)) -> &T {
        &self.data[self.width * index.0 + index.1]
    }
}

impl<T> IndexMut<(usize, usize)> for Matrix<T> {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut T {
        &mut self.data[self.width * index.0 + index.1]
    }
}

impl<T: Copy> Matrix<T> {
    fn transpose(&mut self) {
        let mut transpose_data = Vec::with_capacity(self.data.len());
        for j in 0..self.width {
            for i in 0..self.height {
                transpose_data.push(self[(i, j)])
            }
        }
        self.data = transpose_data;
        let height = self.height;
        self.height = self.width;
        self.width = height;
    }
}

impl<T> Index<usize> for Matrix<T> {
    type Output = [T];
    fn index(&self, index: usize) -> &[T] {
        &self.data[self.width * index..self.width * (index + 1)]
    }
}

enum InvHilbertAutomata {
    State0,
    State1,
    State2,
    State3,
}

impl InvHilbertAutomata {
    pub fn new() -> Self {
        InvHilbertAutomata::State3
    }

    pub fn next(&mut self, pair: u64, i: &mut usize, j: &mut usize) {
        *self = match &self {
            InvHilbertAutomata::State0 => match pair {
                0 => {
                    *i = *i << 1 | 1;
                    *j = *j << 1 | 1;
                    InvHilbertAutomata::State1
                }
                1 => {
                    *i = *i << 1;
                    *j = *j << 1 | 1;
                    InvHilbertAutomata::State0
                }
                2 => {
                    *i = *i << 1;
                    *j = *j << 1;
                    InvHilbertAutomata::State0
                }
                3 => {
                    *i = *i << 1 | 1;
                    *j = *j << 1;
                    InvHilbertAutomata::State3
                }
                _ => {
                    panic!()
                }
            },
            InvHilbertAutomata::State1 => match pair {
                0 => {
                    *i = *i << 1 | 1;
                    *j = *j << 1 | 1;
                    InvHilbertAutomata::State0
                }
                1 => {
                    *i = *i << 1 | 1;
                    *j = *j << 1;
                    InvHilbertAutomata::State1
                }
                2 => {
                    *i = *i << 1;
                    *j = *j << 1;
                    InvHilbertAutomata::State1
                }
                3 => {
                    *i = *i << 1;
                    *j = *j << 1 | 1;
                    InvHilbertAutomata::State2
                }
                _ => {
                    panic!()
                }
            },
            InvHilbertAutomata::State2 => match pair {
                0 => {
                    *i = *i << 1;
                    *j = *j << 1;
                    InvHilbertAutomata::State3
                }
                1 => {
                    *i = *i << 1 | 1;
                    *j = *j << 1;
                    InvHilbertAutomata::State2
                }
                2 => {
                    *i = *i << 1 | 1;
                    *j = *j << 1 | 1;
                    InvHilbertAutomata::State2
                }
                3 => {
                    *i = *i << 1;
                    *j = *j << 1 | 1;
                    InvHilbertAutomata::State1
                }
                _ => {
                    panic!()
                }
            },
            InvHilbertAutomata::State3 => match pair {
                0 => {
                    *i = *i << 1;
                    *j = *j << 1;
                    InvHilbertAutomata::State2
                }
                1 => {
                    *i = *i << 1;
                    *j = *j << 1 | 1;
                    InvHilbertAutomata::State3
                }
                2 => {
                    *i = *i << 1 | 1;
                    *j = *j << 1 | 1;
                    InvHilbertAutomata::State3
                }
                3 => {
                    *i = *i << 1 | 1;
                    *j = *j << 1;
                    InvHilbertAutomata::State0
                }
                _ => {
                    panic!()
                }
            },
        };
    }
}

fn inv_hilbert(h: u64, mut log_grid_size: u8) -> (usize, usize) {
    let (mut i, mut j) = (0, 0);

    let mut automata = InvHilbertAutomata::new();
    let mut mask = 1 << (log_grid_size - 1) | 1 << (log_grid_size - 2);
    while mask > 0 {
        automata.next((h & mask) >> (log_grid_size - 2), &mut i, &mut j);
        log_grid_size -= 2;
        mask >>= 2;
    }
    (i, j)
}

fn lindenmayer_a(
    l: u8,
    i: &mut isize,
    j: &mut isize,
    d: &mut isize,
    a: &Matrix<u32>,
    b: &mut Matrix<u32>,
    c: &mut Matrix<u32>,
) {
    if l == 0 {
        let i = *i as usize;
        let j = *j as usize;
        let a_row = &a[i];
        let b_row = &b[j];
        let mut sum = u32x16::splat(0);
        for k in (0..a.width).step_by(16) {
            sum += u32x16::from_slice(&a_row[k..]) * u32x16::from_slice(&b_row[k..]);
        }
        c[(i, j)] = sum.reduce_sum();
    } else {
        //*d = (*d+3)&3;
        *d = (*d + 3) % 4;
        lindenmayer_b(l - 1, i, j, d, a, b, c);
        //*j += (*d-1)&1;
        *j += (*d - 1) % 2;
        *i += (*d - 2) % 2;
        *d = (*d + 1) % 4;
        lindenmayer_a(l - 1, i, j, d, a, b, c);
        *j += (*d - 1) % 2;
        *i += (*d - 2) % 2;
        lindenmayer_a(l - 1, i, j, d, a, b, c);
        *d = (*d + 1) % 4;
        *j += (*d - 1) % 2;
        *i += (*d - 2) % 2;
        lindenmayer_b(l - 1, i, j, d, a, b, c);
        *d = (*d + 3) % 4;
    }
}

fn lindenmayer_b(
    l: u8,
    i: &mut isize,
    j: &mut isize,
    d: &mut isize,
    a: &Matrix<u32>,
    b: &mut Matrix<u32>,
    c: &mut Matrix<u32>,
) {
    if l == 0 {
        let i = *i as usize;
        let j = *j as usize;
        let a_row = &a[i];
        let b_row = &b[j];
        let mut sum = u32x16::splat(0);
        for k in (0..a.width).step_by(16) {
            sum += u32x16::from_slice(&a_row[k..]) * u32x16::from_slice(&b_row[k..]);
        }
        c[(i, j)] = sum.reduce_sum();
    } else {
        *d = (*d + 1) % 4;
        lindenmayer_a(l - 1, i, j, d, a, b, c);
        *j += (*d - 1) % 2;
        *i += (*d - 2) % 2;
        *d = (*d + 3) % 4;
        lindenmayer_b(l - 1, i, j, d, a, b, c);
        *j += (*d - 1) % 2;
        *i += (*d - 2) % 2;
        lindenmayer_b(l - 1, i, j, d, a, b, c);
        *d = (*d + 3) % 4;
        *j += (*d - 1) % 2;
        *i += (*d - 2) % 2;
        lindenmayer_a(l - 1, i, j, d, a, b, c);
        *d = (*d + 1) % 4;
    }
}

fn classic_mul(a: &Matrix<u32>, b: &Matrix<u32>, c: &mut Matrix<u32>) -> Duration {
    let now = Instant::now();
    start();
    for i in 0..c.height {
        for j in 0..c.width {
            c[(i, j)] = (0..a.width)
                .into_iter()
                .map(|k| a[(i, k)] * b[(k, j)])
                .sum();
        }
    }
    stop();
    now.elapsed()
}

fn transpose_mul(a: &Matrix<u32>, b: &mut Matrix<u32>, c: &mut Matrix<u32>) -> Duration {
    let now = Instant::now();
    start();
    b.transpose();
    for i in 0..c.height {
        for j in 0..c.width {
            c[(i, j)] = (0..a.width)
                .into_iter()
                .map(|k| a[(i, k)] * b[(j, k)])
                .sum();
        }
    }
    stop();
    now.elapsed()
}

fn hilbert_mul(
    a: &Matrix<u32>,
    b: &mut Matrix<u32>,
    c: &mut Matrix<u32>,
    log_grid_size: u8,
) -> Duration {
    let now = Instant::now();
    start();
    b.transpose();
    for h in 0..c.height * c.width {
        let (i, j) = inv_hilbert(h as u64, log_grid_size);
        c[(i, j)] = (0..a.width)
            .into_iter()
            .map(|k| a[(i, k)] * b[(j, k)])
            .sum();
    }
    stop();
    now.elapsed()
}

use std::simd::u32x16;
fn simd_mul(
    a: &Matrix<u32>,
    b: &mut Matrix<u32>,
    c: &mut Matrix<u32>,
    log_grid_size: u8,
) -> Duration {
    let now = Instant::now();
    start();
    b.transpose();
    for h in 0..c.height * c.width {
        let (i, j) = inv_hilbert(h as u64, log_grid_size);
        let a_row = &a[i];
        let b_row = &b[j];
        let mut sum = u32x16::splat(0);
        for k in (0..a.width).step_by(16) {
            sum += u32x16::from_slice(&a_row[k..]) * u32x16::from_slice(&b_row[k..]);
        }
        c[(i, j)] = sum.reduce_sum();
    }
    stop();
    now.elapsed()
}

fn lindenmayer_mul(
    a: &Matrix<u32>,
    b: &mut Matrix<u32>,
    c: &mut Matrix<u32>,
    half_log_grid_size: u8,
) -> Duration {
    let now = Instant::now();
    start();
    b.transpose();
    lindenmayer_a(half_log_grid_size, &mut 0, &mut 0, &mut 3, &a, b, c);
    stop();
    now.elapsed()
}

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
#[command(propagate_version = true)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Benchmark  
    Benchmark,
    /// Classic algorithm - to be used with callgrind
    Classic,
    /// Transpose - to be used with callgrind
    Transpose,
    /// Hilbert finite automata + transpose - to be used with callgrind
    HilbertFiniteAutomata,
    /// Simd + Hilbert finite automata + transpose - to be used with callgrind
    Simd,
    /// Hilbert lindenmayer system + simd + transpose - to be used with callgrind
    HilbertLindenmayerSystem,
}

fn main() {
    let half_log_grid_size = 10;
    let log_grid_size = 2 * half_log_grid_size;
    let size = 1 << half_log_grid_size;

    let a = Matrix::new(size, size, vec![1; size * size]);
    let mut b = Matrix::new(size, size, vec![1; size * size]);
    let mut c = Matrix::new(size, size, vec![0; size * size]);

    let cli = Cli::parse();
    match cli.command {
        Commands::Benchmark => {
            println!(
                "classic time : {:?}\ntranspose time : {:?}\nhilbert_finite_automata time : {:?}\nsimd time : {:?}\nhilber_lindenmayer time : {:?}\n",
                classic_mul(&a, &b, &mut c),
                transpose_mul(&a, &mut b, &mut c),
                hilbert_mul(&a, &mut b, &mut c, log_grid_size),
                simd_mul(&a, &mut b, &mut c, log_grid_size),
            lindenmayer_mul(&a, &mut b,&mut c, half_log_grid_size),
                );
        }
        Commands::Classic => {
            classic_mul(&a, &b, &mut c);
        }
        Commands::Transpose => {
            transpose_mul(&a, &mut b, &mut c);
        }
        Commands::HilbertFiniteAutomata => {
            hilbert_mul(&a, &mut b, &mut c, log_grid_size);
        }
        Commands::Simd => {
            simd_mul(&a, &mut b, &mut c, log_grid_size);
        }
        Commands::HilbertLindenmayerSystem => {
            lindenmayer_mul(&a, &mut b, &mut c, half_log_grid_size);
        }
    }
}
