#![feature(associated_consts)]

extern crate dft;

use dft::c64;

trait Processor {
    fn process(&self, input: &[&[c64]], output: &mut [&mut [c64]]);
}

struct PhaseVocoder<P: Processor> {
    sample_rate: f64,
    freq_res: u32,
    time_res: u32,
    processor: P,
}

impl<P: Processor> PhaseVocoder<P> {
    fn new(sample_rate: f64, freq_res: u32, time_res: u32, processor: P) -> PhaseVocoder<P> {
        PhaseVocoder {
            sample_rate: sample_rate,
            freq_res: freq_res,
            time_res: time_res,
            processor: processor,
        }
    }

    fn write_in_samples(&mut self, samples: &[&[f64]]) {}

    fn read_out_samples(&mut self, samples: &mut [&mut [f64]]) {}
}
