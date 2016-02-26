extern crate dft;

use dft::c64;

trait Processor {
    fn process(&self, input: &[c64], output: &mut [c64]);
}

struct PhaseVocoder<P: Processor> {
    sample_rate: f64,
    block_size: u32,
    overlap: u32,
    processor: P,
}

impl<P: Processor> PhaseVocoder<P> {
    fn new(sample_rate: f64, block_size: u32, overlap: u32, processor: P) -> PhaseVocoder<P> {
        PhaseVocoder {
            sample_rate: sample_rate,
            block_size: block_size,
            overlap: overlap,
            processor: processor,
        }
    }

    fn write_in_samples(&mut self, samples: &[f64]) {}

    fn read_out_samples(&mut self, samples: &mut [f64]) {}
}
