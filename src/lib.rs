extern crate dft;

use dft::{Operation, Plan, c64};
use std::f64::consts::PI;
use std::collections::VecDeque;

#[derive(Copy, Clone)]
pub struct Bin {
    freq: f64,
    amp: f64,
}

impl Bin {
    pub fn new(freq: f64, amp: f64) -> Bin {
        Bin {
            freq: freq,
            amp: amp,
        }
    }
}

pub trait Processor {
    fn process(&self, input: &[Vec<Bin>], output: &mut [Vec<Bin>]);
}

pub struct PhaseVocoder<P: Processor> {
    channels: usize,
    sample_rate: f64,
    frame_size: usize,
    time_res: usize,
    processor: P,

    samples_waiting: usize,
    in_buf: Vec<VecDeque<f64>>,
    out_buf: Vec<VecDeque<f64>>,
    last_phase: Vec<Vec<f64>>,
    sum_phase: Vec<Vec<f64>>,
    output_accum: Vec<VecDeque<f64>>,
}

impl<P: Processor> PhaseVocoder<P> {
    pub fn new(channels: usize,
               sample_rate: f64,
               freq_res: usize,
               time_res: usize,
               processor: P)
               -> PhaseVocoder<P> {
        let frame_size = 1 << freq_res;
        PhaseVocoder {
            channels: channels,
            sample_rate: sample_rate,
            frame_size: frame_size,
            time_res: time_res,
            processor: processor,

            samples_waiting: 0,
            in_buf: vec![VecDeque::new(); channels],
            out_buf: vec![VecDeque::new(); channels],
            last_phase: vec![vec![0.0; frame_size]; channels],
            sum_phase: vec![vec![0.0; frame_size]; channels],
            output_accum: vec![VecDeque::new(); channels],
        }
    }

    pub fn write_in_samples(&mut self, samples: &[&[f64]]) {
        assert_eq!(samples.len(), self.channels);
        for c in 0..samples.len() {
            for s in 0..samples[c].len() {
                self.in_buf[c].push_back(samples[c][s]);
                self.samples_waiting += 1;
                if self.samples_waiting == self.frame_size {
                    self.process();
                    self.samples_waiting = 0;
                }
            }
        }
    }

    pub fn read_out_samples(&mut self, samples: &mut [&mut [f64]]) {
        assert_eq!(samples.len(), self.channels);
        for chan in 0..self.channels {
            for samp in 0..samples[chan].len() {
                samples[chan][samp] = match self.out_buf[chan].pop_front() {
                    Some(x) => x,
                    None => break,
                }
            }
        }
    }

    fn process(&mut self) {
        let frame_size = self.frame_size;
        let step_size = frame_size / self.time_res;
        let expect = 2.0 * PI * (step_size as f64) / (frame_size as f64);
        let freq_per_bin = self.sample_rate / (frame_size as f64);
        let mut analysis_out = vec![vec![Bin::new(0.0, 0.0); frame_size]; self.channels];
        let mut synthesis_in = vec![vec![Bin::new(0.0, 0.0); frame_size]; self.channels];

        // ANALYSIS
        for chan in 0..self.channels {
            let samples = &self.in_buf[chan];
            let mut last_phase = &mut self.last_phase[chan];
            let mut fft_worksp = vec![c64::new(0.0, 0.0); frame_size];
            for i in 0..frame_size {
                let window = window((i as f64) / (frame_size as f64));
                fft_worksp[i] = c64::new(samples[i] * window, 0.0);
            }
            let plan = Plan::new(Operation::Forward, frame_size);
            dft::transform(&mut fft_worksp, &plan);

            for i in 0..frame_size {
                let x = fft_worksp[i];

                let (amp, phase) = x.to_polar();
                let mut tmp = phase - last_phase[i];
                last_phase[i] = phase;
                tmp -= (i as f64) * expect;
                let mut qpd = (tmp / PI) as i64;
                qpd += qpd.signum() * (qpd & 1);
                tmp -= PI * (qpd as f64);
                tmp = (self.time_res as f64) * tmp / (2.0 * PI);
                tmp = (i as f64) * freq_per_bin + tmp * freq_per_bin;

                analysis_out[chan][i] = Bin::new(amp, tmp);
            }
        }

        // PROCESSING
        self.processor.process(&analysis_out, &mut synthesis_in);

        // SYNTHESIS
        for chan in 0..self.channels {
            let mut sum_phase = &mut self.sum_phase[chan];
            let mut fft_worksp = vec![c64::new(0.0, 0.0); frame_size];
            for i in 0..frame_size {
                let amp = synthesis_in[chan][i].amp;
                let mut tmp = synthesis_in[chan][i].freq;

                tmp -= (i as f64) * freq_per_bin;
                tmp /= freq_per_bin;
                tmp = 2.0 * PI * tmp / (self.time_res as f64);
                tmp += (i as f64) * expect;
                sum_phase[i] += tmp;
                let phase = sum_phase[i];

                fft_worksp[i] = c64::from_polar(&amp, &phase);
            }
            let plan = Plan::new(Operation::Backward, frame_size);
            dft::transform(&mut fft_worksp, &plan);
            for i in 0..frame_size {
                let window = window((i as f64) / (frame_size as f64));
                if i > self.output_accum[chan].len() {
                    self.output_accum[chan].push_back(0.0);
                }
                self.output_accum[chan][i] += 2.0 * window * fft_worksp[i].re /
                                              ((frame_size as f64) * (self.time_res as f64));
            }
            for _ in 0..step_size {
                self.out_buf[chan].push_back(self.output_accum[chan].pop_front().unwrap());
            }
        }
    }
}

fn window(x: f64) -> f64 {
    -0.5 * (2.0 * PI * x).cos()
}
