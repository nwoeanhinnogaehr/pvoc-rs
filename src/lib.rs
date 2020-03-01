extern crate apodize;
extern crate rustfft;
#[cfg(test)]
#[macro_use]
extern crate approx;

use rustfft::num_complex::Complex;
use rustfft::num_traits::{Float, FromPrimitive, ToPrimitive};
use std::collections::VecDeque;
use std::f64::consts::PI;
use std::sync::Arc;

#[allow(non_camel_case_types)]
type c64 = Complex<f64>;

/// Represents a component of the spectrum, composed of a frequency and amplitude.
#[derive(Copy, Clone)]
pub struct Bin {
    pub freq: f64,
    pub amp: f64,
}

impl Bin {
    pub fn new(freq: f64, amp: f64) -> Bin {
        Bin {
            freq: freq,
            amp: amp,
        }
    }
    pub fn empty() -> Bin {
        Bin {
            freq: 0.0,
            amp: 0.0,
        }
    }
}

/// A phase vocoder.
///
/// Roughly translated from http://blogs.zynaptiq.com/bernsee/pitch-shifting-using-the-ft/
pub struct PhaseVocoder {
    channels: usize,
    sample_rate: f64,
    frame_size: usize,
    time_res: usize,

    samples_waiting: usize,
    in_buf: Vec<VecDeque<f64>>,
    out_buf: Vec<VecDeque<f64>>,
    last_phase: Vec<Vec<f64>>,
    sum_phase: Vec<Vec<f64>>,
    output_accum: Vec<VecDeque<f64>>,

    forward_fft: Arc<dyn rustfft::FFT<f64>>,
    backward_fft: Arc<dyn rustfft::FFT<f64>>,

    window: Vec<f64>,
}

impl PhaseVocoder {
    /// Constructs a new phase vocoder.
    ///
    /// `channels` is the number of channels of audio.
    ///
    /// `sample_rate` is the sample rate.
    ///
    /// `frame_size` is the fourier transform size. It must be `> 1`.
    /// For optimal computation speed, this should be a power of 2.
    /// Will be rounded to a multiple of `time_res`.
    ///
    /// `time_res` is the number of frames to overlap.
    ///
    /// # Panics
    /// Panics if `frame_size` is `<= 1` after rounding.
    pub fn new(
        channels: usize,
        sample_rate: f64,
        frame_size: usize,
        time_res: usize,
    ) -> PhaseVocoder {
        let mut frame_size = frame_size / time_res * time_res;
        if frame_size == 0 {
            frame_size = time_res;
        }

        // If `frame_size == 1`, computing the window would panic.
        assert!(frame_size > 1);

        let mut planner_forward = rustfft::FFTplanner::new(false);
        let mut planner_backward = rustfft::FFTplanner::new(true);

        PhaseVocoder {
            channels: channels,
            sample_rate: sample_rate,
            frame_size: frame_size,
            time_res: time_res,

            samples_waiting: 0,
            in_buf: vec![VecDeque::new(); channels],
            out_buf: vec![VecDeque::new(); channels],
            last_phase: vec![vec![0.0; frame_size]; channels],
            sum_phase: vec![vec![0.0; frame_size]; channels],
            output_accum: vec![VecDeque::new(); channels],

            forward_fft: planner_forward.plan_fft(frame_size),
            backward_fft: planner_backward.plan_fft(frame_size),

            window: apodize::hanning_iter(frame_size)
                .map(|x| x.sqrt())
                .collect(),
        }
    }

    pub fn num_channels(&self) -> usize {
        self.channels
    }

    pub fn num_bins(&self) -> usize {
        self.frame_size
    }

    pub fn time_res(&self) -> usize {
        self.time_res
    }

    pub fn sample_rate(&self) -> f64 {
        self.sample_rate
    }

    /// Reads samples from `input`, processes the samples, then resynthesizes as many samples as
    /// possible into `output`. Returns the number of frames written to `output`.
    ///
    /// `processor` is a function to manipulate the spectrum before it is resynthesized. Its
    /// arguments are respectively `num_channels`, `num_bins`, `analysis_output` and
    /// `synthesis_input`.
    ///
    /// Samples are expected to be normalized to the range [-1, 1].
    ///
    /// This method can be called multiple times on the same `PhaseVocoder`.
    /// If this happens, in the analysis step, it will be assumed that the `input` is a continuation
    /// of the `input` that was passed during the previous call.
    ///
    /// It is possible that not enough data is available yet to fill `output` completely.
    /// In that case, only the first frames of `output` will be written to.
    /// Conversely, if there is more data available than `output` can hold, the remaining
    /// output is kept in the `PhaseVocoder` and can be retrieved with another call to
    /// `process` when more input data is available.
    pub fn process<S, F>(
        &mut self,
        input: &[&[S]],
        output: &mut [&mut [S]],
        mut processor: F,
    ) -> usize
    where
        S: Float + ToPrimitive + FromPrimitive,
        F: FnMut(usize, usize, &[Vec<Bin>], &mut [Vec<Bin>]),
    {
        assert_eq!(input.len(), self.channels);
        assert_eq!(output.len(), self.channels);

        // push samples to input queue
        for chan in 0..input.len() {
            for samp in 0..input[chan].len() {
                self.in_buf[chan].push_back(input[chan][samp].to_f64().unwrap());
                self.samples_waiting += 1;
            }
        }

        while self.samples_waiting > self.frame_size * self.channels {
            let frame_sizef = self.frame_size as f64;
            let time_resf = self.time_res as f64;
            let step_size = frame_sizef / time_resf;
            let mut fft_in = vec![c64::new(0.0, 0.0); self.frame_size];
            let mut fft_out = vec![c64::new(0.0, 0.0); self.frame_size];

            for _ in 0..self.time_res {
                let mut analysis_out = vec![vec![Bin::empty(); self.frame_size]; self.channels];
                let mut synthesis_in = vec![vec![Bin::empty(); self.frame_size]; self.channels];

                // ANALYSIS
                for chan in 0..self.channels {
                    // read in
                    for i in 0..self.frame_size {
                        fft_in[i] = c64::new(self.in_buf[chan][i] * self.window[i], 0.0);
                    }

                    self.forward_fft.process(&mut fft_in, &mut fft_out);

                    for i in 0..self.frame_size {
                        let x = fft_out[i];
                        let (amp, phase) = x.to_polar();
                        let freq = self.phase_to_frequency(i, phase - self.last_phase[chan][i]);
                        self.last_phase[chan][i] = phase;

                        analysis_out[chan][i] = Bin::new(freq, amp * 2.0);
                    }
                }

                // PROCESSING
                processor(
                    self.channels,
                    self.frame_size,
                    &analysis_out,
                    &mut synthesis_in,
                );

                // SYNTHESIS
                for chan in 0..self.channels {
                    for i in 0..self.frame_size {
                        let amp = synthesis_in[chan][i].amp;
                        let freq = synthesis_in[chan][i].freq;
                        let phase = self.frequency_to_phase(i, freq);
                        self.sum_phase[chan][i] += phase;
                        let phase = self.sum_phase[chan][i];

                        fft_in[i] = c64::from_polar(&amp, &phase);
                    }

                    self.backward_fft.process(&mut fft_in, &mut fft_out);

                    // accumulate
                    for i in 0..self.frame_size {
                        if i == self.output_accum[chan].len() {
                            self.output_accum[chan].push_back(0.0);
                        }
                        self.output_accum[chan][i] +=
                            self.window[i] * fft_out[i].re / (frame_sizef * time_resf);
                    }

                    // write out
                    for _ in 0..step_size as usize {
                        self.out_buf[chan].push_back(self.output_accum[chan].pop_front().unwrap());
                        self.in_buf[chan].pop_front();
                    }
                }
            }
            self.samples_waiting -= self.frame_size * self.channels;
        }

        // pop samples from output queue
        let mut n_written = 0;
        for chan in 0..self.channels {
            for samp in 0..output[chan].len() {
                output[chan][samp] = match self.out_buf[chan].pop_front() {
                    Some(x) => FromPrimitive::from_f64(x).unwrap(),
                    None => break,
                };
                n_written += 1;
            }
        }
        n_written / self.channels
    }

    pub fn phase_to_frequency(&self, bin: usize, phase: f64) -> f64 {
        let frame_sizef = self.frame_size as f64;
        let freq_per_bin = self.sample_rate / frame_sizef;
        let time_resf = self.time_res as f64;
        let step_size = frame_sizef / time_resf;
        let expect = 2.0 * PI * step_size / frame_sizef;
        let mut tmp = phase;
        tmp -= (bin as f64) * expect;
        let mut qpd = (tmp / PI) as i32;
        if qpd >= 0 {
            qpd += qpd & 1;
        } else {
            qpd -= qpd & 1;
        }
        tmp -= PI * (qpd as f64);
        tmp = time_resf * tmp / (2.0 * PI);
        tmp = (bin as f64) * freq_per_bin + tmp * freq_per_bin;
        tmp
    }

    pub fn frequency_to_phase(&self, bin: usize, freq: f64) -> f64 {
        let frame_sizef = self.frame_size as f64;
        let freq_per_bin = self.sample_rate / frame_sizef;
        let time_resf = self.time_res as f64;
        let step_size = frame_sizef / time_resf;
        let expect = 2.0 * PI * step_size / frame_sizef;
        let mut tmp = freq - (bin as f64) * freq_per_bin;
        tmp /= freq_per_bin;
        tmp = 2.0 * PI * tmp / time_resf;
        tmp += (bin as f64) * expect;
        tmp
    }
}

#[cfg(test)]
fn identity(channels: usize, bins: usize, input: &[Vec<Bin>], output: &mut [Vec<Bin>]) {
    for i in 0..channels {
        for j in 0..bins {
            output[i][j] = input[i][j];
        }
    }
}

#[cfg(test)]
fn test_data_is_reconstructed(mut pvoc: PhaseVocoder, input_samples: &[f32]) {
    let mut output_samples = vec![0.0; input_samples.len()];
    let frame_size = pvoc.num_bins();
    // Pre-padding, not collecting any output.
    pvoc.process(&[&vec![0.0; frame_size]], &mut [&mut Vec::new()], identity);
    // The data-itself, collecting some output that we will discard
    let mut scratch = vec![0.0; frame_size];
    pvoc.process(&[&input_samples], &mut [&mut scratch], identity);
    // Post-padding and collecting all output
    pvoc.process(
        &[&vec![0.0; frame_size]],
        &mut [&mut output_samples],
        identity,
    );

    assert_ulps_eq!(input_samples, output_samples.as_slice(), epsilon = 1e-2);
}

#[test]
fn identity_transform_reconstructs_original_data_hat_function() {
    let window_len = 256;
    let pvoc = PhaseVocoder::new(1, 44100.0, window_len, window_len / 4);
    let input_len = 1024;
    let mut input_samples = vec![0.0; input_len];
    for i in 0..input_len {
        if i < input_len / 2 {
            input_samples[i] = (i as f32) / ((input_len / 2) as f32)
        } else {
            input_samples[i] = 2.0 - (i as f32) / ((input_len / 2) as f32);
        }
    }

    test_data_is_reconstructed(pvoc, input_samples.as_slice());
}

#[test]
fn identity_transform_reconstructs_original_data_random_data() {
    let pvoc = PhaseVocoder::new(1, 44100.0, 256, 256 / 4);
    let input_samples = include!("./random_test_data.rs");
    test_data_is_reconstructed(pvoc, &input_samples);
}

#[test]
fn process_works_with_sample_res_equal_to_window() {
    let mut pvoc = PhaseVocoder::new(1, 44100.0, 256, 256);
    let input_len = 1024;
    let input_samples = vec![0.0; input_len];
    let mut output_samples = vec![0.0; input_len];
    pvoc.process(
        &[&input_samples],
        &mut [&mut output_samples],
        |channels: usize, bins: usize, input: &[Vec<Bin>], output: &mut [Vec<Bin>]| {
            for i in 0..channels {
                for j in 0..bins {
                    output[i][j] = input[i][j];
                }
            }
        },
    );
}
