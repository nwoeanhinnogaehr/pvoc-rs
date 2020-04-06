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

#[derive(Clone, Copy, Debug)]
struct PhaseVocoderSettings {
    sample_rate: f64,
    frame_size: usize,
    time_res: usize,
}

impl PhaseVocoderSettings {
    fn new(sample_rate: f64, frame_size: usize, time_res: usize) -> Self {
        let mut frame_size = frame_size / time_res * time_res;
        if frame_size == 0 {
            frame_size = time_res;
        }

        // If `frame_size == 1`, computing the window would panic.
        assert!(frame_size > 1);

        PhaseVocoderSettings {
            sample_rate,
            frame_size,
            time_res,
        }
    }

    fn phase_to_frequency(&self, bin: usize, phase: f64) -> f64 {
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

    fn frequency_to_phase(&self, freq: f64) -> f64 {
        let step_size = self.frame_size as f64 / self.time_res as f64;
        2.0 * PI * freq / self.sample_rate * step_size
    }

    pub fn stepsize(&self) -> usize {
        self.frame_size / self.time_res
    }
}

pub struct PhaseVocoderAnalysis {
    settings: PhaseVocoderSettings,
    in_buf: VecDeque<f64>,
    last_phase: Vec<f64>,
}

impl PhaseVocoderAnalysis {
    pub fn new(sample_rate: f64, frame_size: usize, time_res: usize) -> Self {
        Self::from_settings(PhaseVocoderSettings::new(sample_rate, frame_size, time_res))
    }

    fn from_settings(settings: PhaseVocoderSettings) -> Self {
        Self {
            settings,
            in_buf: VecDeque::new(),
            last_phase: vec![0.0; settings.frame_size],
        }
    }

    pub fn push_samples<S: Float + ToPrimitive + FromPrimitive>(&mut self, samples: &[S]) {
        for sample in samples.iter() {
            self.in_buf.push_back(sample.to_f64().unwrap());
        }
    }

    pub fn pop_samples(&mut self) {
        for _ in 0..self.settings.stepsize() {
            self.in_buf.pop_front();
        }
    }

    pub fn analyse(
        &mut self,
        forward_fft: &dyn rustfft::FFT<f64>,
        fft_in: &mut Vec<c64>,
        fft_out: &mut Vec<c64>,
        window: &Vec<f64>,
        analysis_out: &mut Vec<Bin>,
    ) {
        // read in
        for i in 0..self.settings.frame_size {
            fft_in[i] = c64::new(self.in_buf[i] * window[i], 0.0);
        }

        forward_fft.process(fft_in, fft_out);

        for i in 0..self.settings.frame_size {
            let x = fft_out[i];
            let (amp, phase) = x.to_polar();
            let freq = self
                .settings
                .phase_to_frequency(i, phase - self.last_phase[i]);
            self.last_phase[i] = phase;

            analysis_out[i] = Bin::new(freq, amp * 2.0);
        }
    }
}

/// A phase vocoder.
///
/// Roughly translated from http://blogs.zynaptiq.com/bernsee/pitch-shifting-using-the-ft/
pub struct PhaseVocoder {
    channels: usize,
    settings: PhaseVocoderSettings,

    analysis: Vec<PhaseVocoderAnalysis>,

    samples_waiting: usize,
    out_buf: Vec<VecDeque<f64>>,
    sum_phase: Vec<Vec<f64>>,
    output_accum: Vec<VecDeque<f64>>,

    forward_fft: Arc<dyn rustfft::FFT<f64>>,
    backward_fft: Arc<dyn rustfft::FFT<f64>>,

    window: Vec<f64>,

    fft_in: Vec<c64>,
    fft_out: Vec<c64>,
    analysis_out: Vec<Vec<Bin>>,
    synthesis_in: Vec<Vec<Bin>>,
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
        let settings = PhaseVocoderSettings::new(sample_rate, frame_size, time_res);

        Self::from_settings(channels, settings)
    }

    fn from_settings(channels: usize, settings: PhaseVocoderSettings) -> Self {
        let mut planner_forward = rustfft::FFTplanner::new(false);
        let mut planner_backward = rustfft::FFTplanner::new(true);

        let mut analysis = Vec::new();
        for _ in 0..channels {
            analysis.push(PhaseVocoderAnalysis::from_settings(settings));
        }

        let frame_size = settings.frame_size;

        PhaseVocoder {
            channels,
            settings,

            samples_waiting: 0,
            out_buf: vec![VecDeque::new(); channels],
            sum_phase: vec![vec![0.0; frame_size]; channels],
            output_accum: vec![VecDeque::new(); channels],

            forward_fft: planner_forward.plan_fft(frame_size),
            backward_fft: planner_backward.plan_fft(frame_size),

            window: apodize::hanning_iter(frame_size)
                .map(|x| x.sqrt())
                .collect(),

            fft_in: vec![c64::new(0.0, 0.0); frame_size],
            fft_out: vec![c64::new(0.0, 0.0); frame_size],
            analysis_out: vec![vec![Bin::empty(); frame_size]; channels],
            synthesis_in: vec![vec![Bin::empty(); frame_size]; channels],

            analysis,
        }
    }

    pub fn num_channels(&self) -> usize {
        self.channels
    }

    pub fn num_bins(&self) -> usize {
        self.settings.frame_size
    }

    pub fn time_res(&self) -> usize {
        self.settings.time_res
    }

    pub fn sample_rate(&self) -> f64 {
        self.settings.sample_rate
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
    ///
    /// # Remark
    /// The `synthesis_input` passed to the `processor_function` is currently initialised to empty
    /// bins. This behaviour may change in a future release, so make sure that your implementation
    /// does not rely on it.
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
            self.analysis[chan].push_samples(&input[chan]);
            self.samples_waiting += input[chan].len();
        }

        while self.samples_waiting >= 2 * self.num_bins() * self.channels {
            let frame_sizef = self.num_bins() as f64;
            let time_resf = self.time_res() as f64;
            let step_size = self.settings.stepsize() as f64;

            for _ in 0..self.time_res() {
                // Initialise the synthesis bins to empty bins.
                // This may be removed in a future release.
                for synthesis_channel in self.synthesis_in.iter_mut() {
                    for bin in synthesis_channel.iter_mut() {
                        *bin = Bin::empty();
                    }
                }

                // ANALYSIS
                for chan in 0..self.channels {
                    self.analysis[chan].analyse(
                        self.forward_fft.as_ref(),
                        &mut self.fft_in,
                        &mut self.fft_out,
                        &self.window,
                        &mut self.analysis_out[chan],
                    );
                }

                // PROCESSING
                processor(
                    self.channels,
                    self.num_bins(),
                    &self.analysis_out,
                    &mut self.synthesis_in,
                );

                // SYNTHESIS
                for chan in 0..self.channels {
                    for i in 0..self.num_bins() {
                        let amp = self.synthesis_in[chan][i].amp;
                        let freq = self.synthesis_in[chan][i].freq;
                        let phase = self.frequency_to_phase(freq);
                        self.sum_phase[chan][i] += phase;
                        let phase = self.sum_phase[chan][i];

                        self.fft_in[i] = c64::from_polar(&amp, &phase);
                    }

                    self.backward_fft
                        .process(&mut self.fft_in, &mut self.fft_out);

                    // accumulate
                    for i in 0..self.num_bins() {
                        if i == self.output_accum[chan].len() {
                            self.output_accum[chan].push_back(0.0);
                        }
                        self.output_accum[chan][i] +=
                            self.window[i] * self.fft_out[i].re / (frame_sizef * time_resf);
                    }

                    // write out
                    for _ in 0..step_size as usize {
                        self.out_buf[chan].push_back(self.output_accum[chan].pop_front().unwrap());
                    }
                    self.analysis[chan].pop_samples();
                }
            }
            self.samples_waiting -= self.num_bins() * self.channels;
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
        self.settings.phase_to_frequency(bin, phase)
    }

    pub fn frequency_to_phase(&self, freq: f64) -> f64 {
        self.settings.frequency_to_phase(freq)
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
fn test_data_is_reconstructed_two_channels(
    mut pvoc: PhaseVocoder,
    input_samples_left: &[f32],
    input_samples_right: &[f32],
) {
    let mut output_samples_left = vec![0.0; input_samples_left.len()];
    let mut output_samples_right = vec![0.0; input_samples_right.len()];
    let frame_size = pvoc.num_bins();
    // Pre-padding, not collecting any output.
    pvoc.process(
        &[&vec![0.0; frame_size], &vec![0.0; frame_size]],
        &mut [&mut Vec::new(), &mut Vec::new()],
        identity,
    );
    // The data itself, collecting some output that we will discard
    let mut scratch_left = vec![0.0; frame_size];
    let mut scratch_right = vec![0.0; frame_size];
    pvoc.process(
        &[&input_samples_left, &input_samples_right],
        &mut [&mut scratch_left, &mut scratch_right],
        identity,
    );
    // Post-padding and collecting all output
    pvoc.process(
        &[&vec![0.0; frame_size], &vec![0.0; frame_size]],
        &mut [&mut output_samples_left, &mut output_samples_right],
        identity,
    );

    assert_ulps_eq!(
        input_samples_left,
        output_samples_left.as_slice(),
        epsilon = 1e-2
    );
    assert_ulps_eq!(
        input_samples_right,
        output_samples_right.as_slice(),
        epsilon = 1e-2
    );
}

#[cfg(test)]
fn test_data_is_reconstructed(mut pvoc: PhaseVocoder, input_samples: &[f32]) {
    let mut output_samples = vec![0.0; input_samples.len()];
    let frame_size = pvoc.num_bins();
    // Pre-padding, not collecting any output.
    pvoc.process(&[&vec![0.0; frame_size]], &mut [&mut Vec::new()], identity);
    // The data itself, collecting some output that we will discard
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
fn identity_transform_reconstructs_original_data_random_data_with_two_channels() {
    let pvoc = PhaseVocoder::new(2, 44100.0, 128, 128 / 4);
    let input_samples_all = include!("./random_test_data.rs");
    let (input_samples_left, input_samples_right) =
        input_samples_all.split_at(input_samples_all.len() / 2);
    test_data_is_reconstructed_two_channels(pvoc, &input_samples_left, &input_samples_right);
}

#[test]
fn process_works_with_sample_res_equal_to_window() {
    let mut pvoc = PhaseVocoder::new(1, 44100.0, 256, 256);
    let input_len = 1024;
    let input_samples = vec![0.0; input_len];
    let mut output_samples = vec![0.0; input_len];
    pvoc.process(&[&input_samples], &mut [&mut output_samples], identity);
}

#[test]
fn process_works_with_sample_res_equal_to_window_two_channels() {
    let mut pvoc = PhaseVocoder::new(2, 44100.0, 256, 256);
    let input_len = 1024;
    let input_samples_left = vec![0.0; input_len];
    let input_samples_right = vec![0.0; input_len];
    let mut output_samples_left = vec![0.0; input_len];
    let mut output_samples_right = vec![0.0; input_len];
    pvoc.process(
        &[&input_samples_left, &input_samples_right],
        &mut [&mut output_samples_left, &mut output_samples_right],
        identity,
    );
}

#[test]
fn process_works_when_reading_sample_by_sample() {
    let mut pvoc = PhaseVocoder::new(1, 44100.0, 8, 2);
    let input_len = 32;
    let input_samples = vec![0.0; input_len];
    let mut output_samples = vec![0.0; input_len];
    for i in 0..input_len {
        pvoc.process(
            &[&input_samples[i..i + 1]],
            &mut [&mut output_samples],
            identity,
        );
    }
}

#[test]
fn process_works_when_reading_sample_by_sample_two_channels() {
    let mut pvoc = PhaseVocoder::new(2, 44100.0, 8, 2);
    let input_len = 32;
    let input_samples_left = vec![0.0; input_len];
    let input_samples_right = vec![0.0; input_len];
    let mut output_samples_left = vec![0.0; input_len];
    let mut output_samples_right = vec![0.0; input_len];
    for i in 0..input_len {
        pvoc.process(
            &[
                &input_samples_left[i..i + 1],
                &input_samples_right[i..i + 1],
            ],
            &mut [&mut output_samples_left, &mut output_samples_right],
            identity,
        );
    }
}
