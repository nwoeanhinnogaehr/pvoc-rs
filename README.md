# pvoc-rs
A phase vocoder written in Rust

### Example usage
```rust
let pvoc = PhaseVocoder::new(1, 44100.0, 8, 4);
pvoc.process(&input_samples,
                  &mut output_samples,
                  |channels: usize, bins: usize, input: &[Vec<Bin>], output: &mut [Vec<Bin>]| {
    for i in 0..channels {
        for j in 0..bins {
            output[i][j] = input[i][j]; // change this!
        }
    }
}

```
