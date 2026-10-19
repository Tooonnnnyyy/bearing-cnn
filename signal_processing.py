"""Window-local offline transforms and correctly scaled one-sided spectra.

No cutoff is inferred from the file name. Frequency arguments are in Hz and
require a known sampling rate. Importing this module does not read recordings.
"""
from dataclasses import asdict, dataclass

import numpy as np
from scipy import signal


@dataclass(frozen=True)
class PreprocessConfig:
    detrend: str = 'none'
    sample_rate: float | None = None
    low_hz: float | None = None
    high_hz: float | None = None
    notch_hz: float | None = None
    notch_q: float = 30.0
    order: int = 4

    def validate(self):
        if self.detrend not in ('none', 'constant', 'linear'):
            raise ValueError('detrend must be none, constant or linear')
        if self.sample_rate is not None and (not np.isfinite(self.sample_rate) or self.sample_rate <= 0):
            raise ValueError('sample_rate must be finite and positive')
        if not isinstance(self.order, int) or not 1 <= self.order <= 10:
            raise ValueError('Butterworth order must be an integer from 1 to 10')
        if not np.isfinite(self.notch_q) or self.notch_q <= 0:
            raise ValueError('notch_q must be finite and positive')
        for value in (self.low_hz, self.high_hz, self.notch_hz):
            if value is not None:
                if self.sample_rate is None:
                    raise ValueError('Hz filters require --sample-rate from acquisition metadata')
                if not np.isfinite(value) or not 0 < value < self.sample_rate / 2:
                    raise ValueError('Every filter frequency must be between 0 and Nyquist (fs/2)')
        if self.low_hz is not None and self.high_hz is not None and self.low_hz >= self.high_hz:
            raise ValueError('low_hz must be below high_hz')
        return self

    def to_dict(self):
        return asdict(self)


def checked_windows(windows):
    x = np.asarray(windows, dtype=np.float64)
    if x.ndim != 2 or not x.shape[0] or x.shape[1] < 8 or not np.isfinite(x).all():
        raise ValueError('Expected finite windows shaped (examples, samples), at least 8 samples')
    return x


def preprocess(windows, config):
    """Process each window separately; no filter reads across split boundaries.

Forward-backward filters use future samples *within* a complete window.
They are not causal streaming filters. Edge transients remain possible.
"""
    config.validate()
    x = checked_windows(windows).copy()
    if config.detrend != 'none':
        x = signal.detrend(x, axis=-1, type=config.detrend)
    if config.low_hz is not None or config.high_hz is not None:
        if config.low_hz is not None and config.high_hz is not None:
            cutoff, kind = [config.low_hz, config.high_hz], 'bandpass'
        elif config.low_hz is not None:
            cutoff, kind = config.low_hz, 'highpass'
        else:
            cutoff, kind = config.high_hz, 'lowpass'
        sos = signal.butter(config.order, cutoff, btype=kind, fs=config.sample_rate, output='sos')
        x = signal.sosfiltfilt(sos, x, axis=-1)
    if config.notch_hz is not None:
        b, a = signal.iirnotch(config.notch_hz, config.notch_q, fs=config.sample_rate)
        x = signal.filtfilt(b, a, x, axis=-1)
    if not np.isfinite(x).all():
        raise ValueError('Processing produced non-finite values')
    return x


def amplitude_spectrum(windows, sample_rate=None, taper=True):
    """rFFT amplitude with coherent-gain correction, preserving DC/Nyquist.

For a bin-centered sinusoid this estimates its peak amplitude. Off-bin tones
spread across bins; this is not a PSD and window correction is not denoising.
"""
    x = checked_windows(windows)
    PreprocessConfig(sample_rate=sample_rate).validate()
    n = x.shape[-1]
    window = signal.windows.hann(n, sym=False) if taper else np.ones(n)
    spectrum = np.abs(np.fft.rfft(x * window, axis=-1)) / window.sum()
    spectrum[:, 1:-1 if n % 2 == 0 else None] *= 2
    frequency = np.fft.rfftfreq(n, d=1.0 / (sample_rate or 1.0))
    return frequency, spectrum


def mean_psd(windows, sample_rate=None):
    """Welch PSD per window, then average windows without joining recordings."""
    x = checked_windows(windows)
    PreprocessConfig(sample_rate=sample_rate).validate()
    size = min(256, x.shape[-1])
    frequency, density = signal.welch(
        x, fs=sample_rate or 1.0, window='hann', nperseg=size,
        noverlap=size // 2, detrend=False, scaling='density', axis=-1)
    return frequency, density.mean(axis=0)


def representation(windows, kind='time'):
    x = checked_windows(windows)
    if kind == 'time':
        return x
    if kind == 'log-spectrum':
        return np.log1p(amplitude_spectrum(x)[1])
    raise ValueError('representation must be time or log-spectrum')


def add_preprocessing_arguments(parser):
    parser.add_argument('--detrend', choices=('none', 'constant', 'linear'), default='none')
    parser.add_argument('--sample-rate', type=float, help='Verified acquisition rate in Hz; never inferred')
    parser.add_argument('--low-hz', type=float, help='High-pass edge, or band-pass lower edge')
    parser.add_argument('--high-hz', type=float, help='Low-pass edge, or band-pass upper edge')
    parser.add_argument('--notch-hz', type=float, help='Known narrow-band interference frequency')
    parser.add_argument('--notch-q', type=float, default=30.0)
    parser.add_argument('--order', type=int, default=4)


def config_from_args(args):
    return PreprocessConfig(**{name: getattr(args, name) for name in PreprocessConfig.__dataclass_fields__}).validate()
