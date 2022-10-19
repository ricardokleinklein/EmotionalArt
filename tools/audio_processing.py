"""
Set of methods aimed at extracting features from aural information,
including speech and other sound-related sources of data.
"""

import numpy
import librosa


Array = numpy.array


def compute_spectrogram(from_path: str, sampling_rate: float = 16e3,
                        frame_size: int = 2048, hop_length: int = 512,
                        scale: str = "log") -> \
        Array:
    """ Extract the spectrogram from an audio file.

    Args:
        from_path: Path to audio file.
        sampling_rate: Sample rate (8KHz, 16KHz, 32KHz...).
        frame_size: Windowing size.
        hop_length: Hopping between windows.
        scale: Whether to compute in log scale or leave it linear.

    Returns:
        Spectrogram as an array.
    """
    waveform, _ = librosa.load(from_path, sr=sampling_rate)
    s_scale = librosa.stft(waveform, n_fft=frame_size, hop_length=hop_length)
    y_scale = numpy.abs(s_scale) * numpy.abs(s_scale)
    if scale == "linear":
        return y_scale
    return librosa.power_to_db(y_scale)
