function [Z, theta] = bw_phases(X, fs, band, decim, trimSamples)
% BW_PHASES  Phase vectors from multichannel signals (Sec. 2.1 of the paper).
%
%   [Z, theta] = bw_phases(X, fs, band, decim, trimSamples)
%
%   X      T x p real signals (time x channels), e.g. data_rs1(:,:,subject).
%   fs     sampling rate (Hz); band = [f1 f2] (Hz); decim = decimation factor applied after
%          the analytic signal; trimSamples = samples removed at both ends after decimation.
%   Z      T' x p complex unit-modulus phase vectors, Z(t,j) = exp(1i*theta_j(t)).
%   theta  T' x p phases; pass to bw_weighted_eigenvectors.m and then to bw_kmeans.m.
%
%   Pipeline: 3rd-order Butterworth band-pass (zero phase, filtfilt) -> FFT-based analytic
%   signal (hilbert) -> decimation -> edge trimming. The amplitude is discarded.
%   NOTE: MATLAB port of iplgeom.instantaneous_phase. Untested in MATLAB.

if nargin < 4, decim = 1; end
if nargin < 5, trimSamples = 0; end
[b, a] = butter(3, band / (fs / 2), 'bandpass');
Xf = filtfilt(b, a, double(X));
theta = angle(hilbert(Xf));        % hilbert works column-wise (per channel)
theta = theta(1:decim:end, :);
theta = theta(1 + trimSamples:end - trimSamples, :);
Z = exp(1i * theta);
end
