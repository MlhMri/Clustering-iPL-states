function [Zw, L] = bw_weighted_eigenvectors(theta)
% BW_WEIGHTED_EIGENVECTORS  Eigenvalue-weighted eigenvectors of the iPL matrices (Sec. 2.2).
%
%   [Zw, L] = bw_weighted_eigenvectors(theta)
%
%   theta  T x P instantaneous phases (e.g. from bw_phases.m).
%   Zw     T x 2P, Zw(t,:) = [sqrt(l1)*u1', sqrt(l2)*u2'] of P_t = cos(theta_j - theta_k),
%          i.e. the same layout as Z in the weighted-Procrustes scripts.
%   L      T x 2 eigenvalues [l1, l2], with l1 + l2 = P.
%
%   Closed form (no eigendecomposition): with psi = angle(sum(exp(2i*theta))), the rows of
%   [sqrt(l1)*u1, sqrt(l2)*u2] are the channel phasors relative to the axial mean phase:
%   [cos(theta - psi/2), sin(theta - psi/2)]. Each row has unit length.
%   NOTE: MATLAB port of iplgeom.ipl_eig. Untested in MATLAB; compare with svd([c, s]) once.

psi = angle(sum(exp(2i * theta), 2));          % T x 1
rel = theta - psi / 2;                         % phases relative to the axial mean phase
Zw = [cos(rel), sin(rel)];
P = size(theta, 2);
L = [sum(cos(rel).^2, 2), sum(sin(rel).^2, 2)];
% order so that the first column carries the larger eigenvalue (always true by construction)
assert(all(L(:, 1) >= L(:, 2) - 1e-9) && all(abs(sum(L, 2) - P) < 1e-6));
end
