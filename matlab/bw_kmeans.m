function [labels, M, obj] = bw_kmeans(Zw, K, nInit, maxIter)
% BW_KMEANS  K-means with the rank-two Bures-Wasserstein distance (Sec. 2.4-2.5 of the paper).
%
%   [labels, M, obj] = bw_kmeans(Zw, K, nInit, maxIter)
%
%   Zw     T x 2P eigenvalue-weighted eigenvectors, Zw(t,:) = [sqrt(l1)*u1', sqrt(l2)*u2']
%          (see bw_weighted_eigenvectors.m; the same layout as the weighted-Procrustes Z).
%   M      P x 2 x K centroids. The rows of M(:,:,k) are average phasors: their angles give the
%          relative phase of each channel in state k, their lengths its phase consistency.
%   obj    mean squared BW distance to the assigned centroid.
%
%   Distance between sample t and centroid M (2x2 cross-product C = Ut' * M):
%     d^2 = ||Ut||_F^2 + ||M||_F^2 - 2*(s1 + s2),   (s1 + s2)^2 = ||C||_F^2 + 2*|det C|
%   which equals min over orthogonal Omega of ||M - Ut*Omega||_F^2 (orthogonal Procrustes).
%   Centroid update: generalized Procrustes analysis (align members to M, then average).
%   NOTE: MATLAB port of code/iplgeom.py. Untested in MATLAB; check against the Python
%   implementation on a small example before use. Requires pagesvd (R2021b or later).

if nargin < 3, nInit = 10; end
if nargin < 4, maxIter = 300; end
best = inf;
for rep = 1:nInit
    M = plusplus(Zw, K);
    prevLab = []; prevObj = inf;
    for it = 1:maxIter
        D = bw_sqdist(Zw, M);
        [dmin, lab] = min(D, [], 2);
        o = mean(dmin);
        if ~isempty(prevLab) && isequal(lab, prevLab) && prevObj - o <= 1e-10 * abs(prevObj)
            break
        end
        for k = 1:K
            members = Zw(lab == k, :);
            if isempty(members)                          % re-seed an empty state
                [~, worst] = max(dmin);
                M(:, :, k) = reshape(Zw(worst, :), [], 2);
            else
                M(:, :, k) = procrustes_mean(members, M(:, :, k));
            end
        end
        prevLab = lab; prevObj = o;
    end
    D = bw_sqdist(Zw, M);
    [dmin, lab] = min(D, [], 2);
    if mean(dmin) < best
        best = mean(dmin); labels = lab; Mbest = M;
    end
end
M = Mbest; obj = best;
end

function D = bw_sqdist(Zw, M)
% Squared BW distance between every row of Zw (T x 2P) and every centroid M(:,:,k) (P x 2).
P = size(Zw, 2) / 2;
U1 = Zw(:, 1:P); U2 = Zw(:, P+1:end);
K = size(M, 3);
D = zeros(size(Zw, 1), K);
nz = sum(Zw.^2, 2);
for k = 1:K
    a = U1 * M(:, 1, k); b = U1 * M(:, 2, k);
    c = U2 * M(:, 1, k); d = U2 * M(:, 2, k);
    nuc = sqrt(a.^2 + b.^2 + c.^2 + d.^2 + 2 * abs(a .* d - b .* c));   % s1 + s2 of C = Ut' * M
    D(:, k) = max(nz + sum(M(:, :, k).^2, 'all') - 2 * nuc, 0);
end
end

function M = procrustes_mean(members, M)
% A few sweeps of generalized Procrustes analysis: rotate/mirror every member onto M, average.
n = size(members, 1); P = size(members, 2) / 2;
Ut = reshape(members.', P, 2, n);                   % P x 2 x n weighted eigenvector pairs
for sweep = 1:3
    C = pagemtimes(Ut, 'transpose', M, 'none');       % 2 x 2 x n,  C_t = Ut_t' * M = Q S R'
    [Q, ~, R] = pagesvd(C);
    Omega = pagemtimes(Q, 'none', R, 'transpose');    % aligns Ut_t to M:  Ut_t * Omega_t ~ M
    M = mean(pagemtimes(Ut, Omega), 3);
end
end

function M = plusplus(Zw, K)
T = size(Zw, 1); P = size(Zw, 2) / 2;
idx = randi(T);
M = reshape(Zw(idx, :), P, 2);
dmin = bw_sqdist(Zw, M);
for k = 2:K
    p = dmin / sum(dmin);
    idx(k) = find(rand <= cumsum(p), 1, 'first');
    M(:, :, k) = reshape(Zw(idx(k), :), P, 2);
    dmin = min(dmin, bw_sqdist(Zw, M(:, :, k)));
end
end
