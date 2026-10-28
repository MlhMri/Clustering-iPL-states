# MATLAB port 

Pipeline, following Sec. 2 of the paper:

```matlab
[~, theta] = bw_phases(data_rs1(:,:,s), 160, [8 13], 4, 40);   % T' x 64 phases for subject s
[Zw, L]    = bw_weighted_eigenvectors(theta);                  % [sqrt(l1) u1', sqrt(l2) u2'] per time point
[labels, M] = bw_kmeans(Zw, 4, 10);                            % rank-two Bures-Wasserstein K-means
```

`Zw` has the same layout as `Z` in the weighted-Procrustes scripts, so the K-medoids code can be
swapped for `bw_kmeans` directly. Before relying on the port, compare it with the Python
implementation (`../iplgeom.py`, method `'bw'`) on a small example.
