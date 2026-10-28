function [indices, signed_bins] = multiuser_bins(cfg, guard)
%MULTIUSER_BINS Four nonoverlapping localized DFT-spread allocations.
width = cfg.multi_users*cfg.multi_dft+(cfg.multi_users-1)*guard;
assert(width < cfg.multi_fft, "Allocation exceeds FFT.");
signed_bins = zeros(cfg.multi_dft, cfg.multi_users);
for user = 1:cfg.multi_users
    signed_bins(:, user) = -floor(width/2)+(user-1)*(cfg.multi_dft+guard)+(0:cfg.multi_dft-1).';
end
indices = mod(signed_bins, cfg.multi_fft)+1;
end
