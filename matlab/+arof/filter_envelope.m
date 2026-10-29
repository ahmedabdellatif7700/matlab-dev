function y = filter_envelope(x,fs,guard,response)
%FILTER_ENVELOPE Zero-padded FFT filtering on the declared frequency grid.
x = x(:); nfft = 2^nextpow2(numel(x)+2*guard);
f = (-nfft/2:nfft/2-1).'*fs/nfft;
padded = zeros(nfft,1,"like",complex(x));
padded(guard+(1:numel(x))) = x;
filtered = ifft(fft(padded).*ifftshift(response(f)));
y = filtered(guard+(1:numel(x)));
end
