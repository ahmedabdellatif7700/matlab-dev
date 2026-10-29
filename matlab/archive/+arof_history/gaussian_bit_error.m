function ber = gaussian_bit_error(mu,standard_deviation,reference,modulation)
%GAUSSIAN_BIT_ERROR Conditional cell probabilities, not an EVM-to-BER conversion.
% Used only as a smooth calibration surrogate; validation runs physical noise.
ideal=qammod((0:modulation-1).',modulation,'gray');
labels=de2bi((0:modulation-1).',log2(modulation),'left-msb');
if standard_deviation<=0
    actual=qamdemod(mu,modulation,'gray','OutputType','bit');
    target=qamdemod(reference,modulation,'gray','OutputType','bit');
    ber=nnz(actual~=target)/numel(target); return;
end
levels=unique(real(ideal));
edges=[-Inf;(levels(1:end-1)+levels(2:end))/2;Inf];
truth=labels(qamdemod(reference,modulation,'gray')+1,:);
ber=0;
for k=1:modulation
    i=find(levels==real(ideal(k))); q=find(levels==imag(ideal(k)));
    prob_i=.5*erfc(-(edges(i+1)-real(mu))/(standard_deviation*sqrt(2)))- ...
        .5*erfc(-(edges(i)-real(mu))/(standard_deviation*sqrt(2)));
    prob_q=.5*erfc(-(edges(q+1)-imag(mu))/(standard_deviation*sqrt(2)))- ...
        .5*erfc(-(edges(q)-imag(mu))/(standard_deviation*sqrt(2)));
    ber=ber+mean(prob_i.*prob_q.*sum(truth~=labels(k,:),2)/log2(modulation));
end
end
