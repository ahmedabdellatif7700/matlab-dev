function folder = export_datasets(cfg,opts)
%EXPORT_DATASETS Physical-link I/Q sequences and Q8.24 equalizer samples.
% X/Y are decoded Q8.24 values; integer words are also exported explicitly.
if nargin<1, cfg=link_configuration(); end
if nargin<2, opts=dataset_configuration(); end
names=string(opts.split_names); powers=opts.rop_dbm(:).';
lags=opts.tap_offsets_symbols(:).';
assert(~isempty(powers) && all(isfinite(powers)) && ...
    numel(unique(compose('rop_%+.1f_dbm',powers)))==numel(powers), ...
    'Power directory labels must be finite and distinct at 0.1 dB resolution.');
assert(isequal(names,["train" "validation" "test"]),'Expected three named splits.');
assert(isequal(size(opts.seeds),[numel(powers) 3]),'Seed matrix must match powers/splits.');
assert(all(opts.seeds(:)>=0 & opts.seeds(:)<=2^32-1-900000 & mod(opts.seeds(:),1)==0), ...
    'Seeds must be valid integer stream seeds, including their noise offset.');
assert(numel(unique(opts.seeds(:)))==numel(opts.seeds),'Split seeds must all differ.');
assert(numel(unique(opts.seeds(:)+900000))==numel(opts.seeds),'Noise seeds must differ.');
assert(isempty(intersect(opts.seeds(:),opts.seeds(:)+900000)),'Symbol/noise seeds overlap.');
assert(numel(opts.payload_symbols)==3 && all(opts.payload_symbols>0) && ...
    all(mod(opts.payload_symbols,1)==0),'Invalid split lengths.');
assert(~isempty(lags) && lags(1)==0 && all(lags<=0) && all(mod(lags,1)==0) && ...
    numel(unique(lags))==numel(lags),'Use distinct causal integer tap offsets.');
assert(cfg.training_symbols+1+min(lags)>=1,'Prefix too short for tap history.');
folder=fullfile(cfg.output_root,'equalizer_datasets',cfg.run_id);
assert(~isfolder(folder),'Refusing to overwrite exported datasets.'); mkdir(folder);
write_json(fullfile(folder,'export_configuration.json'),struct('link',cfg,'export',opts));
constellation=qammod((0:cfg.modulation_order-1).',cfg.modulation_order,'gray');
scale=sqrt(mean(abs(constellation).^2));
rows=struct([]);
for pi=1:numel(powers)
    power_dir=sprintf('rop_%+.1f_dbm',powers(pi)); mkdir(fullfile(folder,power_dir));
    for si=1:3
        c=cfg; c.seed=opts.seeds(pi,si); c.noise_seed=c.seed+900000;
        c.payload_symbols=opts.payload_symbols(si); c.drive_rms_v=0.1;
        c.received_optical_power_dbm=powers(pi);
        data=arof.generate_symbols(c);
        baseband=arof.pulse_shape(c,data.symbols);
        tx=arof.upconvert(c,data,baseband);
        c.drive_rms_v=c.drive_rms_v*c.input_vpp_v/(max(tx.if_before_driver)-min(tx.if_before_driver));
        tx=arof.upconvert(c,data,baseband);
        tx.if_voltage=arof.driver(c,tx.if_before_driver);
        optical_tx=arof.modulate(c,tx.if_voltage);
        optical_rx=arof.propagate(c,optical_tx);
        detected=arof.photodetect(c,optical_rx);
        link=arof.wireless(c,detected);
        baseband_rx=arof.downconvert(c,link);
        rx=arof.recover(c,baseband_rx,tx);
        m=arof.measure(c,optical_tx,rx);
        % Prefix and payload are retained, excluding physical burst guards.
        rx_iq=[real(rx.sampled) imag(rx.sampled)];
        tx_iq=[real(rx.reference) imag(rx.reference)];
        target_indices=rx.payload_indices(:);
        tap_indices=target_indices+lags;
        X_float=zeros(c.payload_symbols,2*numel(lags));
        for k=1:numel(lags)
            X_float(:,2*k-1:2*k)=rx_iq(tap_indices(:,k),:)/scale;
        end
        Y_float=tx_iq(target_indices,:)/scale;
        [X_q8_24,x_saturation_count]=to_q8_24(X_float);
        [Y_q8_24,y_saturation_count]=to_q8_24(Y_float);
        X=double(X_q8_24)/2^24; Y=double(Y_q8_24)/2^24;
        is_receiver_prefix=false(size(rx_iq,1),1);
        is_receiver_prefix(1:c.training_symbols)=true;
        is_equalizer_training=false(size(rx_iq,1),1);
        if names(si)=="train", is_equalizer_training(target_indices)=true; end
        metadata=struct('schema_version',1,'split',names(si),'profile',opts.profile, ...
            'received_power_dbm',powers(pi),'symbol_seed',c.seed,'noise_seed',c.noise_seed, ...
            'modulation_order',c.modulation_order,'symbol_rate_hz',c.symbol_rate_hz, ...
            'normalization_scale',scale,'normalization','Known constellation RMS; no sample-estimated scaling', ...
            'tap_offsets_symbols',lags,'feature_order','I(t),Q(t),I(t-1),Q(t-1),... in tap-offset order', ...
            'index_base',1,'fraction_bits',24,'word_bits',32, ...
            'import_rounding','Nearest ties to even, matching supplied fixedpoint.to_q', ...
            'receiver_prefix_symbols',c.training_symbols,'payload_symbols',c.payload_symbols, ...
            'timing_offset_samples',rx.timing_offset_samples, ...
            'receiver_gain_real',real(rx.calibration_gain),'receiver_gain_imag',imag(rx.calibration_gain), ...
            'unequalized_ber',m.ber,'unequalized_evm_percent',m.evm_percent, ...
            'x_saturation_count',x_saturation_count,'y_saturation_count',y_saturation_count, ...
            'input_vpp_v',max(tx.if_before_driver)-min(tx.if_before_driver), ...
            'mzm_vpp_v',m.drive_vpp_v,'driver_gain',c.driver_voltage_gain,'if_noise_psd',c.if_noise_psd);
        configuration=c;
        relative_file=fullfile(power_dir,names(si)+".mat");
        save(fullfile(folder,relative_file),'X','Y','X_q8_24','Y_q8_24', ...
            'rx_iq','tx_iq','target_indices','is_receiver_prefix','is_equalizer_training', ...
            'metadata','configuration','-v7');
        rows=[rows; struct('file',string(relative_file),'split',names(si),'received_power_dbm',powers(pi), ...
            'symbol_seed',c.seed,'noise_seed',c.noise_seed,'samples',size(X,1),'inputs',size(X,2), ...
            'outputs',size(Y,2),'bit_errors',m.bit_errors,'bits',m.bits,'unequalized_ber',m.ber, ...
            'unequalized_evm_percent',m.evm_percent,'x_saturation_count',x_saturation_count, ...
            'y_saturation_count',y_saturation_count)]; %#ok<AGROW>
        writetable(struct2table(rows),fullfile(folder,'manifest.csv'));
        fprintf('Exported %s: %d samples, seed %d.\n',relative_file,size(X,1),c.seed);
    end
end
fprintf('Equalizer datasets: %s\n',folder);
end

function [words,saturations] = to_q8_24(values)
assert(all(isfinite(values(:))),'Nonfinite dataset value.');
% The supplied Python to_q uses round(), i.e. ties to even at import.
% Its INTERNAL round_shift uses ties away from zero; that is unchanged.
scaled=values*2^24; lower=floor(scaled); fraction=scaled-lower;
rounded=lower+(fraction>0.5 | (fraction==0.5 & mod(lower,2)~=0));
saturations=nnz(rounded < -2^31 | rounded > 2^31-1);
words=int32(min(max(rounded,-2^31),2^31-1));
end

function write_json(path,value)
[fid,msg]=fopen(path,'w'); assert(fid>=0,'%s',msg);
cleanup=onCleanup(@() fclose(fid)); fprintf(fid,'%s\n',jsonencode(value,'PrettyPrint',true));
end
