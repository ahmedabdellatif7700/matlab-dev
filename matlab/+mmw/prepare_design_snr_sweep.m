function prepare_design_snr_sweep(cfg, cases, run_dir)
%PREPARE_DESIGN_SNR_SWEEP Fresh payloads; frozen trained receivers, no tuning.
for seed = cfg.confirmation_seeds
    for c = 1:height(cases)
        base = table2struct(cases(c,:));
        for offset = [-6 -3 0 3 6]
            condition = base; condition.Reference_SNR_dB = base.Reference_SNR_dB+offset;
            [rx,tx] = mmw.design_burst(cfg,condition,cfg.confirmation_counts(3),seed+900000);
            keep = (1-min(cfg.tap_offsets):numel(rx)-max(cfg.tap_offsets)).';
            X = zeros(numel(keep),2*numel(cfg.tap_offsets),"single");
            for tap = 1:numel(cfg.tap_offsets)
                value = rx(keep+cfg.tap_offsets(tap));
                X(:,2*tap-1) = single(real(value)); X(:,2*tap) = single(imag(value));
            end
            Y = single([real(tx(keep)) imag(tx(keep))]);
            reference_snr_db = condition.Reference_SNR_dB;
            folder = fullfile(run_dir,"snr_sweep",sprintf("seed_%d",seed),condition.ID);
            if ~isfolder(folder), mkdir(folder); end
            path = fullfile(folder,sprintf("offset_%+03d.mat",offset));
            assert(~isfile(path),"Refusing to overwrite sweep data.");
            save(path,"X","Y","reference_snr_db","-v7");
        end
    end
end
end
