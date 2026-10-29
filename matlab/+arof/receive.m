function rx = receive(cfg,detected,tx)
%RECEIVE Batch convenience; main.m exposes each receiver stage.
link = arof.wireless(cfg,detected);
detector = arof.downconvert(cfg,link);
rx = arof.recover(cfg,detector,tx);
end
