function rx = receive(cfg,detected,tx)
%RECEIVE Batch convenience; main.m exposes each receiver stage.
link = arof_history.wireless(cfg,detected);
detector = arof_history.downconvert(cfg,link);
rx = arof_history.recover(cfg,detector,tx);
end
