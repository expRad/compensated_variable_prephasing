function [FT,w] = cGrid_Cartesian(traj,imsize,w)

% Copyright (c) 2024 Hannah Scholten

    ph = 1;
    FT = NUFFT(traj,w,ph, 0,imsize, 2);

    
end

