function [FT,w] = cGrid_Cartesian(traj,imsize,w)


    ph = 1;
    FT = NUFFT(traj,w,ph, 0,imsize, 2);

    
end

