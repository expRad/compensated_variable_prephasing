clear all;
% Change current directory to that of this .m file
mfile_name          = mfilename('fullpath');
if contains(mfile_name,'LiveEditorEvaluationHelper')
    mfile_name = matlab.desktop.editor.getActiveFilename;
end
[pathstr,name,ext]  = fileparts(mfile_name);
cd(pathstr);
% Add necessary scripts to the MATLAB path
folder1 = '.\EPI_Recon_NUFFT\';
folder2 = '.\GSTF_code\';
addpath(genpath(folder1), folder2);
% Run setup for MIRT toolbox
run ('.\MIRT_toolbox\irt\setup.m');


%% DEFINE DATA PATHS
path2save = '.\EPI_results\';

% Transverse images %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
name2load_meas = '.\EPI_data\EPI_tra_peAP_FOVph80_sliceISO.mat';
name2load_sim = '.\gradient_data\sim_EPI_tra_peAP_FOVph80.mat';
name2load_traj1 = '.\gradient_data\GradMeas_FCVP_EPI_iso13FOVph80_x_10Dum_10VP_9sl_CCoff.mat';
name2load_traj2 = '.\gradient_data\GradMeas_FCVP_EPI_iso13FOVph80_x_10Dum_10VP_9sl_CCon.mat';
nMeas = 20;
nMeas_sim = 1;
B0_correction = 0;
PE_dir_90deg = 0; % Is the PE direction turned by 90°, i.e. for transversal slices R->L instead of A->P?
nom_delay = 2.09e-6;
girf_delay = -0.98e-6;
meas_delay = 7.28e-6;
name2save = 'results_EPI_tra_peAP_FOVph80_new.mat';
% % Delay too small %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% nom_delay = 1e-6;
% girf_delay = -2e-6;
% meas_delay = 6e-6;
% name2save = 'results_EPI_tra_peAP_FOVph80_tooSmallDelay.mat';
% % Delay too large %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% nom_delay = 3e-6;
% girf_delay = 0e-6;
% meas_delay = 8e-6;
% name2save = 'results_EPI_tra_peAP_FOVph80_tooLargeDelay.mat';

%% DEFINE SOME PARAMETERS OF THE ACQUISITION AND FOR THE RECONSTRUCTION
% Acquisition parameters
nAvg = 1;
slice2plot = 1;
cutoff = 6700;

% Reconstruction parameters
SoS = 1; % Do Sum-Of-Squares coil combination?
coil_i = 10; % If SoS==0, the image of this coil element will be stored in the results.

%% LOAD GSTFs
GIRF_x_data = load('.\GSTF_data\Hx_fast_FT0_1030.mat');
GIRF_y_data = load('.\GSTF_data\Hy_fast_FT0_1030.mat');
GIRF_z_data = load('.\GSTF_data\Hz_fast_FT0_1030.mat');

%% READ RAW DATA
disp('Read raw data');
rawdata = load(name2load_meas);
dwelltime = rawdata.dwelltime; % dwelltime in seconds
orientation = rawdata.orientation;
kspace = rawdata.kspace;
% Order of raw data:
%  1) Columns
%  2) Channels/Coils
%  3) Lines
%  4) Partitions
%  5) Slices
%  6) Averages
%  7) (Cardiac-) Phases
%  8) Contrasts/Echoes
%  9) Measurements -> contains b-values, diff.dir. & averages
% 10) Sets
% 11) Segments
nRO = size(kspace,1);
nPE = rawdata.nPE; % Actually measured PE lines
nImageLines = rawdata.nImageLines; % Takes into account reduced FOV in PE-direction
nSlices = size(kspace,5);
nCoils = size(kspace,2);

% Also load navigator data
navi = rawdata.navi;
navi = mean(navi, 6); % Carry out average (only segment 1 contains data to be averaged)
% The actual data for the phase correction seems to be in the last line
navi_orig = navi(:,:,end,:,:,:,:,:,:,:,:); % [Columns, Coils, ... , Segments]

% Adjust some dimensions, so that squeezing will always give the same resulting number of dimensions
if nSlices==1
    kspace = repmat(kspace, [1,1,1,1,2,1,1,1,1,1,1]); % pretend there are 2 slices
    navi_orig = repmat(navi_orig, [1,1,1,1,2,1,1,1,1,1,1]);
    nSlices = 2;
end
if size(kspace,9)==1
    kspace = repmat(kspace, [1,1,1,1,1,1,1,1,2,1,1]); % pretend there are 2 measurements
    navi_orig = repmat(navi_orig, [1,1,1,1,1,1,1,1,2,1,1]);
    nMeas = 2;
end
if nCoils==1
    kspace = repmat(kspace, [1,2,1,1,1,1,1,1,1,1,1]); % pretend there are 2 coils
    navi_orig = repmat(navi_orig, [1,2,1,1,1,1,1,1,1,1,1]);
    % Don't change nCoils here! Will be needed later.
end

% Ramp sampling trajectory for the navigator phase correction
navi_traj = rawdata.navi_traj;

%% LOAD NOMINAL GRADIENT WAVEFORMS
disp('Load sequence simulation');
simdata = load(name2load_sim);

adc = simdata.adc; % [TR_us,numRep]
grad_x = simdata.grad_x; % T/m
grad_y = simdata.grad_y;
grad_z = simdata.grad_z;
clearvars simdata;

%% SELECT RELEVANT REPETITIONS AND ADJUST TIME WINDOW
if size(adc,2)>1
    adc = adc(:,end-nMeas_sim+1:end); % first reps are without acquisition
    grad_x = grad_x(:,end-nMeas_sim+1:end);
    grad_y = grad_y(:,end-nMeas_sim+1:end);
    grad_z = grad_z(:,end-nMeas_sim+1:end);
end

% Find the start of the first gradient
startidx = min([find(grad_x~=0,1), find(grad_y~=0,1), find(grad_z~=0,1)]);
% Start all arrays 10 datapoints (= 1 GRT) before that
adc = adc(startidx-10:end,:);
grad_x = grad_x(startidx-10:end,:);
grad_y = grad_y(startidx-10:end,:);
grad_z = grad_z(startidx-10:end,:);

% Make sure the array length stays a multiple of 10, i.e. of the GRT
if mod(size(adc,1),10)~=0
    grad_x = grad_x(1:end-mod(size(adc,1),10),:);
    grad_y = grad_y(1:end-mod(size(adc,1),10),:);
    grad_z = grad_z(1:end-mod(size(adc,1),10),:);
    adc = adc(1:end-mod(size(adc,1),10),:);
end

%% CHECK INDICES
if 0
    % Plot to check the indices
    figure;
    plot(adc, 'DisplayName','ADC');
    hold on;
    plot(grad_x, '--', 'DisplayName','Grad X');
    plot(grad_y, '-.', 'DisplayName','Grad Y');
    plot(grad_z, ':', 'DisplayName','Grad Z');
    legend;
end
%% DELETE UNNECESSARY PARTS
adc(1:cutoff,:) = 0; % Delete navigator ADC

%% CALCULATE NOMINAL K-SPACE TRAJECTORY
disp('Calculate nominal k-space trajectory');

t_sim = (0:1:size(grad_x,1)-1)*1e-6;
t_axis_ADC = (0:1:nRO-1)*dwelltime + dwelltime/2;

[kRO_nom, kPE_nom, ~, gradRO_nom] = calc_traj(grad_x, grad_y, grad_z, zeros(size(grad_x)), t_sim, t_axis_ADC, adc, nRO, nPE, orientation, PE_dir_90deg);

f_axis_sim = linspace(-500000, 500000, size(grad_z,1));
Gx_sim = zeros(size(grad_x));
Gy_sim = zeros(size(grad_x));
Gz_sim = zeros(size(grad_x));
for meas=1:size(grad_x,2)
    G = fft_1D(grad_x(:,meas),1) .* exp(1i*2*pi*f_axis_sim.' *nom_delay);
    Gx_sim(:,meas) = real(ifft_1D(G,1));
    G = fft_1D(grad_y(:,meas),1) .* exp(1i*2*pi*f_axis_sim.' *nom_delay);
    Gy_sim(:,meas) = real(ifft_1D(G,1));
    G = fft_1D(grad_z(:,meas),1) .* exp(1i*2*pi*f_axis_sim.' *nom_delay);
    Gz_sim(:,meas) = real(ifft_1D(G,1));
end

[kRO_del, kPE_del, ~, ~] = calc_traj(Gx_sim, Gy_sim, Gz_sim, zeros(size(Gx_sim)), t_sim, t_axis_ADC, adc, nRO, nPE, orientation, PE_dir_90deg);

%% ZERO-FILLING
% Append zeros to avoid side effects by the fft calculation
nExtra = (1e6-size(grad_x,1))/2;
grad_x = [zeros(nExtra,size(grad_x,2)); grad_x; zeros(nExtra,size(grad_x,2))];
grad_y = [zeros(nExtra,size(grad_x,2)); grad_y; zeros(nExtra,size(grad_x,2))];
grad_z = [zeros(nExtra,size(grad_x,2)); grad_z; zeros(nExtra,size(grad_x,2))];
f_axis_simextra = linspace(-500000, 500000, size(grad_z,1));

%% GET GSTFs
disp('Prepare GSTFs');
H_x = GIRF_x_data.H_combined;
H_y = GIRF_y_data.H_combined;
H_z = GIRF_z_data.H_combined;

%% INTERPOLATE GSTFs TO GRADIENT f-axis
H_x.gstf(:,2) = H_x.gstf(:,2)./mean(abs(H_x.gstf(ceil(end/2)-10:ceil(end/2)+10,2)),1);
H_y.gstf(:,2) = H_y.gstf(:,2)./mean(abs(H_y.gstf(ceil(end/2)-10:ceil(end/2)+10,2)),1);
H_z.gstf(:,2) = H_z.gstf(:,2)./mean(abs(H_z.gstf(ceil(end/2)-10:ceil(end/2)+10,2)),1);

gstf_x = interp1(H_x.f_axis, H_x.gstf(:,2), f_axis_simextra, 'makima',0).';
gstf_y = interp1(H_y.f_axis, H_y.gstf(:,2), f_axis_simextra, 'makima',0).';
gstf_z = interp1(H_z.f_axis, H_z.gstf(:,2), f_axis_simextra, 'makima',0).';
gstf_x_B0 = interp1(H_x.f_axis, H_x.gstf(:,1), f_axis_simextra, 'makima',0).';
gstf_y_B0 = interp1(H_y.f_axis, H_y.gstf(:,1), f_axis_simextra, 'makima',0).';
gstf_z_B0 = interp1(H_z.f_axis, H_z.gstf(:,1), f_axis_simextra, 'makima',0).';

%% GRADIENT-WAVEFORM CALCULATION BASED ON GSTFs
disp('GIRF corrections');
Gx = zeros(size(grad_x));
Gy = zeros(size(grad_x));
Gz = zeros(size(grad_x));
B0 = zeros(size(grad_x));

for meas=1:size(grad_x,2)
    G = fft_1D(grad_x(:,meas),1) .* gstf_x .* exp(1i*2*pi*f_axis_simextra.' *girf_delay);
    Gx(:,meas) = real(ifft_1D(G,1)) + H_x.fieldOffsets(2);
    G = fft_1D(grad_y(:,meas),1) .* gstf_y .* exp(1i*2*pi*f_axis_simextra.' *girf_delay);
    Gy(:,meas) = real(ifft_1D(G,1)) + H_y.fieldOffsets(2);
    G = fft_1D(grad_z(:,meas),1) .* gstf_z .* exp(1i*2*pi*f_axis_simextra.' *girf_delay);
    Gz(:,meas) = real(ifft_1D(G,1)) + H_z.fieldOffsets(2);
    
    B = fft_1D(grad_x(:,meas),1) .* gstf_x_B0 .* exp(1i*2*pi*f_axis_simextra.' *girf_delay);
    B0(:,meas) = B0(:,meas) + real(ifft_1D(B,1)) + H_x.fieldOffsets(1);
    B = fft_1D(grad_y(:,meas),1) .* gstf_y_B0 .* exp(1i*2*pi*f_axis_simextra.' *girf_delay);
    B0(:,meas) = B0(:,meas) + real(ifft_1D(B,1)) + H_y.fieldOffsets(1);
    B = fft_1D(grad_z(:,meas),1) .* gstf_z_B0 .* exp(1i*2*pi*f_axis_simextra.' *girf_delay);
    B0(:,meas) = B0(:,meas) + real(ifft_1D(B,1)) + H_z.fieldOffsets(1);
end
Gx = Gx(nExtra+1:end-nExtra,:);
Gy = Gy(nExtra+1:end-nExtra,:);
Gz = Gz(nExtra+1:end-nExtra,:);
B0 = B0(nExtra+1:end-nExtra,:);

clearvars G B;

%% TRAJECTORY CALCULATIONS

[kRO_girf, kPE_girf, B0phase_girf, ~] = calc_traj(Gx, Gy, Gz, B0, t_sim, t_axis_ADC, adc, nRO, nPE, orientation, PE_dir_90deg);

%% MEASURED TRAJECTORY

[ grad_traj_CCoff, dwelltime_traj ] = calcMeasOutput_FCVP_selectSlices_mat(name2load_traj1, 1, 1, 1, 2, 0, [], 10, 0, []);
[ grad_traj_VP, dwelltime_traj ] = calcMeasOutput_VP_selectSlices_mat(name2load_traj1, 1, 1, 1, 2, 0, [], 10, 0, []);
[ grad_traj_CCon, dwelltime_traj ] = calcMeasOutput_FCVP_selectSlices_mat(name2load_traj2, 1, 1, 1, 2, 0, [], 10, 1, []);
if strcmp(orientation, 'dCor')
    grad_traj_CCoff = -grad_traj_CCoff;
    grad_traj_VP = -grad_traj_VP;
    grad_traj_CCon = -grad_traj_CCon;
end

[kRO_meas_CCoff, B0phase_meas_CCoff] = calc_kRO(grad_traj_CCoff, dwelltime_traj, meas_delay, t_axis_ADC, t_sim, adc, nRO, nPE);
[kRO_meas_VP, B0phase_meas_VP] = calc_kRO(grad_traj_VP, dwelltime_traj, meas_delay, t_axis_ADC, t_sim, adc, nRO, nPE);
[kRO_meas_CCon, B0phase_meas_CCon] = calc_kRO(grad_traj_CCon, dwelltime_traj, meas_delay, t_axis_ADC, t_sim, adc, nRO, nPE);

%% PREPARE RAW DATA
disp('Raw data preparation')

navi = squeeze(navi_orig); % [Columns, Coils, Slices, Measurements, Segments]

%% NAVIGATOR CORRECTION FOR NOMINAL RECONSTRUCTION
disp('Navigator correction')
% Correction is performed in the hybrid x-ky-space
raw_nom_orig = squeeze(kspace); % [Columns, Coils, Lines, Slices, Measurements, Segments]
if 1
    navi = squeeze(navi); % [Columns, Coils, Slices, Measurements, Segments]
    projections = fft_1D(navi,1); % FFT along readout direction
    % Combine correction data from different (head) coil channels
    combinedProjections = zeros(size(projections,1),size(projections,3),size(projections,4)); % [Columns, Slices, Measurements]
    for slice=1:size(projections,3)
        for meas=1:size(projections,4)
            for coil=1:size(projections,2)
                tmp = squeeze(projections(:,coil,slice,meas,1)) .* conj(squeeze(projections(:,coil,slice,meas,2)));
                combinedProjections(:,slice,meas) = combinedProjections(:,slice,meas) + tmp;
            end
        end
    end
    phi = angle(combinedProjections);
    phi = unwrap(phi, [],1);
    phi_slope = zeros(nSlices,nMeas*nAvg);
    phi_const = zeros(nSlices,nMeas*nAvg);
    x = (1:1:size(projections,1)).'-size(projections,1)/2;
    for slice=1:nSlices
        absProj = abs(combinedProjections(:,slice,1));
        for meas=1:nMeas*nAvg
            phi2fit = squeeze(phi(:,slice,meas));
            phasefit = fit(x(absProj>0.0005),phi2fit(absProj>0.0005),'poly1');
            phasefitparams = coeffvalues(phasefit);
            phi_slope(slice,meas) = phasefitparams(1);
            phi_const(slice,meas) = phasefitparams(2);
        end
    end
    % Comment in to plot linear fit
    % figure;plot(x,phi2fit);hold on;plot(x,x.*phasefitparams(1)+phasefitparams(2));title('navigator phase fit');
    phi_slope = repmat(phi_slope, [1,1,nRO,nPE,size(projections,2)]);
    phi_slope = permute(phi_slope, [3,5,4,1,2]); % [Columns, Coils, Lines, Slices, Measurements]
    phi_const = repmat(phi_const, [1,1,nRO,nPE,size(projections,2)]);
    phi_const = permute(phi_const, [3,5,4,1,2]); % [Columns, Coils, Lines, Slices, Measurements]

    kspace_corr = fft_1D(raw_nom_orig, 1); % FFT along readout direction, [Columns, Coils, Lines, Slices, Measurements,Segments]
    % Correct the phase in the raw data
    x = (1:1:size(projections,1)).'-size(projections,1)/2;
    x = repmat(x, [1,size(kspace_corr,2),size(kspace_corr,3),size(kspace_corr,4),size(kspace_corr,5)]); % [Columns, Coils, Lines, Slices, Measurements]
    
    kspace_corr(:,:,:,:,:,2) = squeeze(kspace_corr(:,:,:,:,:,2)).*exp(1i*(x.*phi_slope+phi_const));

    kspace_corr = ifft_1D(kspace_corr,1); % iFFT along readout direction, so we are back in k-space
    raw_navicorr = kspace_corr;    
else
    raw_navicorr = raw_nom_orig;
end

% Trajectory for navigator-corrected reconstruction
kRO_navi = repmat(navi_traj, [1,nImageLines])./max(abs(navi_traj));
kPE_navi = linspace(1, -1, nImageLines);
kPE_navi = repmat(kPE_navi, [nRO,1]);
traj_navi = kRO_navi + 1i*kPE_navi;

%% RECONSTRUCTIONS WITH NUFFT
disp('Reconstruction (NUFFT)')

max_kRO_nom = max(abs(kRO_del),[],'all');
max_kPE_nom = max(abs(kPE_del),[],'all');
max_kRO_girf = max(abs(kRO_girf),[],'all');
max_kPE_girf = max(abs(kPE_girf),[],'all');
max_kRO_meas = max(abs(kRO_meas_CCoff),[],'all');
max_kRO_meas_VP = max(abs(kRO_meas_VP),[],'all');
max_kRO = max(max_kRO_nom, max(max_kRO_girf, max(max_kRO_meas, max_kRO_meas_VP)));
max_kPE = max(max_kPE_nom, max_kPE_girf);
max_k = max(max_kRO, max_kPE);

traj_nom = (kRO_del + 1i*kPE_del)/max_k;
traj_girf = (kRO_girf + 1i*kPE_girf)/max_k;
traj_meas_CCoff = (kRO_meas_CCoff + 1i*kPE_nom)/max_k;
traj_meas_VP = (kRO_meas_VP + 1i*kPE_nom)/max_k;
traj_meas_CCon = (kRO_meas_CCon + 1i*kPE_nom)/max_k;
traj_factor = 1.9;
matrix_size = [nRO nImageLines];
dc = abs(gradRO_nom)/max(abs(gradRO_nom), [],'all'); % density compensation
Hp = repmat(hamming(30,'symmetric'), [1,size(dc,2)]);
dc_new = conv(dc(:,1), Hp(:,1), 'same');
dc_new = dc_new/max(dc_new)/2;
dc = repmat(dc_new, [1,size(dc,2)]);

% Add odd and even segments
raw = squeeze(kspace); % [Columns, Coils, Lines, Slices, Measurements, Segments]
raw = squeeze(raw(:,:,:,:,:,1) + raw(end:-1:1,:,:,:,:,2)); % [Columns, Coils, Lines, Slices, Measurements]

% For navigator-Recon (trajectory goes only left to right):
raw_navi = squeeze(raw_navicorr(:,:,:,:,:,1) + raw_navicorr(:,:,:,:,:,2)); % [Columns, Coils, Lines, Slices, Measurements]

% Only take first average
raw = raw(:,:,:,:,1:nMeas);
raw_navi = raw_navi(:,:,:,:,1:nMeas);

raw_girf = raw;
raw_meas_CCoff = raw;
raw_meas_VP = raw;
raw_meas_CCon = raw;
% B0 phase correction
if B0_correction
    for coil = 1:nCoils
        for slice = 1:nSlices
            for meas = 1:nMeas*nAvg
                raw_girf(:,coil,:,slice,meas) = squeeze(raw(:,coil,:,slice,meas)) .* squeeze(exp(-1i*B0phase_girf(:,:,meas)));
                raw_meas_CCoff(:,coil,:,slice,meas) = squeeze(raw(:,coil,:,slice,meas)) .* exp(-1i*B0phase_meas_CCoff);
                raw_meas_VP(:,coil,:,slice,meas) = squeeze(raw(:,coil,:,slice,meas)) .* exp(-1i*B0phase_meas_VP);
                raw_meas_CCon(:,coil,:,slice,meas) = squeeze(raw(:,coil,:,slice,meas)) .* exp(-1i*B0phase_meas_CCon);
            end
        end
    end
end

% Do zero-padding for partial-Fourier-reconstruction
nPad = nImageLines - size(raw,3);
raw_filled = cat(3, zeros(size(raw(:,:,1:nPad,:,:))), raw);
raw_girf_filled = cat(3, zeros(size(raw(:,:,1:nPad,:,:))), raw_girf);
raw_meas_CCoff_filled = cat(3, zeros(size(raw(:,:,1:nPad,:,:))), raw_meas_CCoff);
raw_meas_VP_filled = cat(3, zeros(size(raw(:,:,1:nPad,:,:))), raw_meas_VP);
raw_meas_CCon_filled = cat(3, zeros(size(raw(:,:,1:nPad,:,:))), raw_meas_CCon);
raw_navi_filled = cat(3, zeros(size(raw(:,:,1:nPad,:,:))), raw_navi);

d_kPE = (kPE_del(1,1) - kPE_del(1,2))/max_k;
traj_fill = traj_nom(:,1:nPad) + traj_nom(end/2,1) - traj_nom(end/2,nPad) + 1i*d_kPE;
traj_nom_filled = cat(2,traj_fill,traj_nom);
traj_girf_filled = cat(2,traj_fill,traj_girf);
traj_meas_CCoff_filled = cat(2,traj_fill,traj_meas_CCoff);
traj_meas_VP_filled = cat(2,traj_fill,traj_meas_VP);
traj_meas_CCon_filled = cat(2,traj_fill,traj_meas_CCon);
dc_filled = cat(2, dc(:,1:nPad),dc);

FT_nom_dc = cGrid_Cartesian(traj_nom_filled/traj_factor,matrix_size,dc_filled);
FT_girf_dc = cGrid_Cartesian(traj_girf_filled/traj_factor,matrix_size,dc_filled);
FT_meas_CCoff_dc = cGrid_Cartesian(traj_meas_CCoff_filled/traj_factor,matrix_size,dc_filled);
FT_meas_VP_dc = cGrid_Cartesian(traj_meas_VP_filled/traj_factor,matrix_size,dc_filled);
FT_meas_CCon_dc = cGrid_Cartesian(traj_meas_CCon_filled/traj_factor,matrix_size,dc_filled);
FT_navi_dc = cGrid_Cartesian(traj_navi/traj_factor,matrix_size,dc_filled);
%%
reco_nom_dc = zeros(nRO, nImageLines, nCoils, nSlices, nMeas);
reco_navi_dc = zeros(size(reco_nom_dc));
reco_girf_dc = zeros(size(reco_nom_dc));
reco_meas_CCoff_dc = zeros(size(reco_nom_dc));
reco_meas_VP_dc = zeros(size(reco_nom_dc));
reco_meas_CCon_dc = zeros(size(reco_nom_dc));

for coil = 1:nCoils
    for slice = 1:nSlices
        for meas = 1:nMeas*nAvg
            reco_navi_dc(:,:,coil,slice,meas) = FT_navi_dc'*squeeze(raw_navi_filled(:,coil,:,slice,meas));
            reco_nom_dc(:,:,coil,slice,meas) = FT_nom_dc'*squeeze(raw_filled(:,coil,:,slice,meas));
            reco_girf_dc(:,:,coil,slice,meas) = FT_girf_dc'*squeeze(raw_girf_filled(:,coil,:,slice,meas));
            reco_meas_CCoff_dc(:,:,coil,slice,meas) = FT_meas_CCoff_dc'*squeeze(raw_meas_CCoff_filled(:,coil,:,slice,meas));
            reco_meas_VP_dc(:,:,coil,slice,meas) = FT_meas_VP_dc'*squeeze(raw_meas_VP_filled(:,coil,:,slice,meas));
            reco_meas_CCon_dc(:,:,coil,slice,meas) = FT_meas_CCon_dc'*squeeze(raw_meas_CCon_filled(:,coil,:,slice,meas));
        end
    end
end

%% COIL COMBINATION
disp('Coil combination')

if nCoils == 1
    sum_images_nom_dc = squeeze(reco_nom_dc(:,:,1,:,:));
    sum_images_navi_dc = squeeze(reco_navi_dc(:,:,1,:,:));
    sum_images_girf_dc = squeeze(reco_girf_dc(:,:,1,:,:));
    sum_images_meas_CCoff_dc = squeeze(reco_meas_CCoff_dc(:,:,1,:,:));
    sum_images_meas_VP_dc = squeeze(reco_meas_VP_dc(:,:,1,:,:));
    sum_images_meas_CCon_dc = squeeze(reco_meas_CCon_dc(:,:,1,:,:));
elseif SoS
    sum_images_nom_dc = squeeze(sqrt(sum(reco_nom_dc.*conj(reco_nom_dc), 3)));
    sum_images_navi_dc = squeeze(sqrt(sum(reco_navi_dc.*conj(reco_navi_dc), 3)));
    sum_images_girf_dc = squeeze(sqrt(sum(reco_girf_dc.*conj(reco_girf_dc), 3)));
    sum_images_meas_CCoff_dc = squeeze(sqrt(sum(reco_meas_CCoff_dc.*conj(reco_meas_CCoff_dc), 3)));
    sum_images_meas_VP_dc = squeeze(sqrt(sum(reco_meas_VP_dc.*conj(reco_meas_VP_dc), 3)));
    sum_images_meas_CCon_dc = squeeze(sqrt(sum(reco_meas_CCon_dc.*conj(reco_meas_CCon_dc), 3)));
else
    % only take single coil image
    sum_images_nom_dc = squeeze(reco_nom_dc(:,:,coil_i,:,:));
    sum_images_navi_dc = squeeze(reco_navi_dc(:,:,coil_i,:,:));
    sum_images_girf_dc = squeeze(reco_girf_dc(:,:,coil_i,:,:));
    sum_images_meas_CCoff_dc = squeeze(reco_meas_CCoff_dc(:,:,coil_i,:,:));
    sum_images_meas_VP_dc = squeeze(reco_meas_VP_dc(:,:,coil_i,:,:));
    sum_images_meas_CCon_dc = squeeze(reco_meas_CCon_dc(:,:,coil_i,:,:));
end

% Rescale to dicom value range
sum_images_nom_dc = rescale_to_dicom_range(sum_images_nom_dc);
sum_images_navi_dc = rescale_to_dicom_range(sum_images_navi_dc);
sum_images_girf_dc = rescale_to_dicom_range(sum_images_girf_dc);
sum_images_meas_CCoff_dc = rescale_to_dicom_range(sum_images_meas_CCoff_dc);
sum_images_meas_VP_dc = rescale_to_dicom_range(sum_images_meas_VP_dc);
sum_images_meas_CCon_dc = rescale_to_dicom_range(sum_images_meas_CCon_dc);

%% SAVE RESULTS
save([path2save,name2save],'sum_images_nom_dc','sum_images_navi_dc','sum_images_girf_dc','sum_images_meas_CCoff_dc', ...
    'sum_images_meas_VP_dc','sum_images_meas_CCon_dc','girf_delay','nom_delay','meas_delay','nRO','kRO_navi','kRO_girf',...
    'kRO_meas_CCoff','kRO_meas_VP','kRO_meas_CCon','kPE_del','kPE_navi','kPE_girf','max_k');

%% Arrays for plotting
nFillImage = nRO/2 - size(sum_images_nom_dc,2);

sum_images_nom_plot = zeros(nRO/2,nRO/2,size(sum_images_nom_dc,3),size(sum_images_nom_dc,4));
sum_images_navi_plot = zeros(nRO/2,nRO/2,size(sum_images_nom_dc,3),size(sum_images_nom_dc,4));
sum_images_girf_plot = zeros(nRO/2,nRO/2,size(sum_images_nom_dc,3),size(sum_images_nom_dc,4));
sum_images_meas_CCoff_plot = zeros(nRO/2,nRO/2,size(sum_images_nom_dc,3),size(sum_images_nom_dc,4));
sum_images_meas_VP_plot = zeros(nRO/2,nRO/2,size(sum_images_nom_dc,3),size(sum_images_nom_dc,4));
sum_images_meas_CCon_plot = zeros(nRO/2,nRO/2,size(sum_images_nom_dc,3),size(sum_images_nom_dc,4));

sum_images_nom_plot(:,nFillImage/2+1:end-nFillImage/2,:,:) = sum_images_nom_dc(nRO/4+1:end-nRO/4,:,:,:);
sum_images_navi_plot(:,nFillImage/2+1:end-nFillImage/2,:,:) = sum_images_navi_dc(end-nRO/4:-1:nRO/4+1,:,:,:);
sum_images_girf_plot(:,nFillImage/2+1:end-nFillImage/2,:,:) = sum_images_girf_dc(nRO/4+1:end-nRO/4,:,:,:);
sum_images_meas_CCoff_plot(:,nFillImage/2+1:end-nFillImage/2,:,:) = sum_images_meas_CCoff_dc(nRO/4+1:end-nRO/4,:,:,:);
sum_images_meas_VP_plot(:,nFillImage/2+1:end-nFillImage/2,:,:) = sum_images_meas_VP_dc(nRO/4+1:end-nRO/4,:,:,:);
sum_images_meas_CCon_plot(:,nFillImage/2+1:end-nFillImage/2,:,:) = sum_images_meas_CCon_dc(nRO/4+1:end-nRO/4,:,:,:);

%%
% figure;
% colormap gray;
% clim1 = [0 1500];
% clim2 = [3 8.5];
% 
% subplot(2,3,1);
% imagesc(sum_images_navi_plot(:,:,1,1),clim1);
% axis image;
% colorbar();
% 
% subplot(2,3,2);
% imagesc(sum_images_girf_plot(:,:,1,1),clim1);
% axis image;
% colorbar();
% 
% subplot(2,3,3);
% imagesc(sum_images_meas_CCoff_plot(:,:,1,1),clim1);
% axis image;
% colorbar();
% 
% subplot(2,3,4);
% imagesc(log(sum_images_navi_plot(:,:,1,1)),clim2);
% axis image;
% colorbar();
% 
% subplot(2,3,5);
% imagesc(log(sum_images_girf_plot(:,:,1,1)),clim2);
% axis image;
% colorbar();
% 
% subplot(2,3,6);
% imagesc(log(sum_images_meas_CCoff_plot(:,:,1,1)),clim2);
% axis image;
% colorbar();


%%
function [k_RO, B0_phase] = calc_kRO(grad, t_dwell, delay, t_axis_ADC, t_sim, adc, nRO, nPE)
%
    gamma = 267.513*10^6; %Hz/T

    F_traj = 1/t_dwell;
    f_axis_traj = linspace(-F_traj/2, F_traj/2, size(grad,2));
    Gx_traj = zeros(size(grad));
    for term=1:size(grad,1)
        G = fft_1D(grad(term,:),2) .* exp(1i*2*pi*f_axis_traj *delay);
        Gx_traj(term,:) = real(ifft_1D(G,2));
    end
    grad = Gx_traj;
    
    t_axis_ADC_traj = (0:1:size(grad,2)-1)*t_dwell + t_dwell/2;
    t_eps = 1e-12;
    t_axis_ADC_traj_00 = [-2*t_eps, t_axis_ADC_traj(1)-t_eps, t_axis_ADC_traj, t_axis_ADC_traj(end)+t_eps, t_axis_ADC_traj(end)+2*t_eps];
    T_traj_us = t_axis_ADC_traj(end)*1e6;
    t_1us_traj = (0:1:T_traj_us)*1e-6;
    
    grad_traj_00 = [zeros(2,1); grad(2,:).'; zeros(2,1)];
    grad_traj_pp = interp1(t_axis_ADC_traj_00, grad_traj_00, 'linear','pp');
    grad_traj_1us = ppval(grad_traj_pp, t_1us_traj);
    
    B0_traj_00 = [zeros(2,1); grad(1,:).'; zeros(2,1)];
    B0_traj_pp = interp1(t_axis_ADC_traj_00, B0_traj_00, 'linear','pp');
    B0_traj_1us = ppval(B0_traj_pp, t_1us_traj);
    
    kx_meas_pp = fnint(grad_traj_pp);
    B0_meas_phase_pp = fnint(B0_traj_pp);
    
    kx_meas = gamma*ppval(kx_meas_pp, t_sim);
    B0_meas = gamma*ppval(B0_meas_phase_pp, t_sim);
    
    kx_meas = kx_meas(adc==1);
    kx_meas = reshape(kx_meas, [],nPE);
    B0_meas = B0_meas(adc==1);
    B0_meas = reshape(B0_meas, [],nPE);
    % Resample with dwelltime
    k_RO = zeros(nRO,nPE);
    B0_phase = zeros(nRO,nPE);
    t_ax_k = (0:1:size(kx_meas,1)-1)*1e-6;
    t_ax_k_00 = [-2*t_eps, t_ax_k(1)-t_eps, t_ax_k, t_ax_k(end)+t_eps, t_ax_k(end)+2*t_eps];
    for PE=1:nPE
        kx_currPE = kx_meas(:,PE);
        B0_currPE = B0_meas(:,PE);
        kx_currPE_00 = [zeros(2,1); kx_currPE; zeros(2,1)];
        B0_currPE_00 = [zeros(2,1); B0_currPE; zeros(2,1)];
        kx_currPE_pp = interp1(t_ax_k_00, kx_currPE_00, 'linear','pp');
        B0_currPE_pp = interp1(t_ax_k_00, B0_currPE_00, 'linear','pp');
        k_RO(:,PE) = ppval(kx_currPE_pp, t_axis_ADC);
        B0_phase(:,PE) = ppval(B0_currPE_pp, t_axis_ADC);
    end
%
end

%%
function [k_RO, k_PE, B0_phase, grad_RO] = calc_traj(G_x, G_y, G_z, B_0, t_ax, t_axis_ADC, adc, nRO, nPE, orientation, PE_dir_90deg)
%
    gamma = 267.513*10^6; %Hz/T
    t_eps = 1e-12;
    t_ax_00 = [-2*t_eps, t_ax(1)-t_eps, t_ax, t_ax(end)+t_eps, t_ax(end)+2*t_eps];

    Gx_00 = [zeros(2,1); G_x; zeros(2,1)];
    Gy_00 = [zeros(2,1); G_y; zeros(2,1)];
    Gz_00 = [zeros(2,1); G_z; zeros(2,1)];
    B0_00 = [zeros(2,1); B_0; zeros(2,1)];
    
    Gx_pp = interp1(t_ax_00, Gx_00, 'linear','pp');
    Gy_pp = interp1(t_ax_00, Gy_00, 'linear','pp');
    Gz_pp = interp1(t_ax_00, Gz_00, 'linear','pp');
    B0_pp = interp1(t_ax_00, B0_00, 'linear','pp');
    
    kx_pp = fnint(Gx_pp);
    ky_pp = fnint(Gy_pp);
    kz_pp = fnint(Gz_pp);
    B0_phase_pp = fnint(B0_pp);
    
    kx = gamma*ppval(kx_pp, t_ax);
    ky = gamma*ppval(ky_pp, t_ax);
    kz = gamma*ppval(kz_pp, t_ax);
    B0 = gamma*ppval(B0_phase_pp, t_ax);
    
    kx = kx(adc==1);
    kx = reshape(kx, [],nPE);
    ky = ky(adc==1);
    ky = reshape(ky, [],nPE);
    kz = kz(adc==1);
    kz = reshape(kz, [],nPE);
    B0 = B0(adc==1);
    B0 = reshape(B0, [],nPE);
    gradX = G_x(adc==1);
    gradX = reshape(gradX, [],nPE);
    gradY = G_y(adc==1);
    gradY = reshape(gradY, [],nPE);
    gradZ = G_z(adc==1);
    gradZ = reshape(gradZ, [],nPE);
    % Resample with dwelltime
    k_RO = zeros(nRO,nPE);
    k_PE = zeros(nRO,nPE);
    B0_phase = zeros(nRO,nPE);
    grad_RO = zeros(nRO,nPE);
    t_ax_k = (0:1:size(kx,1)-1)*1e-6;
    t_ax_k_00 = [-2*t_eps, t_ax_k(1)-t_eps, t_ax_k, t_ax_k(end)+t_eps, t_ax_k(end)+2*t_eps];
    
    for PE=1:nPE
        B0_currPE = B0(:,PE);
        B0_currPE_00 = [zeros(2,1); B0_currPE; zeros(2,1)];
        B0_currPE_pp = interp1(t_ax_k_00, B0_currPE_00, 'linear','pp');
        B0_phase(:,PE) = ppval(B0_currPE_pp, t_axis_ADC);
        if strcmp(orientation, 'dTra')
            kx_currPE = kx(:,PE);
            ky_currPE = ky(:,PE);
            gradx_currPE = gradX(:,PE);
            grady_currPE = gradY(:,PE);
            kx_currPE_00 = [zeros(2,1); kx_currPE; zeros(2,1)];
            ky_currPE_00 = [zeros(2,1); ky_currPE; zeros(2,1)];
            gradx_currPE_00 = [zeros(2,1); gradx_currPE; zeros(2,1)];
            grady_currPE_00 = [zeros(2,1); grady_currPE; zeros(2,1)];
            kx_currPE_pp = interp1(t_ax_k_00, kx_currPE_00, 'linear','pp');
            ky_currPE_pp = interp1(t_ax_k_00, ky_currPE_00, 'linear','pp');
            gradx_currPE_pp = interp1(t_ax_k_00, gradx_currPE_00, 'linear','pp');
            grady_currPE_pp = interp1(t_ax_k_00, grady_currPE_00, 'linear','pp');
            if PE_dir_90deg
                k_RO(:,PE) = ppval(ky_currPE_pp, t_axis_ADC);
                k_PE(:,PE) = ppval(kx_currPE_pp, t_axis_ADC);
                grad_RO(:,PE) = ppval(grady_currPE_pp, t_axis_ADC);
            else
                k_RO(:,PE) = ppval(kx_currPE_pp, t_axis_ADC);
                k_PE(:,PE) = ppval(ky_currPE_pp, t_axis_ADC);
                grad_RO(:,PE) = ppval(gradx_currPE_pp, t_axis_ADC);
            end
        elseif strcmp(orientation, 'dSag')
            kz_currPE = kz(:,PE);
            ky_currPE = ky(:,PEs);
            gradz_currPE = gradZ(:,PE);
            grady_currPE = gradY(:,PE);
            kz_currPE_00 = [zeros(2,1); kz_currPE; zeros(2,1)];
            ky_currPE_00 = [zeros(2,1); ky_currPE; zeros(2,1)];
            gradz_currPE_00 = [zeros(2,1); gradz_currPE; zeros(2,1)];
            grady_currPE_00 = [zeros(2,1); grady_currPE; zeros(2,1)];
            kz_currPE_pp = interp1(t_ax_k_00, kz_currPE_00, 'linear','pp');
            ky_currPE_pp = interp1(t_ax_k_00, ky_currPE_00, 'linear','pp');
            gradz_currPE_pp = interp1(t_ax_k_00, gradz_currPE_00, 'linear','pp');
            grady_currPE_pp = interp1(t_ax_k_00, grady_currPE_00, 'linear','pp');
            if PE_dir_90deg
                k_RO(:,PE) = ppval(ky_currPE_pp, t_axis_ADC);
                k_PE(:,PE) = ppval(kz_currPE_pp, t_axis_ADC);
                grad_RO(:,PE) = ppval(grady_currPE_pp, t_axis_ADC);
            else
                k_RO(:,PE) = ppval(kz_currPE_pp, t_axis_ADC);
                k_PE(:,PE) = ppval(ky_currPE_pp, t_axis_ADC);
                grad_RO(:,PE) = ppval(gradz_currPE_pp, t_axis_ADC);
            end
        elseif strcmp(orientation, 'dCor')
            kz_currPE = kz(:,PE);
            kx_currPE = kx(:,PE);
            gradz_currPE = gradZ(:,PE);
            gradx_currPE = gradX(:,PE);
            kz_currPE_00 = [zeros(2,1); kz_currPE; zeros(2,1)];
            kx_currPE_00 = [zeros(2,1); kx_currPE; zeros(2,1)];
            gradz_currPE_00 = [zeros(2,1); gradz_currPE; zeros(2,1)];
            gradx_currPE_00 = [zeros(2,1); gradx_currPE; zeros(2,1)];
            kz_currPE_pp = interp1(t_ax_k_00, kz_currPE_00, 'linear','pp');
            kx_currPE_pp = interp1(t_ax_k_00, kx_currPE_00, 'linear','pp');
            gradz_currPE_pp = interp1(t_ax_k_00, gradz_currPE_00, 'linear','pp');
            gradx_currPE_pp = interp1(t_ax_k_00, gradx_currPE_00, 'linear','pp');
            if PE_dir_90deg
                k_RO(:,PE) = ppval(kx_currPE_pp, t_axis_ADC);
                k_PE(:,PE) = ppval(kz_currPE_pp, t_axis_ADC);
                grad_RO(:,PE) = ppval(gradx_currPE_pp, t_axis_ADC);
            else
                k_RO(:,PE) = ppval(kz_currPE_pp, t_axis_ADC);
                k_PE(:,PE) = ppval(kx_currPE_pp, t_axis_ADC);
                grad_RO(:,PE) = ppval(gradz_currPE_pp, t_axis_ADC);
            end
        end
    end
%
end

%%
function [images] = rescale_to_dicom_range(images_unscaled)
    images = zeros(size(images_unscaled));
    % Rescale to dicom value range, see https://stats.stackexchange.com/questions/25894/changing-the-scale-of-a-variable-to-0-100
    for slice=1:size(images_unscaled,3)
        for meas=1:size(images_unscaled,4)
            v_max = max(images_unscaled(:,:,slice,meas), [], 'all');
            v_min = min(images_unscaled(:,:,slice,meas), [], 'all');
            vmax_d = 4096;
            images(:,:,slice,meas) = (vmax_d-0)/(v_max-v_min)*(images_unscaled(:,:,slice,meas)-v_min);
        end
    end
end








