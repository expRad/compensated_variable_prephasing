function [ out_grad, dwelltime ] = calcMeasOutput_VP_selectSlices_mat( file, numRepPerGrad, whichGrad, numGrad, calcChannels, singleCoil, skipCoils, nDummies, compConcomit, deleteSlices)
% This function calculates the measured gradient output for the
% variable-prephasing (VP) method.
% The raw data, however, is assumed to be the same as in the CVP or FCVP method.
% Part of it is neglegted for the calculations in the end.

%% Read rawdata
rawdata = load(file);

weigh_equal = 0; % parameter for the coil weighting

dwelltime = rawdata.dwelltime_FCVP; % dwelltime in seconds
FOV = rawdata.FOV; % FOV width and height in mm
orientation = rawdata.orientation;
PosSlices = rawdata.PosSlices;
PosSlices = sort(PosSlices); % scanner always starts with the most negative slice position

%% Get kspace data
kspace = rawdata.kspace; % [read-out-points, coils, PE steps, partitions, slices, Averages, Cardiac-Phases(1), ADC readouts, measurements(if>1)]
% "PE steps" and "partitions" contain the data from the phase encoding in PE- and RO-direction. Their number is always equal.
% The "Averages"-loop contains the (different) gradient waveforms. If compConcomit=1, it is assumed that each waveform was played out twice, with inverted sign the second time.
kspace = kspace(:,:,:,:,:,nDummies+1:end,:,:,:);

% Delete non-selected slices
for slice = size(kspace,5):-1:1
    if ismember(slice,deleteSlices)
        kspace(:,:,:,:,slice,:,:,:,:) = [];
        PosSlices(slice) = [];
    end
end
numSlices = length(PosSlices);

disp(['size(kspace)=',num2str(size(kspace))])
if (length(size(kspace))==6)
    % for the case of just 1 ADC and 1 measurement: add extra dimensions, so the dimensions in the following are still correct
    kspace=repmat(kspace,1,1,1,1,1,1,1,1,numRepPerGrad);
end

% Determine number of variable-prephasing steps (CVP or FCVP measurement scheme)
% For every VP step, there are 4 measurements:
%   1) with prephasing & test gradient
%   2) like 1) but with inverted signs or without test gradient (depending on compConcomit)
%   3) with prephasing gradient and shifted slice selection
%   4) like 3) but with the sign of the prephaser inverted
numVP = size(kspace,6)/4/numGrad;

disp(['size(kspace)=',num2str(size(kspace))])
clearvars rawdata;

%% Determine some parameters for easier array reshaping later on
numRep = size(kspace,9); % number of repetitions (measurements)
numIter = floor(numRep/numRepPerGrad); % number of loop-counts
numROP = size(kspace,1); % number of Read Out Points
numPE = size(kspace,3); % number of phase encode steps
numADC = size(kspace,8); % number of ADCs
numMeas = size(kspace,6); % number of measurements = numTriang*numVP*2

%% Fourier-Transform k-space data along the two phase-encoding directions
kspace = fft_1D(kspace, 3);
kspace = fft_1D(kspace, 4);

%% Sort out unusable coil elements
for coil=size(kspace,2):-1:1
    if ismember(coil, skipCoils)
        kspace(:,coil,:,:,:,:,:,:,:) = [];
    end
end
disp(['size(kspace)=',num2str(size(kspace))])

%% Get coil-combined magnitude and phase data
[diff_phase, magnitude] = combineCoils_FCVP(kspace, dwelltime, weigh_equal, singleCoil, numRepPerGrad);
% dimensions: [numROP, numPE, numPE, numSlices, numMeas, numADC, numRep]
clearvars kspace;

%% Average over numRepPerGIRF repetitions
diff_phase_avg = zeros(numROP, numPE, numPE, numSlices, numMeas, numADC, numIter);
for i=1:1:numIter
    diff_phase_avg(:,:,:,:,:,:,i) = mean(diff_phase(:,:,:,:,:,:,(i-1)*numRepPerGrad+1:i*numRepPerGrad),7);
end
clearvars diff_phase;
% Take the specified iteration
diff_phase_avg = diff_phase_avg(:,:,:,:,:,:,whichGrad); % [numROP, numPE1, numPE2, numSlices, numMeas, numADC, 1]

%%
% The acquisition order in the numMeas-dimension is as follows:
%   First, third, fifth, ... measurement: with prephasing and test gradient
%   Second, fourth, sixth, ... measurement: with inverted sign or without test gradient

% Now bring the data into the right order for the matrix calculation
diff_phase = permute(diff_phase_avg, [2,3,4,5,1,6,7]); % [numPE1, numPE2, numSlices, numMeas, numROP, numADC, 1]
diff_phase = reshape(diff_phase, [numPE, numPE, numSlices, numMeas, numROP*numADC]);
numPositions = numSlices*numPE*numPE;
diff_phase = reshape(diff_phase, [numPositions, numMeas, numROP*numADC]);
% Separate test- and reference measurements
% FIXME: This part is valid for numGrad=1 so far. It may have to be adapted for numGrad>1!
diff_phase_1 = diff_phase(:,1:4:end,:); % [numPositions, numVP, numTimePoints]
diff_phase_2 = diff_phase(:,2:4:end,:); % [numPositions, numVP, numTimePoints]
diff_phase_3 = diff_phase(:,3:4:end,:); % [numPositions, numVP, numTimePoints]
diff_phase_4 = diff_phase(:,4:4:end,:); % [numPositions, numVP, numTimePoints]

clearvars diff_phase

%% Average magnitude data over numRepPerGIRF repetitions
% The magnitude is needed to sort out unusable voxels further down
magnitude_avg = zeros(numROP, numPE, numPE, numSlices, numMeas, numADC, numIter);
for i=1:1:numIter
    magnitude_avg(:,:,:,:,:,:,i) = mean(magnitude(:,:,:,:,:,:,(i-1)*numRepPerGrad+1:i*numRepPerGrad),7);
end
magnitude_avg = magnitude_avg(:,:,:,:,:,:,whichGrad);
clearvars magnitude;
% Reorder
magnitude = permute(magnitude_avg, [2,3,4,5,1,6,7]); % [numPE1, numPE2, numSlices, numMeas, numROP, numADC, 1]
magnitude = reshape(magnitude, [numPE, numPE, numSlices, numMeas, numROP*numADC]);
magnitude = reshape(magnitude, [numPositions, numMeas, numROP*numADC]);
magnitude_thres = squeeze(mean(magnitude(:,2,10:50),3));
% Separate test- and reference measurements
% FIXME: This part is valid for numGrad=1 so far. It may have to be adapted for numGrad>1!
magnitude_1 = magnitude(:,1:4:end,:); % [numPositions, numVP, numTimePoints]
magnitude_2 = magnitude(:,2:4:end,:); % [numPositions, numVP, numTimePoints]
magnitude_3 = magnitude(:,3:4:end,:); % [numPositions, numVP, numTimePoints]
magnitude_4 = magnitude(:,4:4:end,:); % [numPositions, numVP, numTimePoints]

%% Get the positions of the measured voxels
positions = createPositionArray(orientation, numPE, numSlices, FOV, PosSlices); % [numPE(PE), numPE(part), numSlices, 3]
positions = reshape(positions, [numPE*numPE*numSlices, 3]); % [numVoxels, 3]
disp(['size(positions) = ',num2str(size(positions))]);

%% Sort out unusable voxels
% by thresholding the signal magnitude
max_mag = max(magnitude_thres);

validVoxels = zeros(size(positions,1),1) + 1;

if numPE>1
    for voxel=numPE*numPE*numSlices:-1:1
        r = sqrt(positions(voxel,1)*positions(voxel,1) + positions(voxel,2)*positions(voxel,2) + positions(voxel,3)*positions(voxel,3));
        if 0 % insert desired condition here
            positions(voxel,:) = [];
            diff_phase_test(voxel,:,:) = [];
            diff_phase_ref(voxel,:,:) = [];
            validVoxels(voxel) = 0;
        elseif magnitude_thres(voxel) < max_mag*0.6
            positions(voxel,:) = [];
            diff_phase_test(voxel,:,:) = [];
            diff_phase_ref(voxel,:,:) = [];
            validVoxels(voxel) = 0;
        end
    end
end
clearvars magnitude;

for voxel=1:1:size(positions,1)
    r = sqrt(positions(voxel,1)*positions(voxel,1) + positions(voxel,2)*positions(voxel,2) + positions(voxel,3)*positions(voxel,3));
    disp(['voxel ',num2str(voxel)])
    disp(['x=',num2str(positions(voxel,1)),', y=',num2str(positions(voxel,2)),', z=',num2str(positions(voxel,3)),'\n'])
    disp(['r = ',num2str(r)])
end

FIDs.validVoxels = validVoxels;
%% Reshape into matrices
diff_phase_1 = reshape(diff_phase_1, [],numROP*numADC); % [numPositions*numVP, numTimePoints]
diff_phase_2 = reshape(diff_phase_2, [],numROP*numADC); % [numPositions*numVP, numTimePoints]
diff_phase_3 = reshape(diff_phase_3, [],numROP*numADC); % [numPositions*numVP, numTimePoints]
diff_phase_4 = reshape(diff_phase_4, [],numROP*numADC); % [numPositions*numVP, numTimePoints]

magnitude_1 = reshape(magnitude_1, [],numROP*numADC); % [numPositions*numVP, numTimePoints]
magnitude_2 = reshape(magnitude_2, [],numROP*numADC); % [numPositions*numVP, numTimePoints]
magnitude_3 = reshape(magnitude_3, [],numROP*numADC); % [numPositions*numVP, numTimePoints]
magnitude_4 = reshape(magnitude_4, [],numROP*numADC); % [numPositions*numVP, numTimePoints]

% Smoothing for robustness against noise
diff_phase_1 = smoothdata(diff_phase_1,2,'movmedian',3);
diff_phase_2 = smoothdata(diff_phase_2,2,'movmedian',3);
diff_phase_3 = smoothdata(diff_phase_3,2,'movmedian',3);
diff_phase_4 = smoothdata(diff_phase_4,2,'movmedian',3);

%% Stack the phase derivatives to get the left handside of the matrix equation
diff_phase = cat(1, diff_phase_1, diff_phase_2); % [2*numPositions*numVP, numTimePoints]
disp(['size(diff_phase) = ',num2str(size(diff_phase))]);
magnitude = cat(1, magnitude_1, magnitude_2); % [2*numPositions*numVP, numTimePoints]
clearvars diff_phase_1 diff_phase_2 diff_phase_3 diff_phase_4 magnitude_1 magnitude_2 magnitude_3 magnitude_4

%% Get the probing matrix
if numPE==1
    probingMatrix = zeros(size(positions,1), calcChannels);
    for slice=1:1:size(positions,1)
        probingMatrix(slice,1) = 1;
        if calcChannels>1
            if strcmp(orientation,'dTra')
                slicePosition = positions(slice,3);
            elseif strcmp(orientation,'dSag')
                slicePosition = positions(slice,1);
            elseif strcmp(orientation,'dCor')
                slicePosition = positions(slice,2);
            end
            probingMatrix(slice,2) = slicePosition;
            if calcChannels>2
                probingMatrix(slice,3) = 2*slicePosition*slicePosition;
            end
        end
    end
else
    probingMatrix = createProbingMatrix(positions,calcChannels); % [numValidVoxels, 4/9/16], depending on the maximum expansion order
    numPositions = size(probingMatrix,1);
end

%% Create the coefficient matrix of the matrix equation
N_p = size(probingMatrix,1); % number of positions
N_L = size(probingMatrix,2); % number of basis functions / channels
coeff_mat = zeros(size(diff_phase,1), N_L+N_p);
    
tmp = repmat(probingMatrix, [numVP,1]);
coeff_mat(1:numVP*N_p,1:N_L) = tmp;
if compConcomit
    coeff_mat(numVP*N_p+1:2*numVP*N_p,1:N_L) = -tmp;
else
    coeff_mat(numVP*N_p+1:2*numVP*N_p,1:N_L) = zeros(size(tmp));
end

coeff_mat(:,N_L+1:end) = repmat(eye(N_p),[2*numVP,1]);

clearvars tmp;

%% Calculate the solution matrix...

% ... for each time point separately, so measurements with a too low amplitude can be sorted out
sol_mat = zeros(size(coeff_mat,2), size(diff_phase,2)); % [(1+numVP)*numChannels+2*numVP*numPositions, numTimePoints]
for t = 1:size(diff_phase,2)
    f = diff_phase(:,t); % [4*numVP*numPositions, 1]
    m = magnitude(:,t); % [4*numVP*numPositions, 1]
    invcov = zeros(size(f,1));
    for i = 1:size(f,1)
        invcov(i,i) = m(i)*m(i);
    end
    invcov(end,end) = m(end)*m(end);
    x = (transpose(coeff_mat)*invcov*coeff_mat) \ (transpose(coeff_mat)*invcov*f);
    sol_mat(:,t) = x;
end

%% Calculate the output signals
gamma = 267.513*10^6; %Hz/T
out_grad = 1/gamma * sol_mat(1:N_L,:); % [numChannels, numTimePoints]
out_BG = sol_mat(N_L+1:end,:); % [numVP*numChannels+2*numVP*numPositions, numTimePoints]

end
