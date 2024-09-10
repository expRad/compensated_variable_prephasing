clear all;
% Change current directory to that of this .m file
mfile_name          = mfilename('fullpath');
if contains(mfile_name,'LiveEditorEvaluationHelper')
    mfile_name = matlab.desktop.editor.getActiveFilename;
end
[pathstr,name,ext]  = fileparts(mfile_name);
cd(pathstr);
% Add necessary scripts to the MATLAB path
folder1 = '.\helper_functions\';
addpath(folder1);
%%
path_1 = '.\GSTF_data\';
path_2 = '.\gradient_data\';

%% Load GSTFs
H_x = load([path_1,'Hx_fast_FT0_1030.mat']).H_combined;
H_y = load([path_1,'Hy_fast_FT0_1030.mat']).H_combined;
H_z = load([path_1,'Hz_fast_FT0_1030.mat']).H_combined;
%% Normalize GSTFs
H_x.gstf(:,2) = H_x.gstf(:,2)./mean(abs(H_x.gstf(ceil(end/2)-10:ceil(end/2)+10,2)),1);
H_y.gstf(:,2) = H_y.gstf(:,2)./mean(abs(H_y.gstf(ceil(end/2)-10:ceil(end/2)+10,2)),1);
H_z.gstf(:,2) = H_z.gstf(:,2)./mean(abs(H_z.gstf(ceil(end/2)-10:ceil(end/2)+10,2)),1);

%% Evaluate VP, CVP, and FCVP measurements of the trapezoidal test gradient on the x-axis
compConcomit = 0;
[ grad_trap_x_CVP, dwelltime_gradTrap, grad_VPonly_x, invcovmat_trapx_CCoff ] = calcMeasOutput_FCVP_selectSlices_mat( [path_2,'GradMeas_FCVP_Trap_x_10Dum_10VP_9sl_CCoff.mat'], 1, 1, 1, 2, 0, [], 10, compConcomit, []);
[ grad_trap_x_VP, dwelltime_gradTrap ] = calcMeasOutput_VP_selectSlices_mat( [path_2,'GradMeas_FCVP_Trap_x_10Dum_10VP_9sl_CCoff.mat'], 1, 1, 1, 2, 0, [], 10, compConcomit, []);
compConcomit = 1;
[ grad_trap_x_FCVP, dwelltime_gradTrap, ~, invcovmat_trapx_CCon ] = calcMeasOutput_FCVP_selectSlices_mat( [path_2,'GradMeas_FCVP_Trap_x_10Dum_10VP_9sl_CCon.mat'], 1, 1, 1, 2, 0, [], 10, compConcomit, []);
t_gradTrap = ((1:size(grad_trap_x_CVP,2))-0.5)*dwelltime_gradTrap;
%% Evaluate VP, CVP, and FCVP measurements of the trapezoidal test gradient on the z-axis
% compConcomit = 0;
% [ grad_trap_z_CVP, dwelltime_gradTrap, grad_VPonly_z, ~ ] = calcMeasOutput_FCVP_selectSlices_mat( [path_2,'GradMeas_FCVP_Trap_z_10Dum_10VP_9sl_CCoff.mat'], 1, 1, 1, 2, 0, [], 10, compConcomit, []);
% [ grad_trap_z_VP, dwelltime_gradTrap ] = calcMeasOutput_VP_selectSlices_mat( [path_2,'GradMeas_FCVP_Trap_z_10Dum_10VP_9sl_CCoff.mat'], 1, 1, 1, 2, 0, [], 10, compConcomit, []);
% compConcomit = 1;
% [ grad_trap_z_FCVP, dwelltime_gradTrap, ~ , ~] = calcMeasOutput_FCVP_selectSlices_mat( [path_2,'GradMeas_FCVP_Trap_z_10Dum_10VP_9sl_CCon.mat'], 1, 1, 1, 2, 0, [], 10, compConcomit, []);
%% Evaluate VP, CVP, and FCVP measurements of the trapezoidal test gradient on the y-axis
% compConcomit = 0;
% [ grad_trap_y_CVP, dwelltime_gradTrap, grad_VPonly_y, ~ ] = calcMeasOutput_FCVP_selectSlices_mat( [path_2,'GradMeas_FCVP_Trap_y_10Dum_10VP_9sl_CCoff.mat'], 1, 1, 1, 2, 0, [], 10, compConcomit, []);
% [ grad_trap_y_VP, dwelltime_gradTrap ] = calcMeasOutput_VP_selectSlices_mat( [path_2,'GradMeas_FCVP_Trap_y_10Dum_10VP_9sl_CCoff.mat'], 1, 1, 1, 2, 0, [], 10, compConcomit, []);
% compConcomit = 1;
% [ grad_trap_y_FCVP, dwelltime_gradTrap, ~, ~ ] = calcMeasOutput_FCVP_selectSlices_mat( [path_2,'GradMeas_FCVP_Trap_y_10Dum_10VP_9sl_CCon.mat'], 1, 1, 1, 2, 0, [], 10, compConcomit, []);

%% Evaluate VP, CVP, and FCVP measurements of the EPI readout gradient on the x-axis
compConcomit = 0;
[ grad_EPI_x_CVP, dwelltime_gradEPI, grad_EPI_x_VPonly, invcovmat_CCoff ] = calcMeasOutput_FCVP_selectSlices_mat( [path_2,'GradMeas_FCVP_EPI_iso13FOVph80_x_10Dum_10VP_9sl_CCoff.mat'], 1, 1, 1, 2, 0, [], 10, compConcomit, []);
[ grad_EPI_x_VP, dwelltime_gradEPI ] = calcMeasOutput_VP_selectSlices_mat( [path_2,'GradMeas_FCVP_EPI_iso13FOVph80_x_10Dum_10VP_9sl_CCoff.mat'], 1, 1, 1, 2, 0, [], 10, compConcomit, []);
compConcomit = 1;
[ grad_EPI_x_FCVP, dwelltime_gradEPI, ~ , invcovmat_CCon] = calcMeasOutput_FCVP_selectSlices_mat( [path_2,'GradMeas_FCVP_EPI_iso13FOVph80_x_10Dum_10VP_9sl_CCon.mat'], 1, 1, 1, 2, 0, [], 10, compConcomit, []);
t_gradEPI = ((1:size(grad_EPI_x_CVP,2))-0.5)*dwelltime_gradEPI;

%% Calculate GSTF prediction for trapezoidal test gradient
sim_trap = load([path_2, 'sim_GradMeas_Trap_TR20.mat']);
adc = sim_trap.adc; % [TR_us,1]
grad_x = sim_trap.grad_x; % T/m
grad_x(adc==0) = 0;
f_axis_sim = linspace(-500000, 500000, size(grad_x,1));
% Append zeros to avoid side effects by the fft calculation
nExtra = (1e6-size(grad_x,1))/2;
grad_x = [zeros(nExtra,size(grad_x,2)); grad_x; zeros(nExtra,size(grad_x,2))];
f_axis_simextra = linspace(-500000, 500000, size(grad_x,1));
% Interpolate GSTF to the appropriate frequency axis
gstf_x = interp1(H_x.f_axis, H_x.gstf(:,2), f_axis_simextra, 'makima',0).';
% Calculate GSTF-predicted gradient
G = fft_1D(grad_x,1) .* gstf_x .* exp(1i*2*pi*f_axis_simextra.'*(-1e-6));
Gx = real(ifft_1D(G,1)) + H_x.fieldOffsets(2);
Gx = Gx(nExtra+1:end-nExtra);
grad_trap_x_gstf = Gx(adc==1);
t_gradTrap_gstf = (1:size(grad_trap_x_gstf,1))*1e-6;
grad_trap_x_gstf = interp1(t_gradTrap_gstf, grad_trap_x_gstf, t_gradTrap);

%% Calculate GSTF prediction for EPI readout gradient
sim_trap = load([path_2, 'sim_EPI_tra_peAP_FOVph80.mat']);
adc = sim_trap.adc; % [TR_us,1]
grad_x = sim_trap.grad_x; % T/m
f_axis_sim = linspace(-500000, 500000, size(grad_x,1));
% Append zeros to avoid side effects by the fft calculation
nExtra = (1e6-size(grad_x,1))/2;
grad_x = [zeros(nExtra,size(grad_x,2)); grad_x; zeros(nExtra,size(grad_x,2))];
f_axis_simextra = linspace(-500000, 500000, size(grad_x,1));
% Interpolate GSTF to the appropriate frequency axis
gstf_x = interp1(H_x.f_axis, H_x.gstf(:,2), f_axis_simextra, 'makima',0).';
% Calculate GSTF-predicted gradient
G = fft_1D(grad_x,1) .* gstf_x .* exp(1i*2*pi*f_axis_simextra.'*(-6.3e-6));
Gx = real(ifft_1D(G,1)) + H_x.fieldOffsets(2);
Gx = Gx(nExtra+1:end-nExtra);
grad_EPI_x_gstf = Gx;
t_gradEPI_gstf = (1:size(grad_EPI_x_gstf,1))*1e-6;
grad_EPI_x_gstf = interp1(t_gradEPI_gstf, grad_EPI_x_gstf, t_gradEPI,'linear',0);

%% Integrate the EPI gradient waveforms
grad_EPI_x_FCVP_CCoff_pp = interp1(t_gradEPI, grad_EPI_x_CVP(2,:).', 'linear', 'pp');
grad_EPI_x_VP_CCoff_pp = interp1(t_gradEPI, grad_EPI_x_VP(2,:).', 'linear', 'pp');
grad_EPI_x_FCVP_CCon_pp = interp1(t_gradEPI, grad_EPI_x_FCVP(2,:).', 'linear', 'pp');
grad_EPI_x_gstf_pp = interp1(t_gradEPI, grad_EPI_x_gstf, 'linear', 'pp');

k_EPI_x_FCVP_CCoff_pp = fnint(grad_EPI_x_FCVP_CCoff_pp);
k_EPI_x_VP_CCoff_pp = fnint(grad_EPI_x_VP_CCoff_pp);
k_EPI_x_FCVP_CCon_pp = fnint(grad_EPI_x_FCVP_CCon_pp);
k_EPI_x_gstf_pp = fnint(grad_EPI_x_gstf_pp);

gamma = 267.513*10^6; %Hz/T
k_EPI_x_FCVP_CCoff = gamma*ppval(k_EPI_x_FCVP_CCoff_pp, t_gradEPI);
k_EPI_x_VP_CCoff = gamma*ppval(k_EPI_x_VP_CCoff_pp, t_gradEPI);
k_EPI_x_FCVP_CCon = gamma*ppval(k_EPI_x_FCVP_CCon_pp, t_gradEPI);
k_EPI_x_gstf = gamma*ppval(k_EPI_x_gstf_pp, t_gradEPI);

%%
blue = [0 0.4470 0.7410];
orange = [0.8500 0.3250 0.0980];
violet = [0.4940 0.1840 0.5560];
green = [0.4660 0.6740 0.1880];
yellow = [0.9290 0.6940 0.1250];
lblue = [0.3010 0.7450 0.9330];
dred = [0.6350 0.0780 0.1840];

%% Calculate weighted sum over lingering effects from the prephasing gradients
weights = zeros(10,size(grad_EPI_x_CVP,2));
for i=0:9
    weights(i+1,:) = mean(invcovmat_CCoff(i*9+1:(i+1)*9,:),1); % average over slices
end
weights_new = zeros(size(weights));
for t=1:size(weights_new,2)
    weights_new(:,t) = weights(:,t)/sum(weights(:,t));
end
grad_EPI_x_onlyVPs = grad_EPI_x_VPonly(2:2:20,:);
weighted_sum_VPs = sum(weights_new.*grad_EPI_x_onlyVPs, 1);

%% Plot trapezoid measured with VP, and GSTF-predicted (Figure 2)
h = 20;
fig = figure('Units','centimeters', 'InnerPosition',[0 0 17.56 h]);

dy = 0.18*21/h;
dx1 = 0.85;
dx2 = 0.385;
x1 = 0.13;
d_y = 0.06;
a = 0.805;

ax1 = subplot('Position',[x1 a dx1 dy]);
plot(t_gradTrap(1:end-3)*1000, 1000*grad_trap_x_VP(2,1:end-3).','DisplayName','variable-prephasing (VP)','LineWidth',1.5);
hold on;
plot(t_gradTrap(1:end-3)*1000, 1000*grad_trap_x_gstf(1,1:end-3).','-.','DisplayName','GSTF prediction','LineWidth',1.5);
legend('FontName','Arial','Fontsize',8);
xlim([0 10.5]);
ylim([-5 45]);
ylabel('Gradient (mT/m)','FontName','Arial','Fontsize',9.5);
xlabel('Time (ms)','FontName','Arial','Fontsize',9.5)
rectangle('Position',[0 40.8 1 2],'LineWidth',0.8,'EdgeColor',[1 1 1]*0.5);
text(1.03,42,'(B)','FontName','Arial','Fontsize',8,'FontWeight','bold','Color',[1 1 1]*0.5);
rectangle('Position',[2 -1 8 2],'LineWidth',0.8,'EdgeColor',[1 1 1]*0.5);
text(2,4,'(C)','FontName','Arial','Fontsize',8,'FontWeight','bold','Color',[1 1 1]*0.5);

text(-1.45,42,'(A)','FontName','Arial','Fontsize',11,'FontWeight','bold');
text(-1.45,-23,'(B)','FontName','Arial','Fontsize',11,'FontWeight','bold');
text(4.9,-23,'(C)','FontName','Arial','Fontsize',11,'FontWeight','bold');
text(-1.45,-90,'(D)','FontName','Arial','Fontsize',11,'FontWeight','bold');
text(-1.45,-156,'(E)','FontName','Arial','Fontsize',11,'FontWeight','bold');

ax2 = subplot('Position',[x1 a-dy-d_y dx2 dy]);
plot(t_gradTrap(1:end-3)*1000, 1000*grad_trap_x_VP(2,1:end-3).','DisplayName','CVP','LineWidth',1.5);
hold on;
plot(t_gradTrap(1:end-3)*1000, 1000*grad_trap_x_gstf(1,1:end-3).','-.','DisplayName','GSTF prediction','LineWidth',1.5);
xlim([0 1]);
ylim([41.3 42.3]);
ylabel('Gradient (mT/m)','FontName','Arial','Fontsize',9.5);
xlabel('Time (ms)','FontName','Arial','Fontsize',9.5)

ax3 = subplot('Position',[0.595 a-dy-d_y dx2 dy]);
plot(t_gradTrap(1:end-3)*1000, 1000*grad_trap_x_VP(2,1:end-3).','DisplayName','CVP','LineWidth',1.5);
hold on;
plot(t_gradTrap(1:end-3)*1000, 1000*grad_trap_x_gstf(1,1:end-3).','-.','DisplayName','GSTF prediction','LineWidth',2);
xlim([2 10]);
ylim([-1 1]*0.1);
xlabel('Time (ms)','FontName','Arial','Fontsize',9.5)

ax4 = subplot('Position',[x1 a-2*dy-2*d_y dx1 dy]);
plot(t_gradTrap(1:end-3)*1000, 1000*(grad_trap_x_VP(2,1:end-3).'-grad_trap_x_gstf(1,1:end-3).'),'DisplayName','VP-GSTF','LineWidth',1.5);
legend('FontName','Arial','Fontsize',8);
xlim([0 10.5]);
ylim([-1 1]*0.3);
ylabel({'Gradient',' difference (mT/m)'},'FontName','Arial','Fontsize',9.5);
xlabel('Time (ms)','FontName','Arial','Fontsize',9.5)

ax5 = subplot('Position',[x1 a-3*dy-3*d_y dx1 dy]);
plot(t_gradTrap(1:end-3)*1000, 1000*(grad_trap_x_VP(2,1:end-3).'-grad_trap_x_gstf(1,1:end-3).'),'DisplayName','VP-GSTF','LineWidth',1.5);
hold on;
plot(t_gradTrap(1:end-3)*1000, 1000*grad_VPonly_x(20,1:end-3).','--','Color',green,'DisplayName','remainder of largest prephasing gradient','LineWidth',1.5);
legend('FontName','Arial','Fontsize',8);
xlim([0 10.5]);
ylim([-1 1]*0.3);
ylabel({'Gradient',' difference (mT/m)'},'FontName','Arial','Fontsize',9.5);
xlabel('Time (ms)','FontName','Arial','Fontsize',9.5);

%% Plot trapezoid, measured with CVP and FCVP, and GSTF-predicted (Figure 3)
h = 15;
fig = figure('Units','centimeters', 'InnerPosition',[0 0 17.56 h]);

dy = 0.18*21/h;
dx1 = 0.85;
dx2 = 0.385;
x1 = 0.13;
a = 0.74;
d_y = 0.08;

ax1 = subplot('Position',[x1 a dx1 dy]);
plot(t_gradTrap(1:end-3)*1000, 1000*grad_trap_x_CVP(2,1:end-3).','Color',yellow,'DisplayName','compensated variable-prephasing (CVP)','LineWidth',1.5);
hold on;
plot(t_gradTrap(1:end-3)*1000, 1000*grad_trap_x_FCVP(2,1:end-3).','--','Color',lblue,'DisplayName','fully compensated variable-prephasing (FCVP)','LineWidth',1.5);
plot(t_gradTrap(1:end-3)*1000, 1000*grad_trap_x_gstf(1,1:end-3).','-.','Color',orange,'DisplayName','GSTF prediction','LineWidth',1.5);
legend('FontName','Arial','Fontsize',8);
xlim([0 10.5]);
ylim([-5 45]);
ylabel('Gradient (mT/m)','FontName','Arial','Fontsize',9.5);
xlabel('Time (ms)','FontName','Arial','Fontsize',9.5)
rectangle('Position',[0 40.8 1 2],'LineWidth',0.8,'EdgeColor',[1 1 1]*0.5);
text(1.03,42,'(B)','FontName','Arial','Fontsize',8,'FontWeight','bold','Color',[1 1 1]*0.5);
rectangle('Position',[2 -1 8 2],'LineWidth',0.8,'EdgeColor',[1 1 1]*0.5);
text(2,4,'(C)','FontName','Arial','Fontsize',8,'FontWeight','bold','Color',[1 1 1]*0.5);

text(-1.45,42,'(A)','FontName','Arial','Fontsize',11,'FontWeight','bold');
text(-1.45,-23,'(B)','FontName','Arial','Fontsize',11,'FontWeight','bold');
text(4.9,-23,'(C)','FontName','Arial','Fontsize',11,'FontWeight','bold');
text(-1.45,-90,'(D)','FontName','Arial','Fontsize',11,'FontWeight','bold');

ax2 = subplot('Position',[x1 a-dy-d_y dx2 dy]);
plot(t_gradTrap(1:end-3)*1000, 1000*grad_trap_x_CVP(2,1:end-3).','Color',yellow,'DisplayName','compensated variable-prephasing (CVP)','LineWidth',1.5);
hold on;
plot(t_gradTrap(1:end-3)*1000, 1000*grad_trap_x_FCVP(2,1:end-3).','--','Color',lblue,'DisplayName','fully compensated variable-prephasing (FCVP)','LineWidth',1.5);
plot(t_gradTrap(1:end-3)*1000, 1000*grad_trap_x_gstf(1,1:end-3).','-.','Color',orange,'DisplayName','GSTF prediction','LineWidth',1.5);
xlim([0 1]);
ylim([41.3 42.3]);
ylabel('Gradient (mT/m)','FontName','Arial','Fontsize',9.5);
xlabel('Time (ms)','FontName','Arial','Fontsize',9.5)

ax3 = subplot('Position',[0.595 a-dy-d_y dx2 dy]);
plot(t_gradTrap(1:end-3)*1000, 1000*grad_trap_x_CVP(2,1:end-3).','Color',yellow,'DisplayName','compensated variable-prephasing (CVP)','LineWidth',1.5);
hold on;
plot(t_gradTrap(1:end-3)*1000, 1000*grad_trap_x_FCVP(2,1:end-3).','--','Color',lblue,'DisplayName','fully compensated variable-prephasing (FCVP)','LineWidth',1.5);
plot(t_gradTrap(1:end-3)*1000, 1000*grad_trap_x_gstf(1,1:end-3).','-.','Color',orange,'DisplayName','GSTF prediction','LineWidth',1.5);
xlim([2 10]);
ylim([-1 1]*0.1);
xlabel('Time (ms)','FontName','Arial','Fontsize',9.5)

ax4 = subplot('Position',[x1 a-2*dy-2*d_y dx1 dy]);
plot(t_gradTrap(1:end-3)*1000, 1000*(grad_trap_x_CVP(2,1:end-3).'-grad_trap_x_gstf(1,1:end-3).'),'Color',yellow,'DisplayName','CVP-GSTF','LineWidth',1.5);
hold on;
plot(t_gradTrap(1:end-3)*1000, 1000*(grad_trap_x_FCVP(2,1:end-3).'-grad_trap_x_gstf(1,1:end-3).'),'Color',lblue,'DisplayName','FCVP-GSTF','LineWidth',1.5);
legend('FontName','Arial','Fontsize',8);
xlim([0 10.5]);
ylim([-1 1]*0.3);
xlabel('Time (ms)','FontName','Arial','Fontsize',9.5)
ylabel({'Gradient',' difference (mT/m)'},'FontName','Arial','Fontsize',9.5);

%% Plot the measured EPI gradient trains (Figure 4)
h = 23;
fig = figure('Units','centimeters', 'InnerPosition',[0 0 17.56 h]);

dy = 0.12*21/h;
dx1 = 0.87;
dx2 = 0.23;
dx3 = 0.4;
x1 = 0.11;
d_y = 0.031;
dy_l = 0.02;
a = 0.865;

ax1 = subplot('Position',[x1 a dx1 dy]);
plot(t_gradEPI(1:end-3)*1000, 1000*grad_EPI_x_VP(2,1:end-3).','DisplayName','VP','LineWidth',1.5,'Color',blue);
hold on;
plot(t_gradEPI(1:end-3)*1000, 1000*grad_EPI_x_CVP(2,1:end-3).','--','DisplayName','CVP','LineWidth',1.7,'Color',yellow);
plot(t_gradEPI(1:end-3)*1000, 1000*grad_EPI_x_FCVP(2,1:end-3).','-.','DisplayName','FCVP','Color',lblue,'LineWidth',1.2);
plot(t_gradEPI(1:end-3)*1000, 1000*grad_EPI_x_gstf(1,1:end-3).',':','DisplayName','GSTF prediction','Color',orange,'LineWidth',1.3);
legend('Position',[x1 a+dy-0.001 dx1 dy_l], 'numColumns',4,'FontName','Arial','Fontsize',8);
ylabel('Gradient (mT/m)','FontName','Arial','Fontsize',9.5);
rectangle('Position',[10 -45 4 90],'LineWidth',0.8,'EdgeColor',[1 1 1]*0.5);
text(14.5,-43,'(B)','FontName','Arial','Fontsize',8,'FontWeight','bold','Color',[1 1 1]*0.5);

text(-11.5,60,'(A)','FontName','Arial','Fontsize',11,'FontWeight','bold');
text(-11.5,-85,'(B)','FontName','Arial','Fontsize',11,'FontWeight','bold');
text(28,-85,'(C)','FontName','Arial','Fontsize',11,'FontWeight','bold');
text(65,-85,'(D)','FontName','Arial','Fontsize',11,'FontWeight','bold');
text(-11.5,-215,'(E)','FontName','Arial','Fontsize',11,'FontWeight','bold');
text(-11.5,-335,'(F)','FontName','Arial','Fontsize',11,'FontWeight','bold');
text(-11.5,-455,'(G)','FontName','Arial','Fontsize',11,'FontWeight','bold');
text(-11.5,-585,'(H)','FontName','Arial','Fontsize',11,'FontWeight','bold');
text(-11.5,-705,'(I)','FontName','Arial','Fontsize',11,'FontWeight','bold');

ax2 = subplot('Position',[x1 a-dy-d_y dx2 dy]);
plot(t_gradEPI(1:end-3)*1000, 1000*grad_EPI_x_VP(2,1:end-3).','DisplayName','VP','LineWidth',1.5,'Color',blue);
hold on;
plot(t_gradEPI(1:end-3)*1000, 1000*grad_EPI_x_CVP(2,1:end-3).','--','DisplayName','CVP','LineWidth',1.7,'Color',yellow);
plot(t_gradEPI(1:end-3)*1000, 1000*grad_EPI_x_FCVP(2,1:end-3).','-.','DisplayName','FCVP','Color',lblue,'LineWidth',1.2);
plot(t_gradEPI(1:end-3)*1000, 1000*grad_EPI_x_gstf(1,1:end-3).',':','DisplayName','GSTF prediction','Color',orange,'LineWidth',1.3);
xlim([10 14]);
ylim([-1 1]*45);
ylabel('Gradient (mT/m)','FontName','Arial','Fontsize',9.5);
rectangle('Position',[11.4 32.7 0.35 5],'LineWidth',0.8,'EdgeColor',[1 1 1]*0.5);
text(11.8,35.2,'(C)','FontName','Arial','Fontsize',8,'FontWeight','bold','Color',[1 1 1]*0.5);
rectangle('Position',[12.2 -37.7 0.35 5],'LineWidth',0.8,'EdgeColor',[1 1 1]*0.5);
text(12.6,-35.2,'(D)','FontName','Arial','Fontsize',8,'FontWeight','bold','Color',[1 1 1]*0.5);

ax3 = subplot('Position',[0.425 a-dy-d_y dx2 dy]);
plot(t_gradEPI(1:end-3)*1000, 1000*grad_EPI_x_VP(2,1:end-3).','DisplayName','VP','LineWidth',1.5,'Color',blue);
hold on;
plot(t_gradEPI(1:end-3)*1000, 1000*grad_EPI_x_CVP(2,1:end-3).','--','DisplayName','CVP','LineWidth',1.7,'Color',yellow);
plot(t_gradEPI(1:end-3)*1000, 1000*grad_EPI_x_FCVP(2,1:end-3).','-.','DisplayName','FCVP','Color',lblue,'LineWidth',1.2);
plot(t_gradEPI(1:end-3)*1000, 1000*grad_EPI_x_gstf(1,1:end-3).',':','DisplayName','GSTF prediction','Color',orange,'LineWidth',1.3);
xlim([11.4 11.75]);
ylim([-1 1]*0.5+35.2);

ax4 = subplot('Position',[0.75 a-dy-d_y dx2 dy]);
plot(t_gradEPI(1:end-3)*1000, 1000*grad_EPI_x_VP(2,1:end-3).','DisplayName','VP','LineWidth',1.5,'Color',blue);
hold on;
plot(t_gradEPI(1:end-3)*1000, 1000*grad_EPI_x_CVP(2,1:end-3).','--','DisplayName','CVP','LineWidth',1.7,'Color',yellow);
plot(t_gradEPI(1:end-3)*1000, 1000*grad_EPI_x_FCVP(2,1:end-3).','-.','DisplayName','FCVP','Color',lblue,'LineWidth',1.2);
plot(t_gradEPI(1:end-3)*1000, 1000*grad_EPI_x_gstf(1,1:end-3).',':','DisplayName','GSTF prediction','Color',orange,'LineWidth',1.3);
xlim([12.2 12.55]);
ylim([-1 1]*0.5-35.2);

ax5 = subplot('Position',[x1 a-1.75*dy-2*d_y-dy_l dx1 0.75*dy]);
plot(t_gradEPI(1:end-3)*1000, 1000*(grad_EPI_x_VP(2,1:end-3).'-grad_EPI_x_CVP(2,1:end-3).'),'DisplayName','VP-CVP','LineWidth',1.5);
legend('Position',[x1 a-dy-2*d_y-dy_l-0.001 dx1 dy_l], 'numColumns',1,'FontName','Arial','Fontsize',8);
ylim([-1 1]*0.028);
ylabel({'Gradient', 'difference', '(mT/m)'},'FontName','Arial','Fontsize',9.5);

ax6 = subplot('Position',[x1 a-2.5*dy-3*d_y-2*dy_l dx1 0.75*dy]);
plot(t_gradEPI(1:end-3)*1000, 1000*(grad_EPI_x_VP(2,1:end-3).'-grad_EPI_x_CVP(2,1:end-3).'),'DisplayName','VP-CVP','LineWidth',1.5);
hold on;
plot(t_gradEPI(1:end-3)*1000, 1000*weighted_sum_VPs(:,1:end-3).',':','DisplayName','weighted sum of prephasing gradients','LineWidth',1.5,'Color',green);
legend('Position',[x1 a-1.75*dy-3*d_y-2*dy_l-0.001 dx1 dy_l], 'numColumns',2,'FontName','Arial','Fontsize',8);
ylim([-1 1]*0.03);
ylabel({'Gradient', 'difference', '(mT/m)'},'FontName','Arial','Fontsize',9.5);
rectangle('Position',[0.2 -0.027 29.8 0.054],'LineWidth',0.8,'EdgeColor',[1 1 1]*0.5);
text(30.5,-0.02,'(G)','FontName','Arial','Fontsize',8,'FontWeight','bold','Color',[1 1 1]*0.5);

ax7 = subplot('Position',[x1 a-3.5*dy-4*d_y-2*dy_l dx1 dy]);
plot(t_gradEPI(1:end-3)*1000, 1000*(grad_EPI_x_VP(2,1:end-3).'-grad_EPI_x_CVP(2,1:end-3).'),'DisplayName','VP-CVP','LineWidth',1.5);
hold on;
plot(t_gradEPI(1:end-3)*1000, 1000*weighted_sum_VPs(:,1:end-3).',':','DisplayName','weighted sum of prephasing gradients','LineWidth',1.5,'Color',green);
xlim([0 30]);
ylim([-1 1]*0.028);
ylabel({'Gradient', 'difference', '(mT/m)'},'FontName','Arial','Fontsize',9.5);

ax8 = subplot('Position',[x1 a-4.25*dy-5*d_y-3*dy_l dx1 0.75*dy]);
plot(t_gradEPI(1:end-3)*1000, 1000*(grad_EPI_x_CVP(2,1:end-3).'-grad_EPI_x_FCVP(2,1:end-3).'),'DisplayName','CVP-FCVP','Color',violet,'LineWidth',1.5);
legend('Position',[x1 a-3.5*dy-5*d_y-3*dy_l-0.001 dx1 dy_l], 'numColumns',1,'FontName','Arial','Fontsize',8);
ylim([-1 1]*0.13);
ylabel({'Gradient', 'difference', '(mT/m)'},'FontName','Arial','Fontsize',9.5);
rectangle('Position',[10 -0.1 20 0.2],'LineWidth',0.8,'EdgeColor',[1 1 1]*0.5);
text(30.5,-0.09,'(I)','FontName','Arial','Fontsize',8,'FontWeight','bold','Color',[1 1 1]*0.5);

ax9 = subplot('Position',[x1 a-5.25*dy-6*d_y-3*dy_l dx1 dy]);
plot(t_gradEPI(1:end-3)*1000, 1000*(grad_EPI_x_CVP(2,1:end-3).'-grad_EPI_x_FCVP(2,1:end-3).'),'DisplayName','CVP-FCVP','Color',violet,'LineWidth',1.5);
xlim([10 30]);
xlabel('Time (ms)','FontName','Arial','Fontsize',9.5);
ylabel({'Gradient', 'difference', '(mT/m)'},'FontName','Arial','Fontsize',9.5);















    
    
