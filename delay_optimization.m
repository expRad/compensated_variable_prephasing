clear all;
% Change current directory to that of this .m file
mfile_name          = mfilename('fullpath');
if contains(mfile_name,'LiveEditorEvaluationHelper')
    mfile_name = matlab.desktop.editor.getActiveFilename;
end
[pathstr,name,ext]  = fileparts(mfile_name);
cd(pathstr);

%% Load results from multi delay reconstructions
path2save = '.\EPI_results\';
name2save = 'multiDelay_EPI_tra_peAP_FOVph80_CVP_new.mat';
data = load([path2save,name2save]);

% Load image data for plotting
img_largeDelay = load([path2save,'results_EPI_tra_peAP_FOVph80_tooLargeDelay.mat']);
img_smallDelay = load([path2save,'results_EPI_tra_peAP_FOVph80_tooSmallDelay.mat']);
img_optDelay = load([path2save,'results_EPI_tra_peAP_FOVph80.mat']);

%% Fit polynomial to costfunction-vs-delay for each acquisition
sum_out_del = squeeze(data.sum_out_del(1,1,:));
sum_out_girf = squeeze(data.sum_out_girf(1,1,:));
sum_out_meas = squeeze(data.sum_out_meas(1,1,:));
delays_del = data.delays_nom.';
delays_girf = data.delays_girf.';
delays_meas = data.delays_meas.';
x0 = data.x0;
y0 = data.y0;
r_out = data.r_out;

% Initial guess for the parameters
initial_guess = [1, 1, 1, 1, 1];
% polynomial of 4th order
quad_function = @(params, x) params(1)*x.^4 + params(2)*x.^3 + params(3)*x.^2 + params(4)*x + params(5);

% Fit the data using lsqcurvefit
params_del_sum = lsqcurvefit(quad_function, initial_guess, delays_del*1e5, sum_out_del);
params_girf_sum = lsqcurvefit(quad_function, initial_guess, delays_girf*1e5, sum_out_girf);
params_meas_sum = lsqcurvefit(quad_function, initial_guess, delays_meas*1e5, sum_out_meas);

% Find extrema of sum_out_del
disp('sum_out_del');
min_sum_del = find_minimum(params_del_sum);
disp(' ');
% Find extrema of sum_out_girf
disp('sum_out_girf');
min_sum_girf = find_minimum(params_girf_sum);
disp(' ');
% Find extrema of sum_out_meas
disp('sum_out_meas');
min_sum_meas = find_minimum(params_meas_sum);
disp(' ');

delays_del_fine = linspace(delays_del(1),delays_del(end), 200)*1e5;
delays_girf_fine = linspace(delays_girf(1),delays_girf(end), 200)*1e5;
delays_meas_fine = linspace(delays_meas(1),delays_meas(end), 200)*1e5;

fit_del_sum = quad_function(params_del_sum, delays_del_fine);
fit_girf_sum = quad_function(params_girf_sum, delays_girf_fine);
fit_meas_sum = quad_function(params_meas_sum, delays_meas_fine);

delays_del_fine = delays_del_fine*1e-5;
delays_girf_fine = delays_girf_fine*1e-5;
delays_meas_fine = delays_meas_fine*1e-5;

%% Define mask and prepare images for plotting
reco_girf_sd = squeeze(img_smallDelay.sum_images_girf_dc(:,:,1,1));
reco_girf_ld = squeeze(img_largeDelay.sum_images_girf_dc(:,:,1,1));
reco_girf_od = squeeze(img_optDelay.sum_images_girf_dc(:,:,1,1));

[xx,yy] = meshgrid((1:size(reco_girf_od,2)),(1:size(reco_girf_od,1)));
mask_out = double(sqrt((xx-x0).^2+(yy-y0).^2)>r_out);

nRO = size(reco_girf_od,1);
nFillImage = nRO/2 - size(reco_girf_od,2);
x0_plot = x0 + nFillImage/2;
y0_plot = y0 - nRO/4;

reco_girf_sd_plot = zeros(nRO/2);
reco_girf_ld_plot = zeros(nRO/2);
reco_girf_od_plot = zeros(nRO/2);

reco_girf_sd_plot(:,nFillImage/2+1:end-nFillImage/2) = reco_girf_sd(nRO/4+1:end-nRO/4,:);
reco_girf_ld_plot(:,nFillImage/2+1:end-nFillImage/2) = reco_girf_ld(nRO/4+1:end-nRO/4,:);
reco_girf_od_plot(:,nFillImage/2+1:end-nFillImage/2) = reco_girf_od(nRO/4+1:end-nRO/4,:);

% Changes to make the mask visible in two different ways:
% reco_girf_sd_plot(:,nFillImage/2+1:end-nFillImage/2) = reco_girf_sd(nRO/4+1:end-nRO/4,:)+mask_out(nRO/4+1:end-nRO/4,:)*1000;
% reco_girf_ld_plot(:,nFillImage/2+1:end-nFillImage/2) = reco_girf_ld(nRO/4+1:end-nRO/4,:)+mask_out(nRO/4+1:end-nRO/4,:)*1000;
% reco_girf_od_plot(:,nFillImage/2+1:end-nFillImage/2) = reco_girf_od(nRO/4+1:end-nRO/4,:)+mask_out(nRO/4+1:end-nRO/4,:)*1000;

% reco_girf_sd_plot(:,nFillImage/2+1:end-nFillImage/2) = reco_girf_sd(nRO/4+1:end-nRO/4,:).*(mask_out(nRO/4+1:end-nRO/4,:)+0);
% reco_girf_ld_plot(:,nFillImage/2+1:end-nFillImage/2) = reco_girf_ld(nRO/4+1:end-nRO/4,:).*(mask_out(nRO/4+1:end-nRO/4,:)+0);
% reco_girf_od_plot(:,nFillImage/2+1:end-nFillImage/2) = reco_girf_od(nRO/4+1:end-nRO/4,:).*(mask_out(nRO/4+1:end-nRO/4,:)+0);

%% Define colors
blue = [0 0.4470 0.7410];
orange = [0.8500 0.3250 0.0980];
violet = [0.4940 0.1840 0.5560];
green = [0.4660 0.6740 0.1880];
yellow = [0.9290 0.6940 0.1250];
lblue = [0.3010 0.7450 0.9330];
dred = [0.6350 0.0780 0.1840];

%% Plot
    
figure('Units','centimeters','Position',[5 5 17.56 14]);
ax1 = subplot(2,2,1);
plot(delays_girf*1e6, sum_out_girf, 'o', 'DisplayName','summed intensity (GSTF)');
hold on;
plot(delays_girf_fine*1e6, fit_girf_sum, '-', 'DisplayName','fit (GSTF)','LineWidth',1.2,'Color',blue);
xline(min_sum_girf*1e6,'-.', 'DisplayName','optimum delay for GSTF correction','LineWidth',1.2);
ylabel('summed signal intensity (a.u.)');
xlabel('Delay (\mus)');
leg = legend('Location','northoutside');
leg.ItemTokenSize = [17 5];
ylim([0.4 1.2]);
set(gca,'FontName','Times','Fontsize',9);
title('(A)');

ax2 = subplot(2,2,2);
plot(delays_meas*1e6, sum_out_meas, 'o', 'DisplayName','summed intensity (CVP)','Color',orange);
hold on;
plot(delays_meas_fine*1e6, fit_meas_sum, '-', 'DisplayName','fit (CVP)','LineWidth',1.2,'Color',orange);
xline(min_sum_meas*1e6,'--', 'DisplayName','optimum delay for CVP correction','LineWidth',1.2);
leg = legend('Location','northoutside');
leg.ItemTokenSize = [17 5];
ylim([0.4 1.2]);
xlabel('Delay (\mus)');
set(gca,'FontName','Times','Fontsize',9);
title('(B)');

ax3 = subplot(2,3,4);
imagesc(reco_girf_sd_plot,[0 Inf]);
colormap(ax3, 'gray');
hold on;
rectangle('Position',[x0_plot-r_out y0_plot-r_out 2*r_out 2*r_out],'Curvature',[1,1],'EdgeColor',green,'LineWidth',1.5);
axis image; axis off;
title('(C)');

ax4 = subplot(2,3,5);
imagesc(reco_girf_od_plot,[0 Inf]);
colormap(ax4, 'gray');
rectangle('Position',[x0_plot-r_out y0_plot-r_out 2*r_out 2*r_out],'Curvature',[1,1],'EdgeColor',green,'LineWidth',1.5);
axis image; axis off;
title('(D)');

ax5 = subplot(2,3,6);
imagesc(reco_girf_ld_plot,[0 Inf]);
colormap(ax5, 'gray');
rectangle('Position',[x0_plot-r_out y0_plot-r_out 2*r_out 2*r_out],'Curvature',[1,1],'EdgeColor',green,'LineWidth',1.5);
axis image; axis off;
title('(E)');



%%
function minimum = find_minimum(params)
extrema = roots([4*params(1) 3*params(2) 2*params(3) params(4)]);
for j=1:3
    if imag(extrema(j))==0
        if 12*params(1)*extrema(j)^2 + 6*params(2)*extrema(j) + 2*params(3) > 0
            disp(['Min = ',num2str(extrema(j)*1e-5)]);
            minimum = extrema(j)*1e-5;
        end
    end
end
end





