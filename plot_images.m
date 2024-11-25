% Copyright (c) 2024 Hannah Scholten

clear all;
% Change current directory to that of this .m file
mfile_name          = mfilename('fullpath');
if contains(mfile_name,'LiveEditorEvaluationHelper')
    mfile_name = matlab.desktop.editor.getActiveFilename;
end
[pathstr,name,ext]  = fileparts(mfile_name);
cd(pathstr);
% Add necessary scripts to the MATLAB path
folder1 = ['.' filesep 'helper_functions' filesep];
addpath(folder1);

%% Load images and trajectory data
img_tra_peAP = load(['.' filesep 'EPI_results' filesep 'results_EPI_tra_peAP_FOVph80_new.mat']);

nRO = img_tra_peAP.nRO;
kRO_navi = img_tra_peAP.kRO_navi;
kRO_girf = img_tra_peAP.kRO_girf;
kRO_meas_CCoff = img_tra_peAP.kRO_meas_CCoff;
kRO_meas_CCon = img_tra_peAP.kRO_meas_CCon;
kRO_meas_VP = img_tra_peAP.kRO_meas_VP;
kPE_del = img_tra_peAP.kPE_del;
kPE_navi = img_tra_peAP.kPE_navi;
kPE_girf = img_tra_peAP.kPE_girf;
max_k = img_tra_peAP.max_k;

%%
img_test_navi = squeeze(img_tra_peAP.sum_images_navi_dc(end-nRO/4:-1:nRO/4+1,:,1,1));
img_test_girf = squeeze(img_tra_peAP.sum_images_girf_dc(nRO/4+1:end-nRO/4,:,1,1));
img_test_CVP = squeeze(img_tra_peAP.sum_images_meas_CCoff_dc(nRO/4+1:end-nRO/4,:,1,1));
img_test_VP = squeeze(img_tra_peAP.sum_images_meas_VP_dc(nRO/4+1:end-nRO/4,:,1,1));
img_test_FCVP = squeeze(img_tra_peAP.sum_images_meas_CCon_dc(nRO/4+1:end-nRO/4,:,1,1));

img_test_navi_shift = circshift(img_test_navi,68,2);
img_test_girf_shift = circshift(img_test_girf,68,2);
img_test_CVP_shift = circshift(img_test_CVP,68,2);
img_test_VP_shift = circshift(img_test_VP,68,2);
img_test_FCVP_shift = circshift(img_test_FCVP,68,2);

% Load data with mask information
data4mask = load(['.' filesep 'EPI_results' filesep 'multiDelay_EPI_tra_peAP_FOVph80_CVP_new.mat']);
x0 = data4mask.x0;
y0 = data4mask.y0;
r_out = data4mask.r_out;
[xx,yy] = meshgrid((1:size(img_test_navi,2)),(1:size(img_test_navi,1)));
mask_out = double(sqrt((xx-x0).^2+(yy-y0+nRO/4).^2)>r_out); % mask covering the object

% Test ROI for ghost quantification
x1 = 23;
y1 = 141;
r1 = 5;
ROIghost = double(sqrt((xx-x1).^2+(yy-y1).^2)<r1);

%% Arrange for plotting
nFillImage = nRO/2 - size(img_tra_peAP.sum_images_nom_dc,2);

img_tra_peAP_navi = zeros(nRO/2,nRO/2);
img_tra_peAP_girf = zeros(nRO/2,nRO/2);
img_tra_peAP_meas_CCoff = zeros(nRO/2,nRO/2);
img_tra_peAP_meas_VP = zeros(nRO/2,nRO/2);
img_tra_peAP_meas_CCon = zeros(nRO/2,nRO/2);
mask_out_filled = zeros(nRO/2,nRO/2);

img_tra_peAP_navi(:,nFillImage/2+1:end-nFillImage/2) = squeeze(img_tra_peAP.sum_images_navi_dc(end-nRO/4:-1:nRO/4+1,:,1,1));
img_tra_peAP_girf(:,nFillImage/2+1:end-nFillImage/2) = squeeze(img_tra_peAP.sum_images_girf_dc(nRO/4+1:end-nRO/4,:,1,1));
img_tra_peAP_meas_CCoff(:,nFillImage/2+1:end-nFillImage/2) = squeeze(img_tra_peAP.sum_images_meas_CCoff_dc(nRO/4+1:end-nRO/4,:,1,1));
img_tra_peAP_meas_VP(:,nFillImage/2+1:end-nFillImage/2) = squeeze(img_tra_peAP.sum_images_meas_VP_dc(nRO/4+1:end-nRO/4,:,1,1));
img_tra_peAP_meas_CCon(:,nFillImage/2+1:end-nFillImage/2) = squeeze(img_tra_peAP.sum_images_meas_CCon_dc(nRO/4+1:end-nRO/4,:,1,1));

mask_out_filled(:,nFillImage/2+1:end-nFillImage/2) = mask_out;

%% Define ROIs for ghost quantification
x1 = 27+nFillImage/2;
y1 = 144;
r1 = 5;
[xx,yy] = meshgrid((1:nRO/2),(1:nRO/2));
ROI1 = double(sqrt((xx-x1).^2+(yy-y1).^2)<r1);

x2 = x1 + size(img_tra_peAP.sum_images_nom_dc,2)/2;
y2 = y1;
r2 = r1;
ROI2 = double(sqrt((xx-x2).^2+(yy-y2).^2)<r2);

% figure;
% subplot(1,2,1);
% % imagesc(img_tra_peAP_navi+500.*(ROI1 + ROI2));
% imagesc(img_tra_peAP_girf+500.*(ROI1 + ROI2));
% colormap('gray');
% 
% subplot(1,2,2);
% % imagesc((img_tra_peAP_navi+500.*(ROI1 + ROI2)).*mask_out_filled);
% imagesc((img_tra_peAP_girf+500.*(ROI1 + ROI2)).*mask_out_filled);
% colormap('gray');

ghost_level_navi = max(img_tra_peAP_navi.*ROI1,[],'all')/max(img_tra_peAP_navi.*ROI2,[],'all');
ghost_level_girf = max(img_tra_peAP_girf.*ROI1,[],'all')/max(img_tra_peAP_girf.*ROI2,[],'all');
ghost_level_CVP = max(img_tra_peAP_meas_CCoff.*ROI1,[],'all')/max(img_tra_peAP_meas_CCoff.*ROI2,[],'all');
ghost_level_VP = max(img_tra_peAP_meas_VP.*ROI1,[],'all')/max(img_tra_peAP_meas_VP.*ROI2,[],'all');
ghost_level_FCVP = max(img_tra_peAP_meas_CCon.*ROI1,[],'all')/max(img_tra_peAP_meas_CCon.*ROI2,[],'all');

fprintf(['ghost_level_navi = ',num2str(ghost_level_navi),'\n\n']);
fprintf(['ghost_level_gstf = ',num2str(ghost_level_girf),'\n\n']);
fprintf(['ghost_level_CVP = ',num2str(ghost_level_CVP),'\n\n']);
fprintf(['ghost_level_VP = ',num2str(ghost_level_VP),'\n\n']);
fprintf(['ghost_level_FCVP = ',num2str(ghost_level_FCVP),'\n\n']);

%% Define colors
blue = [0 0.4470 0.7410];
orange = [0.8500 0.3250 0.0980];
violet = [0.4940 0.1840 0.5560];
green = [0.4660 0.6740 0.1880];
yellow = [0.9290 0.6940 0.1250];
lblue = [0.3010 0.7450 0.9330];
dred = [0.6350 0.0780 0.1840];

%% Transverse images (navigator, GSTF, and CVP correction)
h = 23;
figure('Unit','centimeter','Position',[0.5,0.5,17.56,h],'Color','black');

clim1 = [0 2000];

dx1 = 6.5/17.56;
dy1 = 6.5/h;
d_y = 0.02;
d_x = 0.05;
a = 0.71;

ax1 = subplot('Position',[0.02 a dx1 dy1],'Color','black');
imagesc(ax1,img_tra_peAP_navi, clim1);
colormap(ax1, 'gray');
axis image;
axis off;
text(0,5,'(A)','FontName','Arial','Fontsize',11,'FontWeight','bold','Color','white');
text(85,10,['phase correction with' newline 'reference echoes (\Gamma_r_e_f = ',num2str(round(ghost_level_navi,3)*100),' %)'],'FontName','Arial','Fontsize',9,'FontWeight','bold','Color','white','HorizontalAlignment','center');
rectangle('Position',[x1-r1 y1-r1 2*r1 2*r1],'Curvature',[1,1],'EdgeColor',blue,'LineWidth',1.5);
rectangle('Position',[x2-r2 y2-r2 2*r2 2*r2],'Curvature',[1,1],'EdgeColor',orange,'LineWidth',1.5);
text(x1,y1+3*r1, 'ROI 1','FontName','Arial','FontSize',9,'Color',blue,'HorizontalAlignment','center');
text(x2,y2+3*r2, 'ROI 2','FontName','Arial','FontSize',9,'Color',orange,'HorizontalAlignment','center');

ax2 = subplot('Position',[0.02+dx1+d_x a dx1 dy1]);
imagesc(ax2,img_tra_peAP_girf, clim1);
colormap(ax2, 'gray');
axis image;
axis off;
cb = colorbar('Position',[0.02+dx1+1.5*d_x+0.4 a-dy1-d_y/2 0.04 2*dy1],'Color','white');
ylabel(cb,'image intensity (a.u.)','Rotation',270,'FontName','Arial','Fontsize',9);
text(0,5,'(B)','FontName','Arial','Fontsize',11,'FontWeight','bold','Color','white');
text(85,10,['trajectory correction' newline 'with GSTF (\Gamma_G_S_T_F = ',num2str(round(ghost_level_girf,3)*100),' %)'],'FontName','Arial','Fontsize',9,'FontWeight','bold','Color','white','HorizontalAlignment','center');

ax3 = subplot('Position',[0.02 a-dy1-d_y dx1 dy1]);
imagesc(ax3,img_tra_peAP_meas_VP, clim1);
colormap(ax3, 'gray');
axis image;
axis off;
text(0,5,'(C)','FontName','Arial','Fontsize',11,'FontWeight','bold','Color','white');
text(85,10,['readout gradient measured' newline 'with VP (\Gamma_V_P = ',num2str(round(ghost_level_VP,3)*100),' %)'],'FontName','Arial','Fontsize',9,'FontWeight','bold','Color','white','HorizontalAlignment','center');

ax4 = subplot('Position',[0.02+dx1+d_x a-dy1-d_y dx1 dy1]);
imagesc(ax4,img_tra_peAP_meas_CCoff, clim1);
colormap(ax4, 'gray');
axis image;
axis off;
text(0,5,'(D)','FontName','Arial','Fontsize',11,'FontWeight','bold','Color','white');
text(85,10,['readout gradient measured' newline 'with CVP (\Gamma_C_V_P = ',num2str(round(ghost_level_CVP,3)*100),' %)'],'FontName','Arial','Fontsize',9,'FontWeight','bold','Color','white','HorizontalAlignment','center');

ax5 = subplot('Position',[0.02 a-2*dy1-2*d_y dx1 dy1]);
imagesc(ax5,img_tra_peAP_meas_CCon, clim1);
colormap(ax5, 'gray');
axis image;
axis off;
text(0,5,'(E)','FontName','Arial','Fontsize',11,'FontWeight','bold','Color','white');
text(170,5,'(F)','FontName','Arial','Fontsize',11,'FontWeight','bold','Color','white');
text(85,10,['readout gradient measured' newline 'with FCVP (\Gamma_F_C_V_P = ',num2str(round(ghost_level_FCVP,3)*100),' %)'],'FontName','Arial','Fontsize',9,'FontWeight','bold','Color','white','HorizontalAlignment','center');

ax6 = subplot('Position',[0.02+dx1+1.5*d_x 0.05 0.36 0.34]);
plot(reshape(kRO_navi(nRO/2+1,nFillImage+1:end)+1i*kPE_navi(nRO/2+1,nFillImage+1:end),[],1),'Marker','o','LineStyle','None','DisplayName',['nominal' newline 'trajectory'],'Color',dred,'MarkerSize',2,'MarkerFaceColor',dred);
hold on; grid on;
plot(reshape(kRO_girf(nRO/2+1,:)+1i*kPE_girf(nRO/2+1,:),[],1)/max_k,'Marker','*','LineStyle','None','DisplayName',['GSTF-' newline 'corrected'],'Color',orange,'MarkerSize',4);
plot(reshape(kRO_meas_VP(nRO/2+1,:)+1i*kPE_del(nRO/2+1,:),[],1)/max_k,'Marker','square','LineStyle','None','DisplayName',['measured' newline 'with VP'],'Color',blue,'MarkerSize',4,'MarkerFaceColor',blue);
plot(reshape(kRO_meas_CCoff(nRO/2+1,:)+1i*kPE_del(nRO/2+1,:),[],1)/max_k,'Marker','+','LineStyle','None','DisplayName',['measured' newline 'with CVP'],'Color',yellow,'MarkerSize',5,'MarkerFaceColor',yellow,'LineWidth',1.2);
plot(reshape(kRO_meas_CCon(nRO/2+1,:)+1i*kPE_del(nRO/2+1,:),[],1)/max_k,'Marker','diamond','LineStyle','None','DisplayName',['measured' newline 'with FCVP'],'Color',lblue,'MarkerSize',3,'MarkerFaceColor',lblue);
xlabel('normalized k_R_O (1)','Color','white');
ylabel('normalized k_P_E (1)','Color','white');
legend('Position',[0.02+dx1+1.5*d_x+0.4 0.05 0.1 0.34]);
xlim([-1 1]*0.03);
ylim([-1.1 0.6]);
ax6.XColor = 'white';
ax6.YColor = 'white';
ax6.GridColor = 'black';
ax6.GridAlpha = 0.3; 
set(gca,'FontName','Arial','Fontsize',9);

set(gcf, 'InvertHardcopy', 'off');









