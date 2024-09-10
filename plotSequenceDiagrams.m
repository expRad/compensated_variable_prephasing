%% rf-pulse
x_rf = linspace(-0.5,0.5,100);
y_rf = sin(x_rf*19)./(x_rf*19);
offset_rf = 2.5;
offset_rf2 = -5.6;
offset_rf3 = -13.4;
offset_rf4 = -21.5;

%% slice-selection gradient
x_ss = [-0.5, -0.4, 0.4, 0.5, 0.6, 0.9, 1, 1.25];
y_ss = [0,    1,    1,   0,   -1,   -1, 0, 0];
offset_ss = 0;
offset_ss2 = -8.6;
offset_ss3 = -15.9;
offset_ss4 = -24.5;

%% variable-prephasing gradient
x_vp = [0, 0.1, 0.65, 0.75, 1];
y_vp = [0, 1,   1,    0,    0];

%% trapezoidal test gradient
x_tg = [0, 0.5, 0.9, 1.6, 2];
y_tg = [0, 0,   1.5, 1.5, 0];

%% rectangle
x_rect = [-4, -4, 14, 14, -4]-1;
y_rect = [-3,4.5,4.5,  -3, -3];
y_rect2 = [-2,4.9,4.9,  -2, -2];

%%
lw = 1.5;
lw2 = 1.2;

fig = figure('Units','centimeters', 'InnerPosition', [0 0 17.56 18]);

ax1 = subplot('Position',[0.03 0.01 0.45 0.98]);
hold on;
% RF line 1
plot([-1.5, -0.5],[0,0]+offset_rf, 'LineWidth',lw,'Color','k');
plot(x_rf, y_rf+offset_rf, 'LineWidth',lw,'Color','k');
plot([0.5,12],[0,0]+offset_rf, 'LineWidth',lw,'Color','k');
x = x_ss(end)-x_ss(1)+x_vp(end)+x_ss(end);
rectangle('Position',[x, offset_rf-0.5, 7.5, 1], 'FaceColor','k');
text(x+3.75, offset_rf, 'ADC', 'Color','w','FontWeight','bold','HorizontalAlignment','center');
% Gradient line 1
plot([-1.5, -0.5],[0,0]+offset_ss, 'LineWidth',lw,'Color','k');
plot(x_ss, y_ss+offset_ss, 'LineWidth',lw,'Color','k');
plot(x_vp+x_ss(end), -y_vp*0.5+offset_ss,':', 'LineWidth',lw,'Color','k');
plot(x_vp+x_ss(end), zeros(size(x_vp))*1.0+offset_ss,':', 'LineWidth',lw,'Color','k');
plot(x_vp+x_ss(end), -y_vp*1.0+offset_ss,':', 'LineWidth',lw,'Color','k');
plot(x_vp+x_ss(end), -y_vp*1.5+offset_ss, 'LineWidth',lw,'Color','k');
plot(x_vp+x_ss(end), -y_vp*2.0+offset_ss,':', 'LineWidth',lw,'Color','k');
plot(x_vp+x_ss(end), -y_vp*2.5+offset_ss,':', 'LineWidth',lw,'Color','k');
x = x_vp(end)+x_ss(end);
text(x, -y_vp(3)*1.0+offset_ss, 'variable-prephasing');
text(x, -y_vp(3)*2.0+offset_ss, 'gradient');
rectangle('Position',[x_ss(end)-0.1 offset_ss-y_vp(3)*2.5-0.2 0.95 2.9],'LineStyle','none','FaceColor', [0, 0, 0, 0.2]);
rectangle('Position',[x_ss(end)-0.1+0.95 offset_ss-y_vp(3)*2.5-0.2 8.5 2.2],'LineStyle','none','FaceColor', [0, 0, 0, 0.2]);
plot(x_ss-x_ss(1)+x, zeros(size(x_ss))+offset_ss, 'LineWidth',lw,'Color','k');
x = x_ss(end)-x_ss(1)+x;
plot(x_tg+x, y_tg+offset_ss, 'LineWidth',lw,'Color','k');
x = x_tg(end)+x;
text(x-0.1, y_tg(3)+offset_ss-0.1, 'test gradient');
rectangle('Position',[x-1.6 offset_ss-0.2 1.7 2.1],'LineStyle','none','FaceColor', [0, 0, 0, 0.2]);
rectangle('Position',[x+0.1 offset_ss+0.7 5.2 1.2],'LineStyle','none','FaceColor', [0, 0, 0, 0.2]);
plot([x, 12],[0,0]+offset_ss, 'LineWidth',lw,'Color','k');
% rectangle and text 1
plot(x_rect, y_rect+offset_ss,'--', 'LineWidth',lw2,'Color','k');
text(-4.3, offset_rf, 'RF');
text(-4.3, offset_ss, 'G_s_l_i_c_e');
text(-4.5, offset_rf+1.3, '(1)', 'FontWeight','bold','FontSize',11);
% RF line 2
plot([-1.5, -0.5],[0,0]+offset_rf2, 'LineWidth',lw,'Color','k');
plot(x_rf, y_rf+offset_rf2, 'LineWidth',lw,'Color','k');
plot([0.5,12],[0,0]+offset_rf2, 'LineWidth',lw,'Color','k');
x = x_ss(end)-x_ss(1)+x_vp(end)+x_ss(end);
rectangle('Position',[x, offset_rf2-0.5, 7.5, 1], 'FaceColor','k');
text(x+3.75, offset_rf2, 'ADC', 'Color','w','FontWeight','bold','HorizontalAlignment','center');
% Gradient line 2
plot([-1.5, -0.5],[0,0]+offset_ss2, 'LineWidth',lw,'Color','k');
plot(x_ss, y_ss+offset_ss2, 'LineWidth',lw,'Color','k');
x = x_ss(end);
plot([x, 12],[0,0]+offset_ss2, 'LineWidth',lw,'Color','k');
% rectangle and text 2
plot(x_rect, y_rect2+offset_ss2,'-.', 'LineWidth',lw2,'Color','k');
text(-4.3, offset_rf2, 'RF');
text(-4.3, offset_ss2, 'G_s_l_i_c_e');
text(-4.5, offset_rf2+1.3, '(2)', 'FontWeight','bold','FontSize',11);
% RF line 3
plot([-1.5, x_vp(end)+x_ss(end)],[0,0]+offset_rf3, 'LineWidth',lw,'Color','k');
plot(x_rf-x_rf(1)+x_vp(end)+x_ss(end), y_rf+offset_rf3, 'LineWidth',lw,'Color','k');
plot([x_rf(end)-x_rf(1)+x_vp(end)+x_ss(end),12],[0,0]+offset_rf3, 'LineWidth',lw,'Color','k');
x = x_ss(end)-x_ss(1)+x_vp(end)+x_ss(end);
rectangle('Position',[x, offset_rf3-0.5, 7.5, 1], 'FaceColor','k');
text(x+3.75, offset_rf3, 'ADC', 'Color','w','FontWeight','bold','HorizontalAlignment','center');
% Gradient line 3
plot([-1.5, x_ss(end)],[0,0]+offset_ss3, 'LineWidth',lw,'Color','k');
plot(x_vp+x_ss(end), -y_vp*0.5+offset_ss3,':', 'LineWidth',lw,'Color','k');
plot(x_vp+x_ss(end), zeros(size(x_vp))*1.0+offset_ss3,':', 'LineWidth',lw,'Color','k');
plot(x_vp+x_ss(end), -y_vp*1.0+offset_ss3,':', 'LineWidth',lw,'Color','k');
plot(x_vp+x_ss(end), -y_vp*1.5+offset_ss3, 'LineWidth',lw,'Color','k');
plot(x_vp+x_ss(end), -y_vp*2.0+offset_ss3,':', 'LineWidth',lw,'Color','k');
plot(x_vp+x_ss(end), -y_vp*2.5+offset_ss3,':', 'LineWidth',lw,'Color','k');
x = x_vp(end)+x_ss(end);
plot(x_ss-x_ss(1)+x, y_ss+offset_ss3, 'LineWidth',lw,'Color','k');
x = x_ss(end)-x_ss(1)+x;
plot([x, 12],[0,0]+offset_ss3, 'LineWidth',lw,'Color','k');
% rectangle and text 3
plot(x_rect, y_rect+offset_ss3,'-.', 'LineWidth',lw2,'Color','k');
text(-4.3, offset_rf3, 'RF');
text(-4.3, offset_ss3, 'G_s_l_i_c_e');
text(-4.5, offset_rf3+1.3, '(3)', 'FontWeight','bold','FontSize',11);
% RF line 4
plot([-1.5, x_vp(end)+x_ss(end)],[0,0]+offset_rf4, 'LineWidth',lw,'Color','k');
plot(x_rf-x_rf(1)+x_vp(end)+x_ss(end), y_rf+offset_rf4, 'LineWidth',lw,'Color','k');
plot([x_rf(end)-x_rf(1)+x_vp(end)+x_ss(end),12],[0,0]+offset_rf4, 'LineWidth',lw,'Color','k');
x = x_ss(end)-x_ss(1)+x_vp(end)+x_ss(end);
rectangle('Position',[x, offset_rf4-0.5, 7.5, 1], 'FaceColor','k');
text(x+3.75, offset_rf4, 'ADC', 'Color','w','FontWeight','bold','HorizontalAlignment','center');
% Gradient line 4
plot([-1.5, x_ss(end)+x_vp(end)],[0,0]+offset_ss4, 'LineWidth',lw,'Color','k');
x = x_vp(end)+x_ss(end);
plot(x_ss-x_ss(1)+x, y_ss+offset_ss4, 'LineWidth',lw,'Color','k');
x = x_ss(end)-x_ss(1)+x;
plot([x, 12],[0,0]+offset_ss4, 'LineWidth',lw,'Color','k');
% rectangle and text 4
plot(x_rect, y_rect2+offset_ss4,'-.', 'LineWidth',lw2,'Color','k');
text(-4.3, offset_rf4, 'RF');
text(-4.3, offset_ss4, 'G_s_l_i_c_e');
text(-4.5, offset_rf4+1.3, '(4)', 'FontWeight','bold','FontSize',11);
% other stuff
plot([-3.75, -4.5, -4.5, -3.75]-1, [5.5, 5.5, -2.5+offset_ss4, -2.5+offset_ss4], 'LineWidth',lw2,'Color','k');
plot([13.75, 14.5, 14.5, 13.75]-1, [5.5, 5.5, -2.5+offset_ss4, -2.5+offset_ss4], 'LineWidth',lw2,'Color','k');
text(-4.5, 5.5, 'for n = 1 ... N (variable-prephasing steps)');
plot([-4, -5, -5, -4]-1, [6.5, 6.5, -3+offset_ss4, -3+offset_ss4], 'LineWidth',lw2,'Color','k');
plot([14, 15, 15, 14]-1, [6.5, 6.5, -3+offset_ss4, -3+offset_ss4], 'LineWidth',lw2,'Color','k');
text(-4.5, 6.5, 'for m = 1 ... M (slices)');
xlim([x_rect(1)-1.2 x_rect(3)+1.2]);
ylim([y_rect2(1)+offset_ss4-1.2 y_rect(2)+offset_ss+3.2]);
text(-7, 7.5, '(A)', 'FontWeight','bold', 'FontSize',12);
set(gca,'XColor','none','YColor','none','TickDir','out');

ax2 = subplot('Position',[0.53 0.01 0.45 0.98]);
hold on;
% RF line 1
plot([-1.5, -0.5],[0,0]+offset_rf, 'LineWidth',lw,'Color','k');
plot(x_rf, y_rf+offset_rf, 'LineWidth',lw,'Color','k');
plot([0.5,12],[0,0]+offset_rf, 'LineWidth',lw,'Color','k');
x = x_ss(end)-x_ss(1)+x_vp(end)+x_ss(end);
rectangle('Position',[x, offset_rf-0.5, 7.5, 1], 'FaceColor','k');
text(x+3.75, offset_rf, 'ADC', 'Color','w','FontWeight','bold','HorizontalAlignment','center');
% Gradient line 1
plot([-1.5, -0.5],[0,0]+offset_ss, 'LineWidth',lw,'Color','k');
plot(x_ss, y_ss+offset_ss, 'LineWidth',lw,'Color','k');
plot(x_vp+x_ss(end), -y_vp*0.5+offset_ss,':', 'LineWidth',lw,'Color','k');
plot(x_vp+x_ss(end), zeros(size(x_vp))*1.0+offset_ss,':', 'LineWidth',lw,'Color','k');
plot(x_vp+x_ss(end), -y_vp*1.0+offset_ss,':', 'LineWidth',lw,'Color','k');
plot(x_vp+x_ss(end), -y_vp*1.5+offset_ss, 'LineWidth',lw,'Color','k');
plot(x_vp+x_ss(end), -y_vp*2.0+offset_ss,':', 'LineWidth',lw,'Color','k');
plot(x_vp+x_ss(end), -y_vp*2.5+offset_ss,':', 'LineWidth',lw,'Color','k');
x = x_vp(end)+x_ss(end);
plot(x_ss-x_ss(1)+x, zeros(size(x_ss))+offset_ss, 'LineWidth',lw,'Color','k');
x = x_ss(end)-x_ss(1)+x;
plot(x_tg+x, y_tg+offset_ss, 'LineWidth',lw,'Color','k');
x = x_tg(end)+x;
plot([x, 12],[0,0]+offset_ss, 'LineWidth',lw,'Color','k');
% rectangle and text 1
plot(x_rect, y_rect+offset_ss,'--', 'LineWidth',lw2,'Color','k');
text(-4.3, offset_rf, 'RF');
text(-4.3, offset_ss, 'G_s_l_i_c_e');
text(-4.5, offset_rf+1.3, '(1)', 'FontWeight','bold','FontSize',11);
% RF line 2
plot([-1.5, -0.5],[0,0]+offset_rf2, 'LineWidth',lw,'Color','k');
plot(x_rf, y_rf+offset_rf2, 'LineWidth',lw,'Color','k');
plot([0.5,12],[0,0]+offset_rf2, 'LineWidth',lw,'Color','k');
x = x_ss(end)-x_ss(1)+x_vp(end)+x_ss(end);
rectangle('Position',[x, offset_rf2-0.5, 7.5, 1], 'FaceColor','k');
text(x+3.75, offset_rf2, 'ADC', 'Color','w','FontWeight','bold','HorizontalAlignment','center');
% Gradient line 2
plot([-1.5, -0.5],[0,0]+offset_ss2, 'LineWidth',lw,'Color','k');
plot(x_ss, y_ss+offset_ss2, 'LineWidth',lw,'Color','k');
plot(x_vp+x_ss(end), y_vp*0.5+offset_ss2,':', 'LineWidth',lw,'Color','k');
plot(x_vp+x_ss(end), zeros(size(x_vp))*1.0+offset_ss2,':', 'LineWidth',lw,'Color','k');
plot(x_vp+x_ss(end), y_vp*1.0+offset_ss2,':', 'LineWidth',lw,'Color','k');
plot(x_vp+x_ss(end), y_vp*1.5+offset_ss2, 'LineWidth',lw,'Color','k');
plot(x_vp+x_ss(end), y_vp*2.0+offset_ss2,':', 'LineWidth',lw,'Color','k');
plot(x_vp+x_ss(end), y_vp*2.5+offset_ss2,':', 'LineWidth',lw,'Color','k');
x = x_vp(end)+x_ss(end);
plot(x_ss-x_ss(1)+x, zeros(size(x_ss))+offset_ss2, 'LineWidth',lw,'Color','k');
x = x_ss(end)-x_ss(1)+x;
plot(x_tg+x, -y_tg+offset_ss2, 'LineWidth',lw,'Color','k');
x = x_tg(end)+x;
plot([x, 12],[0,0]+offset_ss2, 'LineWidth',lw,'Color','k');
% rectangle and text 2
plot(x_rect, y_rect2+offset_ss2,'-.', 'LineWidth',lw2,'Color','k');
text(-4.3, offset_rf2, 'RF');
text(-4.3, offset_ss2, 'G_s_l_i_c_e');
text(-4.5, offset_rf2+1.3, '(2)', 'FontWeight','bold','FontSize',11);
% RF line 3
plot([-1.5, x_vp(end)+x_ss(end)],[0,0]+offset_rf3, 'LineWidth',lw,'Color','k');
plot(x_rf-x_rf(1)+x_vp(end)+x_ss(end), y_rf+offset_rf3, 'LineWidth',lw,'Color','k');
plot([x_rf(end)-x_rf(1)+x_vp(end)+x_ss(end),12],[0,0]+offset_rf3, 'LineWidth',lw,'Color','k');
x = x_ss(end)-x_ss(1)+x_vp(end)+x_ss(end);
rectangle('Position',[x, offset_rf3-0.5, 7.5, 1], 'FaceColor','k');
text(x+3.75, offset_rf3, 'ADC', 'Color','w','FontWeight','bold','HorizontalAlignment','center');
% Gradient line 3
plot([-1.5, x_ss(end)],[0,0]+offset_ss3, 'LineWidth',lw,'Color','k');
plot(x_vp+x_ss(end), -y_vp*0.5+offset_ss3,':', 'LineWidth',lw,'Color','k');
plot(x_vp+x_ss(end), zeros(size(x_vp))*1.0+offset_ss3,':', 'LineWidth',lw,'Color','k');
plot(x_vp+x_ss(end), -y_vp*1.0+offset_ss3,':', 'LineWidth',lw,'Color','k');
plot(x_vp+x_ss(end), -y_vp*1.5+offset_ss3, 'LineWidth',lw,'Color','k');
plot(x_vp+x_ss(end), -y_vp*2.0+offset_ss3,':', 'LineWidth',lw,'Color','k');
plot(x_vp+x_ss(end), -y_vp*2.5+offset_ss3,':', 'LineWidth',lw,'Color','k');
x = x_vp(end)+x_ss(end);
plot(x_ss-x_ss(1)+x, y_ss+offset_ss3, 'LineWidth',lw,'Color','k');
x = x_ss(end)-x_ss(1)+x;
plot([x, 12],[0,0]+offset_ss3, 'LineWidth',lw,'Color','k');
% rectangle and text 3
plot(x_rect, y_rect+offset_ss3,'-.', 'LineWidth',lw2,'Color','k');
text(-4.3, offset_rf3, 'RF');
text(-4.3, offset_ss3, 'G_s_l_i_c_e');
text(-4.5, offset_rf3+1.3, '(3)', 'FontWeight','bold','FontSize',11);
% RF line 4
plot([-1.5, x_vp(end)+x_ss(end)],[0,0]+offset_rf4, 'LineWidth',lw,'Color','k');
plot(x_rf-x_rf(1)+x_vp(end)+x_ss(end), y_rf+offset_rf4, 'LineWidth',lw,'Color','k');
plot([x_rf(end)-x_rf(1)+x_vp(end)+x_ss(end),12],[0,0]+offset_rf4, 'LineWidth',lw,'Color','k');
x = x_ss(end)-x_ss(1)+x_vp(end)+x_ss(end);
rectangle('Position',[x, offset_rf4-0.5, 7.5, 1], 'FaceColor','k');
text(x+3.75, offset_rf4, 'ADC', 'Color','w','FontWeight','bold','HorizontalAlignment','center');
% Gradient line 4
plot([-1.5, x_ss(end)],[0,0]+offset_ss4, 'LineWidth',lw,'Color','k');
plot(x_vp+x_ss(end), y_vp*0.5+offset_ss4,':', 'LineWidth',lw,'Color','k');
plot(x_vp+x_ss(end), zeros(size(x_vp))*1.0+offset_ss4,':', 'LineWidth',lw,'Color','k');
plot(x_vp+x_ss(end), y_vp*1.0+offset_ss4,':', 'LineWidth',lw,'Color','k');
plot(x_vp+x_ss(end), y_vp*1.5+offset_ss4, 'LineWidth',lw,'Color','k');
plot(x_vp+x_ss(end), y_vp*2.0+offset_ss4,':', 'LineWidth',lw,'Color','k');
plot(x_vp+x_ss(end), y_vp*2.5+offset_ss4,':', 'LineWidth',lw,'Color','k');
x = x_vp(end)+x_ss(end);
plot(x_ss-x_ss(1)+x, y_ss+offset_ss4, 'LineWidth',lw,'Color','k');
x = x_ss(end)-x_ss(1)+x;
plot([x, 12],[0,0]+offset_ss4, 'LineWidth',lw,'Color','k');
% rectangle and text 4
plot(x_rect, y_rect2+offset_ss4,'-.', 'LineWidth',lw2,'Color','k');
text(-4.3, offset_rf4, 'RF');
text(-4.3, offset_ss4, 'G_s_l_i_c_e');
text(-4.5, offset_rf4+1.3, '(4)', 'FontWeight','bold','FontSize',11);
% other stuff
plot([-3.75, -4.5, -4.5, -3.75]-1, [5.5, 5.5, -2.5+offset_ss4, -2.5+offset_ss4], 'LineWidth',lw2,'Color','k');
plot([13.75, 14.5, 14.5, 13.75]-1, [5.5, 5.5, -2.5+offset_ss4, -2.5+offset_ss4], 'LineWidth',lw2,'Color','k');
text(-4.5, 5.5, 'for n = 1 ... N (variable-prephasing steps)');
plot([-4, -5, -5, -4]-1, [6.5, 6.5, -3+offset_ss4, -3+offset_ss4], 'LineWidth',lw2,'Color','k');
plot([14, 15, 15, 14]-1, [6.5, 6.5, -3+offset_ss4, -3+offset_ss4], 'LineWidth',lw2,'Color','k');
text(-4.5, 6.5, 'for m = 1 ... M (slices)');
xlim([x_rect(1)-1.2 x_rect(3)+1.2]);
ylim([y_rect2(1)+offset_ss4-1.2 y_rect(2)+offset_ss+3.2]);
text(-7, 7.5, '(B)', 'FontWeight','bold', 'FontSize',12);
set(gca,'XColor','none','YColor','none','TickDir','out');

