clc; clear; close all;
set(0,'DefaultFigureVisible','on');  % force figures to show
figure('Visible','on'); plot(1:10); grid on; title('SANITY');
drawnow;
out = fullfile(pwd,'results'); if ~isfolder(out), mkdir(out); end
saveas(gcf, fullfile(out,'_sanity_plot.png'));
disp(fullfile(out,'_sanity_plot.png'));
