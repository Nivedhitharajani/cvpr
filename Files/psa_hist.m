%% pca_on_hs.m
clc; clear; close all;
set(0,'DefaultFigureVisible','on');

descDir = fullfile(pwd,'descriptors');
indexFile = fullfile(descDir,'index_hs16x8.mat'); 
S = load(indexFile);
X = S.X; labels = S.labels; paths = S.paths;
fprintf('Loaded %d features, dim=%d\n', size(X,2), size(X,1));


%% PCA on HS histogram features
X = X';  % Transpose: now N x D (samples x features)
[coeff, score, latent, ~, explained] = pca(X);

figure;
plot(cumsum(explained), 'LineWidth', 2);
xlabel('Number of principal components');
ylabel('Cumulative variance explained (%)');
title('PCA variance retention on HS histograms');
grid on;


keepVar = 95;  % percent
k = find(cumsum(explained) >= keepVar, 1);
fprintf('Keeping %d components (%.2f%% variance retained)\n', k, sum(explained(1:k)));


X_reduced = score(:, 1:k);  % reduced features (N x k)
X_reduced = X_reduced';     % back to D x N format
