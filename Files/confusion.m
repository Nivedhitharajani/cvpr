%% build_confusion_matrix.m

clc; clear; close all;
set(0,'DefaultFigureVisible','on');

distance   = 'chi2';                     % 'euclidean'|'chi2'|'cosine'|'manhattan'
descDir    = fullfile(pwd,'descriptors');
resultsDir = fullfile(pwd,'results'); if ~isfolder(resultsDir), mkdir(resultsDir); end

% Either hard-code the index file OR pick it interactively:
% indexFile = fullfile(descDir,'index_bins8.mat');          % baseline
% indexFile = fullfile(descDir,'index_grid2x2_bins8.mat');  % grid version
[fn,fp] = uigetfile(fullfile(descDir,'*.mat'),'Select descriptor index MAT file');
assert(~isequal(fn,0), 'No file selected.');
indexFile = fullfile(fp,fn);

fprintf('Loading index: %s\n', indexFile);
S = load(indexFile);

% Robustly get X, labels, paths (support different field names if any)
if isfield(S,'X'), X = S.X; else, error('Descriptor matrix X not found in MAT.'); end
if isfield(S,'labels'), labels = S.labels; else, error('labels not found in MAT.'); end
if isfield(S,'paths'), paths = S.paths; else, paths = []; end %#ok<NASGU>

% Normalize label type
if isstring(labels), labels = cellstr(labels); end
labels = labels(:);
[D,N] = size(X);
fprintf('Feature size: %d x %d (D x N)\n', D, N);

uniq = unique(labels);
C = zeros(numel(uniq));

fprintf('Computing top-1 predictions using %s distance...\n', distance);
for qi = 1:N
    q = X(:,qi);
    d = local_compare_distance(q, X, distance);
    % remove self-match
    d(qi) = inf;
    [~, idx] = min(d);           % top-1 retrieval
    gt  = labels{qi};
    prd = labels{idx};
    r = find(strcmp(uniq,gt),1);
    c = find(strcmp(uniq,prd),1);
    C(r,c) = C(r,c) + 1;
end

% Normalized (per-row) confusion for readability
rowSums = sum(C,2); rowSums(rowSums==0) = 1;
Cnorm = C ./ rowSums;

fig = figure('Name',sprintf('Confusion (top-1, %s)',distance));
imagesc(Cnorm); axis image; colorbar;
title(sprintf('Confusion (top-1) – %s', distance));
set(gca,'XTick',1:numel(uniq),'XTickLabel',uniq,'XTickLabelRotation',45);
set(gca,'YTick',1:numel(uniq),'YTickLabel',uniq);


[~, baseName, ~] = fileparts(indexFile);
pngPath = fullfile(resultsDir, sprintf('confusion_%s_%s.png', distance, baseName));
csvCounts = fullfile(resultsDir, sprintf('confusion_counts_%s_%s.csv', distance, baseName));
csvNorm   = fullfile(resultsDir, sprintf('confusion_normalized_%s_%s.csv', distance, baseName));

saveas(fig, pngPath);
fprintf('Saved heatmap: %s\n', pngPath);

writecell([{''} uniq'; uniq num2cell(C)], csvCounts);
writecell([{''} uniq'; uniq num2cell(Cnorm)], csvNorm);
fprintf('Saved CSVs:\n  %s\n  %s\n', csvCounts, csvNorm);

disp('Done.');

function d = local_compare_distance(q, X, metric)
    q = double(q(:));
    X = double(X);
    switch lower(metric)
        case 'euclidean'
            dif = X - q;
            d = sqrt(sum(dif.^2,1));
        case 'manhattan'
            d = sum(abs(X - q),1);
        case 'chi2'
            num = (X - q).^2;
            den = (X + q + eps);
            d = 0.5 * sum(num ./ den,1);
        case 'cosine'
            num = q' * X;
            d = 1 - (num ./ (norm(q) * sqrt(sum(X.^2,1)) + eps));
        otherwise
            error('Unknown distance: %s', metric);
    end
    d = d(:);
end
