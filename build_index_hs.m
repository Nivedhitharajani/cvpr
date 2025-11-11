%% build_index_hs.m
% Stand-alone HS (Hue–Saturation) index + optional evaluation.
% No external dependencies: hs_hist() is defined at the bottom.

clc; clear; close all;
set(0,'DefaultFigureVisible','on');

%% ===== SETTINGS =====
datasetRoot = fullfile(pwd, 'data', 'msrc_objcategimagedatabase_v2'); % change if needed
nH = 16;                 % Hue bins
nS = 8;                  % Saturation bins  -> dim = nH*nS
doEval = true;           % set false to only build the index
distList = {'chi2','euclidean','cosine','manhattan'};
numQueries = 50;
rng(0);

%% ===== PREP =====
assert(isfolder(datasetRoot), 'Dataset not found: %s', datasetRoot);
descDir = fullfile(pwd,'descriptors'); if ~isfolder(descDir), mkdir(descDir); end
resultsDir = fullfile(pwd,'results');   if ~isfolder(resultsDir), mkdir(resultsDir); end
indexFile = fullfile(descDir, sprintf('index_hs%dx%d.mat', nH, nS));

%% ===== SCAN DATASET =====
[paths, labels] = listImagesWithLabels(datasetRoot);
assert(~isempty(paths), 'No images found under %s', datasetRoot);

%% ===== BUILD FEATURES (HS) =====
fprintf('Building HS histograms (%dx%d) for %d images...\n', nH, nS, numel(paths));
f1 = hs_hist(imread(paths{1}), nH, nS);     % local function below
D  = numel(f1); N = numel(paths);
X  = zeros(D, N, 'single');

t0 = tic;
for i = 1:N
    I = imread(paths{i});
    X(:,i) = hs_hist(I, nH, nS);
    if mod(i, max(1, floor(N/20)))==0
        fprintf('  %d/%d\n', i, N);
    end
end
fprintf('Done in %.1fs. Dim = %d\n', toc(t0), D);

%% ===== SAVE INDEX =====
save(indexFile, 'paths','labels','X','nH','nS','-v7.3');
fprintf('Saved: %s\n', indexFile);

%% ===== OPTIONAL EVALUATION =====
if doEval
    fprintf('Evaluating distances on HS features...\n');
    qi_list = randi(N, [1, numQueries]);
    resultsCSV = fullfile(resultsDir, sprintf('summary_distance_sweep_HS%dx%d.csv', nH, nS));
    fid = fopen(resultsCSV,'w'); fprintf(fid, "distance,meanAP\n");
    for k = 1:numel(distList)
        metric = distList{k};
        [mP, mR, mAP] = eval_macro_pr(X, labels, metric, qi_list);
        fig = figure('Name', sprintf('Macro PR (HS %dx%d, %s)', nH, nS, metric));
        plot(mR, mP, 'LineWidth',2); grid on;
        xlabel('Recall'); ylabel('Precision');
        title(sprintf('Macro PR – HS %dx%d (%s), mAP=%.3f', nH, nS, metric, mAP));
        outPNG = fullfile(resultsDir, sprintf('PR_macro_HS%dx%d_%s.png', nH, nS, metric));
        saveas(fig, outPNG);
        fprintf('Saved PR: %s\n', outPNG);
        fprintf(fid, "%s,%.6f\n", metric, mAP);
    end
    fclose(fid);
    fprintf('Saved sweep summary: %s\n', resultsCSV);
end

disp('HS build complete.');

%% ====================== LOCAL FUNCTIONS ======================

function [paths, labels] = listImagesWithLabels(root)
S = dir(fullfile(root, '**', '*.*'));
S = S(~[S.isdir]);
valid = endsWith(lower({S.name}), {'.jpg','.jpeg','.png','.bmp','.tif','.tiff'});
S = S(valid);
paths = fullfile({S.folder}, {S.name})';
labels = cell(numel(paths),1);
for i=1:numel(paths)
    labels{i} = parentFolder(paths{i});
end
end

function p = parentFolder(f)
[folder,~,~] = fileparts(f);
[~,p] = fileparts(folder);
end

function [mP, mR, mAP] = eval_macro_pr(X, labels, metric, qi_list)
allPrec = []; allRec = []; AP = zeros(numel(qi_list),1);
for t = 1:numel(qi_list)
    qi = qi_list(t);
    q = X(:,qi);
    d = local_compare_distance(q, X, metric);
    [~, order] = sort(d,'ascend');
    isRel = strcmp(labels(order), labels{qi});
    [p, r] = pr_curve(isRel);
    [allPrec, allRec] = padcat_cols(allPrec, allRec, p, r);
    AP(t) = average_precision(isRel);
end
[mP, mR] = macro_average_pr(allPrec, allRec);
mAP = mean(AP,'omitnan');
end

function ap = average_precision(isRel)
isRel = isRel(:);
tp = cumsum(isRel);
k  = (1:numel(isRel))';
prec = tp ./ k;
ap = sum(prec(isRel)) / max(1,sum(isRel));
end

function [mp, mr] = macro_average_pr(allPrec, allRec)
rg = (0:0.01:1)'; P = nan(numel(rg), size(allPrec,2));
for i=1:size(allPrec,2)
    p = allPrec(:,i); r = allRec(:,i);
    m = ~isnan(p) & ~isnan(r);
    p = p(m); r = r(m);
    if isempty(p), continue; end
    [r, iu] = unique(r); p = p(iu);
    P(:,i) = interp1(r, p, rg, 'previous', 'extrap');
end
mp = nanmean(P,2); mr = rg;
end

function [A, B] = padcat_cols(A, B, v1, v2)
if isempty(A), A=v1; B=v2; return; end
m = size(A,1);
if numel(v1)>m
    A=[A; nan(numel(v1)-m, size(A,2))];
    B=[B; nan(numel(v1)-m, size(B,2))];
elseif numel(v1)<m
    v1=[v1; nan(m-numel(v1),1)];
    v2=[v2; nan(m-numel(v2),1)];
end
A=[A, v1]; B=[B, v2];
end

function d = local_compare_distance(q, X, metric)
q = double(q(:)); X = double(X);
switch lower(metric)
    case 'euclidean'
        dif = X - q; d = sqrt(sum(dif.^2,1));
    case 'manhattan'
        d = sum(abs(X - q),1);
    case 'chi2'
        num = (X - q).^2; den = (X + q + eps); d = 0.5*sum(num./den,1);
    case 'cosine'
        num = q' * X; d = 1 - (num ./ (norm(q)*sqrt(sum(X.^2,1)) + eps));
    otherwise
        error('Unknown distance: %s', metric);
end
d = d(:);
end

function [prec, rec] = pr_curve(isRel)
isRel = isRel(:);
TP = cumsum(isRel);
P  = sum(isRel);
k  = (1:numel(isRel))';
prec = TP ./ k;
rec  = TP ./ max(P,1);
end

function h = hs_hist(I, nH, nS)
% HS histogram (Hue–Saturation), L1-normalised (dims = nH*nS)
if nargin<2, nH=16; end
if nargin<3, nS=8;  end
if size(I,3)==1, I = repmat(I,[1 1 3]); end
I = im2single(I);
HSV = rgb2hsv(I);
H = HSV(:,:,1); S = HSV(:,:,2);
hidx = min(floor(H*nH), nH-1);
sidx = min(floor(S*nS), nS-1);
idx = hidx*nS + sidx + 1;           % 1..(nH*nS)
h2d = accumarray(idx(:), 1, [nH*nS, 1]);
h = single(h2d / (sum(h2d)+eps));
end
% Example: visualise one HS histogram vector `h`
% If you have an image:
% I = imread(<your image path>);
% h = hs_hist(I, 16, 8);

assert(isvector(h), 'h must be a 1D vector');
h = double(h(:));                 % make column, double
figure; bar(h); grid on;
xlabel('HS bin index (1..nH*nS)');
ylabel('Normalised count');
title('HS histogram (1D bars)');

% Optional: show tick labels every saturation-block
nH = 16; nS = 8;                  % set to your settings
xt = 1:nS:numel(h);
xticklabels = arrayfun(@(k) sprintf('H%02d',k), 1:nH, 'UniformOutput', false);
set(gca, 'XTick', xt, 'XTickLabel', xticklabels, 'XTickLabelRotation', 0);
