%% visual_search_main.m

clc; clear; close all;

set(0,'DefaultFigureVisible','on');  % show figures
dbstop if error;                     
fprintf('CWD: %s\n', pwd);
fprintf('About to build/load features...\n');

datasetRoot = fullfile(pwd, 'data', 'msrc_objcategimagedatabase_v2'); 
numBins     = 32;          % per channel (8 -> 512-D 3D hist)
distance    = 'manhattan';% 'euclidean' | 'chi2' | 'cosine' | 'manhattan'
topK        = 20;         % images to display
numQueries  = 20;         % random queries for evaluation
rng(0);


if ~isfolder(datasetRoot)
    error('Dataset folder not found: %s', datasetRoot);
end
descDir    = fullfile(pwd, 'descriptors'); if ~isfolder(descDir), mkdir(descDir); end
resultsDir = fullfile(pwd, 'results');     if ~isfolder(resultsDir), mkdir(resultsDir); end
indexFile  = fullfile(descDir, 'index.mat');

if exist(indexFile, 'file')
    S = load(indexFile);
    paths  = S.paths; 
    labels = S.labels; 
    X      = S.X;        % D x N
    fprintf('Loaded cached descriptors: %s\n', indexFile);
else
    fprintf('Computing descriptors (this may take a minute)...\n');
    [paths, labels] = listImagesWithLabels(datasetRoot);
    if isempty(paths), error('No images found under %s', datasetRoot); end

    % Compute 3D RGB hist per image
    ex = rgb_hist3d(imread(paths{1}), numBins);
    D  = numel(ex); 
    N  = numel(paths);
    X  = zeros(D, N, 'single');

    t0 = tic;
    for i = 1:N
        I = imread(paths{i});
        gi = grid_hist_rgb(I, numBins, [2 2]);
        if i==1, X = zeros(numel(f), N, 'single'); end
        X(:,i) = f;
        if mod(i, max(1, floor(N/20))) == 0
            fprintf('  %d/%d\n', i, N);
        end
    end
    fprintf('Feature build done in %.1fs\n', toc(t0));

    save(indexFile, 'paths', 'labels', 'X', 'numBins', '-v7.3');
    fprintf('Saved index: %s\n', indexFile);
end


fprintf('Feature matrix size: %d x %d (D x N)\n', size(X,1), size(X,2));
assert(~isempty(X) && ~isempty(paths), 'No features/paths loaded.');
assert(~any(isnan(X(:))), 'NaNs in X.');
fprintf('Example path: %s\n', paths{1});

N = numel(paths);
allPrec = []; allRec = [];

for q = 1:numQueries
    qi = randi(N);
    qfeat  = X(:, qi);
    qlabel = labels{qi};

    d = compare_distance(qfeat, X, distance);
    [~, order] = sort(d, 'ascend');

    % Display ranked results
    f1 = figure(1); clf(f1); set(f1, 'Name', sprintf('Query %d Top-%d (%s)', q, topK, distance));
    show_ranked(paths, labels, order, qlabel, topK, qi);

    % Save montage per query
    saveas(f1, fullfile(resultsDir, sprintf('top%d_q%d.png', topK, q)));

    % Compute PR for this query
    isRel = strcmp(labels(order), qlabel);
    [prec, rec] = pr_curve(isRel);
    [allPrec, allRec] = padcat_cols(allPrec, allRec, prec, rec);
end

disp('Creating figures now...'); drawnow;


[meanPrec, meanRec] = macro_average_pr(allPrec, allRec);
f2 = figure(2); clf(f2);
plot(meanRec, meanPrec, 'LineWidth', 2); grid on;
xlabel('Recall'); ylabel('Precision');
title(sprintf('Macro-averaged PR over %d queries (%s)', numQueries, distance));
xlim([0 1]); ylim([0 1]);
saveas(f2, fullfile(resultsDir, 'PR_macro.png'));

fprintf('Done. Figures saved in: %s\n', resultsDir);

function plot_confusion_from_retrieval(X, labels, metric, outpng)
    N = numel(labels);
    uniq = unique(labels);
    C = zeros(numel(uniq));
    for qi = 1:N
        d = compare_distance(X(:,qi), X, metric);
        [~, order] = sort(d,'ascend');
        order(order==qi) = [];                 % drop self
        pred = labels{order(1)};               % top-1
        C(strcmp(uniq, labels{qi}), strcmp(uniq, pred)) = ...
            C(strcmp(uniq, labels{qi}), strcmp(uniq, pred)) + 1;
    end
    figure; imagesc(C); axis image; colorbar;
    set(gca,'XTick',1:numel(uniq),'XTickLabel',uniq,'XTickLabelRotation',45);
    set(gca,'YTick',1:numel(uniq),'YTickLabel',uniq);
    title(sprintf('Confusion (top-1, %s)', metric));
    if nargin>3, saveas(gcf,outpng); end
end

function [paths, labels] = listImagesWithLabels(root)
    S = dir(fullfile(root, '**', '*.*'));
    S = S(~[S.isdir]); % files only
    valid = endsWith(lower({S.name}), {'.jpg','.jpeg','.png','.bmp','.tif','.tiff'});
    S = S(valid);

    paths = fullfile({S.folder}, {S.name})';
    labels = cell(numel(paths), 1);
    for i = 1:numel(paths)
        labels{i} = parentFolder(paths{i});
    end
end

function h = rgb_hist3d(I, numBins)
    % 3D RGB histogram, L1-normalised, single precision
    if size(I,3) == 1
        I = repmat(I, [1 1 3]);
    end
    if ~isa(I,'uint8')
        I = im2uint8(I);
    end

    edges = linspace(0, 255, numBins+1);
    Rq = discretize(I(:,:,1), edges);
    Gq = discretize(I(:,:,2), edges);
    Bq = discretize(I(:,:,3), edges);

    idx = sub2ind([numBins, numBins, numBins], Rq, Gq, Bq);
    h3  = accumarray(idx(:), 1, [numBins^3, 1]);
    h   = single(h3(:));
    h   = h / (sum(h) + eps);
end

function d = compare_distance(q, X, metric)
    % q: Dx1, X: DxN
    q = double(q(:));
    X = double(X);

    switch lower(metric)
        case 'euclidean'
            dif = X - q;                     % DxN
            d = sqrt(sum(dif.^2, 1));

        case 'manhattan'
            d = sum(abs(X - q), 1);

        case 'chi2'
            num = (X - q).^2;
            den = (X + q + eps);
            d = 0.5 * sum(num ./ den, 1);

        case 'cosine'
            num = q' * X;                    % 1xN
            d = 1 - (num ./ (norm(q) * sqrt(sum(X.^2,1)) + eps));

        otherwise
            error('Unknown distance: %s', metric);
    end
    d = d(:); % Nx1
end

function show_ranked(paths, labels, order, qlabel, K, qi)
    % Display query image + a grid of top-K
    K = min(K, numel(order));
    cols = 5; rows = ceil((K+1)/cols); % +1 for the query image
    for k = 1:(K+1)
        subplot(rows, cols, k);
        if k == 1
            imshow(imread(paths{qi}));
            title(sprintf('Query (%s)', qlabel), 'Interpreter','none');
        else
            idx = order(k-1);
            imshow(imread(paths{idx}));
            good = strcmp(labels{idx}, qlabel);
            title(sprintf('#%d %s', k-1, tf(good)), 'Color', good*[0 0.5 0] + ~good*[0.6 0 0], ...
                  'FontWeight','bold');
        end
        axis image off;
    end
end

function [prec, rec] = pr_curve(isRel)
    % isRel: logical (N x 1) over ranked list
    isRel = isRel(:);
    TP = cumsum(isRel);
    P  = sum(isRel);
    Rk = (1:numel(isRel))';
    prec = TP ./ Rk;
    rec  = TP ./ max(P,1);
end

function [A, B] = padcat_cols(A, B, v1, v2)
    % Right-pad 2 cols matrices with NaNs to align vector lengths
    if isempty(A); A = v1; B = v2; return; end
    m = size(A,1);
    if numel(v1) > m
        A = [A; nan(numel(v1)-m, size(A,2))];
        B = [B; nan(numel(v1)-m, size(B,2))];
    elseif numel(v1) < m
        v1 = [v1; nan(m-numel(v1),1)];
        v2 = [v2; nan(m-numel(v2),1)];
    end
    A = [A, v1]; B = [B, v2];
end

function [mp, mr] = macro_average_pr(allPrec, allRec)
    rg = (0:0.01:1)';
    P = nan(numel(rg), size(allPrec,2));
    for i = 1:size(allPrec,2)
        p = allPrec(:,i); r = allRec(:,i);
        m = ~isnan(p) & ~isnan(r);
        p = p(m); r = r(m);
        if isempty(p), continue; end
        % make recall monotone and unique for interp
        [r, iu] = unique(r);
        p = p(iu);
        P(:,i) = interp1(r, p, rg, 'previous', 'extrap');
    end
    mp = nanmean(P, 2);
    mr = rg;
end

function s = tf(b)
    if b, s = '✓'; else, s = '✗'; end
end

function p = parentFolder(f)
    [folder,~,~] = fileparts(f);
    [~,p] = fileparts(folder);
end


distList = {'euclidean','chi2','cosine','manhattan'};
numQueries = 50;       % increase for more stable curves
topK = 20;
qi_list = randi(numel(paths), [1,numQueries]);  % fixed query set for fairness

results = run_distance_sweep(X, paths, labels, distList, topK, qi_list, fullfile(pwd,'results'));

function results = run_distance_sweep(X, paths, labels, distList, topK, qi_list, outdir)
    if ~isfolder(outdir), mkdir(outdir); end
    results = struct('distance',{},'AP',[]);
    for dsi = 1:numel(distList)
        metric = distList{dsi};
        allPrec = []; allRec = []; AP = zeros(numel(qi_list),1);
        for t = 1:numel(qi_list)
            qi = qi_list(t);
            qfeat = X(:,qi); qlabel = labels{qi};
            d = compare_distance(qfeat, X, metric);
            [~, order] = sort(d, 'ascend');
            isRel = strcmp(labels(order), qlabel);
            [prec, rec] = pr_curve(isRel);
            [allPrec, allRec] = padcat_cols(allPrec, allRec, prec, rec);
            AP(t) = average_precision(isRel);  % mAP component
        end
        [mP, mR] = macro_average_pr(allPrec, allRec);
        figure; plot(mR, mP, 'LineWidth', 2); grid on;
        title(sprintf('Macro PR (%s)', metric)); xlabel('Recall'); ylabel('Precision');
        saveas(gcf, fullfile(outdir, sprintf('PR_macro_%s.png', metric)));
        results(dsi).distance = metric;
        results(dsi).AP = AP;
    end
    % Save a simple CSV summary (mean AP)
    fid = fopen(fullfile(outdir,'summary_distance_sweep.csv'),'w');
    fprintf(fid,'distance,meanAP\n');
    for i=1:numel(results)
        fprintf(fid,'%s,%.4f\n', results(i).distance, mean(results(i).AP,'omitnan'));
    end
    fclose(fid);
end

function ap = average_precision(isRel)
    isRel = isRel(:);
    tp = cumsum(isRel);
    k  = (1:numel(isRel))';
    prec = tp ./ k;
    ap = sum(prec(isRel)) / max(1,sum(isRel));
end


function h = grid_hist_rgb(I, numBins, rc)
% Concatenate RGB 3D histograms over an r x c grid
if nargin<3, rc = [2 2]; end
I = ensure_rgb_uint8(I);
[hgt,wid,~] = size(I);
rEdges = round(linspace(1, hgt+1, rc(1)+1));
cEdges = round(linspace(1, wid+1, rc(2)+1));
parts = cell(rc(1)*rc(2),1); k=1;
for r=1:rc(1)
  for c=1:rc(2)
    R1=rEdges(r); R2=rEdges(r+1)-1; C1=cEdges(c); C2=cEdges(c+1)-1;
    patch = I(R1:R2, C1:C2, :);
    parts{k} = rgb_hist3d(patch, numBins); k=k+1;
  end
end
h = single(cat(1, parts{:}));
h = h / (sum(h)+eps);
end

function I = ensure_rgb_uint8(I)
if size(I,3)==1, I = repmat(I,[1 1 3]); end
if ~isa(I,'uint8'), I = im2uint8(I); end
end



