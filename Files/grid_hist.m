function h = grid_hist_rgb(I, numBins, rc)

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


indexFile = fullfile(descDir, 'index_grid2x2_bins8.mat');  % before save()
save(indexFile, 'paths','labels','X','numBins','-v7.3');

function plot_confusion_from_retrieval(X, labels, metric, outpng)
    % X: DxN, labels: Nx1 cellstr or string
    if isstring(labels), labels = cellstr(labels); end
    N = numel(labels);
    uniq = unique(labels);
    C = zeros(numel(uniq));

    for qi = 1:N
        d = compare_distance(X(:,qi), X, metric);
        [~, order] = sort(d,'ascend');
        order(order==qi) = [];                   % remove self
        pred = labels{order(1)};                 % top-1 prediction
        r = find(strcmp(uniq, labels{qi}));
        c = find(strcmp(uniq, pred));
        C(r,c) = C(r,c) + 1;
    end

    figure; imagesc(C); axis image; colorbar;
    set(gca,'XTick',1:numel(uniq),'XTickLabel',uniq,'XTickLabelRotation',45);
    set(gca,'YTick',1:numel(uniq),'YTickLabel',uniq);
    title(sprintf('Confusion (top-1, %s)', metric));
    if nargin>=4 && ~isempty(outpng), saveas(gcf, outpng); end
end

function h = hs_hist(I, nH, nS)
if nargin<2, nH=16; end
if nargin<3, nS=8; end
if size(I,3)==1, I = repmat(I,[1 1 3]); end
I = im2single(I);
HSV = rgb2hsv(I);
H = HSV(:,:,1); S = HSV(:,:,2);
hidx = min(floor(H*nH), nH-1);
sidx = min(floor(S*nS), nS-1);
idx = hidx*nS + sidx + 1;
h2d = accumarray(idx(:), 1, [nH*nS, 1]);
h = single(h2d / (sum(h2d)+eps));
end


fprintf('Feature size: %d x %d (D x N)\n', size(X,1), size(X,2));
assert(~any(isnan(X(:))), 'NaNs in features – check extractor.');
disp('Example label + path:'); disp(labels{1}); disp(paths{1});
