
%% (c) John Collomosse 2010  (J.Collomosse@surrey.ac.uk)
%% Centre for Vision Speech and Signal Processing (CVSSP)
%% University of Surrey, United Kingdom

close all;
clear all;

function quick_retrieval_demo(varargin)



p = inputParser;
addParameter(p,'dataset', fullfile('data','msrcv2'));  % not strictly needed; image paths stored in mats
addParameter(p,'descroot', fullfile('desc'));          % root of all descriptor folders
addParameter(p,'sub','color8');                        % descriptor subfolder: color8 or color8_g2
addParameter(p,'metric','CHI2');                       % L2 | L1 | CHI2
addParameter(p,'show',15);                             % how many results to display
parse(p,varargin{:});
DATASET_FOLDER       = p.Results.dataset;
DESCRIPTOR_FOLDER    = p.Results.descroot;
DESCRIPTOR_SUBFOLDER = p.Results.sub;
METRIC               = upper(p.Results.metric);
SHOW                 = p.Results.show;

% Sanity checks
assert(isfolder(DESCRIPTOR_FOLDER), 'Descriptor root not found: %s', DESCRIPTOR_FOLDER);
subdir = fullfile(DESCRIPTOR_FOLDER, DESCRIPTOR_SUBFOLDER);
assert(isfolder(subdir), 'Descriptor subfolder not found: %s', subdir);

%% 1) Load all descriptors into ALLFEAT (one row per image)
ALLFEAT  = [];
ALLFILES = {};
ctr = 1;

matfiles = dir(fullfile(subdir,'**','*.mat'));  % recursive through class folders
assert(~isempty(matfiles), 'No .mat descriptors found under %s. Run cvpr_computedescriptors first.', subdir);

for k = 1:numel(matfiles)
    pmat = fullfile(matfiles(k).folder, matfiles(k).name);
    S = load(pmat);
    % Your pipeline saved 'f' (row vector), 'label', and 'imgp' (image path)
    if isfield(S,'f')
        F = S.f;              %#ok<NASGU>
    elseif isfield(S,'F')
        F = S.F;              %#ok<NASGU> % support legacy variable name
    else
        error('Descriptor file missing variable f/F: %s', pmat);
    end
    if isfield(S,'f'), feat = double(S.f(:)).'; else, feat = double(S.F(:)).'; end
    ALLFEAT(ctr,1:numel(feat)) = feat; %#ok<AGROW>
    if isfield(S,'imgp'), ALLFILES{ctr,1} = char(S.imgp); else
        % Fallback: reconstruct from dataset + filename stem if needed
        [~,stem,~] = fileparts(matfiles(k).name);
        candidateJPG = fullfile(DATASET_FOLDER,'Images',[stem '.jpg']);
        candidateBMP = fullfile(DATASET_FOLDER,'Images',[stem '.bmp']);
        if exist(candidateJPG,'file'), ALLFILES{ctr,1} = candidateJPG;
        elseif exist(candidateBMP,'file'), ALLFILES{ctr,1} = candidateBMP;
        else, ALLFILES{ctr,1} = ''; % we'll check later
        end
    end
    ctr = ctr + 1;
end

% Verify we have image paths
missing = find(cellfun(@isempty, ALLFILES));
if ~isempty(missing)
    warning('%d images have unknown paths; they will be skipped in display.', numel(missing));
end

%% 2) Pick a random query image (MATLAB indices start at 1)
NIMG = size(ALLFEAT,1);
queryimg = randi(NIMG); % FIX: your original floor(rand()*N) could be 0 (invalid)

%% 3) Compute distances to the query
dst = zeros(NIMG,2);
for i = 1:NIMG
    candidate = ALLFEAT(i,:);
    query     = ALLFEAT(queryimg,:);
    d = cvpr_compare(query, candidate, METRIC); % pass metric explicitly
    dst(i,:) = [d, i];
end
% Exclude the query itself explicitly (so it never appears in top results)
dst(queryimg,1) = inf;

% Sort ascending distance
dst = sortrows(dst, 1);

%% 4) Visualise top results
SHOW = min(SHOW, size(dst,1)); 
topIdx = dst(1:SHOW,2);

% Build a montage row
tiles = {};
% First tile: the query (highlight mentally)
try
    qimg = safe_read(ALLFILES{queryimg});
    tiles{end+1} = qimg; %#ok<AGROW>
catch
    tiles{end+1} = uint8(255*ones(81,81,3)); %#ok<AGROW>
end

for ii = 1:numel(topIdx)
    try
        img = safe_read(ALLFILES{topIdx(ii)});
        tiles{end+1} = img; %#ok<AGROW>
    catch
        tiles{end+1} = uint8(255*ones(81,81,3)); %#ok<AGROW>
    end
end

% Make uniform size thumbnails
thumbs = cellfun(@(im) resize_crop(im, [81 81]), tiles, 'UniformOutput', false);
stack  = cat(4, thumbs{:});

figure('Name', sprintf('Query + Top-%d (%s, %s)', SHOW, DESCRIPTOR_SUBFOLDER, METRIC));
montage(stack);
title(sprintf('Query (left) then Top-%d — %s / %s', SHOW, DESCRIPTOR_SUBFOLDER, METRIC));

end % function


function im = safe_read(p)
if ~exist(p,'file'), error('Missing image: %s', p); end
im = imread(p);
if size(im,3)==1, im = repmat(im,[1 1 3]); end
% Downsample for display like your original
im = im(1:2:end, 1:2:end, :);
end

function out = resize_crop(im, targetHW)
% Resize shortest side, then center-crop to [H W].
H = targetHW(1); W = targetHW(2);
[h,w,~] = size(im);
scale = max(H/h, W/w);
im2 = imresize(im, scale);
[h2,w2,~] = size(im2);
r1 = floor((h2-H)/2)+1; r2 = r1+H-1;
c1 = floor((w2-W)/2)+1; c2 = c1+W-1;
out = im2(r1:r2, c1:c2, :);
end
