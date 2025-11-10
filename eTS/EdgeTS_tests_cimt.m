% edge time series and optimizations

% load in resting state cimt cci and shm
addpath(genpath('/Users/fei/Desktop/Harris_Lab/dFNC/eTS/EdgeTS_codes'))
addpath(genpath('/Users/fei/BCT/2019_03_03_BCT'))
datapath = '/Users/fei/Desktop/Harris_Lab/dFNC/CIMT_data/resting_state';

cci_file_list = dir(fullfile(datapath, 'CCI_none*.csv'));
shm_file_list = dir(fullfile(datapath, 'SHM_none*.csv'));
first_file = fullfile(datapath, cci_file_list(1).name);
[len, w] = size(readmatrix(first_file)); % going to be 450 frames x 124 rois
num_files_cci = length(cci_file_list);
TS_cci = zeros(len, w, num_files_cci);
num_files_shm = length(shm_file_list);
TS_shm = zeros(len, w, num_files_shm);

for i = 1:num_files_cci
    file_path = fullfile(datapath, cci_file_list(i).name);
    TS_cci(:, :, i) = readmatrix(file_path);
end
for i = 1:num_files_shm
    file_path = fullfile(datapath, shm_file_list(i).name);
    TS_shm(:, :, i) = readmatrix(file_path);
end

% load in groups with the subjects - cci then sham
TS = cat(3, TS_cci, TS_shm);
group_list = [ones(num_files_cci, 1); zeros(num_files_shm, 1)];
numfr=round(0.1*length(TS));

% optimization to see which nodes are contributory to group
[ini_tmpl,opt_tmpl,perf] = optim_template(TS,group_list,numfr, 'H',500); % creates optimized template to correlate with behavior
subplot(3, 1, 1)
imagesc(ini_tmpl)
title('Initial template')
colormap('gray')
subplot(3, 1, 2)
imagesc(opt_tmpl)
title('Optimal template')
colormap('gray')
subplot(3, 1, 3)
imagesc(opt_tmpl-ini_tmpl)
title('Optimal-Initial template')
colormap('gray')
disp("Nodes differentiating cci vs shm ")
disp(find(opt_tmpl==1)');

% get the edge ts (average for each group - or get edge ts for each subject
% then find average)
ets_cci_agg = [];
for i=1:size(TS_cci,3)
    ets_cci_agg = cat(3, ets_cci_agg, fcn_edgets(TS_cci(:, :, i)));
end
mean_ets_cci = mean(ets_cci_agg, 3); % gets 450 x 7626 (num edges)

rss_cci = sum(mean_ets_cci.^2,2).^0.5; % sum along 2nd dimension, which is time points
top_indices_cci = find(rss_cci >= prctile(rss_cci, 90)); % time points with top rss
TS_repopulated_cci = zeros(size(mean_ets_cci));
TS_repopulated_cci(top_indices_cci, :) = mean_ets_cci(top_indices_cci, :); % zero everywhere besides top indices

ets_shm_agg = [];
for i=1:size(TS_shm,3)
    ets_shm_agg = cat(3, ets_shm_agg, fcn_edgets(TS_shm(:, :, i)));
end
mean_ets_shm = mean(ets_shm_agg, 3); % gets 450 x 7626 (num edges)

rss_shm = sum(mean_ets_shm.^2,2).^0.5;
top_indices_shm = find(rss_shm >= prctile(rss_shm, 90));
TS_repopulated_shm = zeros(size(mean_ets_shm));
TS_repopulated_shm(top_indices_shm, :) = mean_ets_shm(top_indices_shm, :);

figure;
subplot(2, 1, 1);
shmim = imagesc(gca, TS_repopulated_shm');  % Averaging over subjects & runs
colormap("cool");
alphaData = TS_repopulated_shm' ~= 0; 
set(shmim, 'AlphaData', alphaData);
title('Filtered Time Series (Top 10% RSS) Shm');
xlabel('Time');
ylabel('Edges');

subplot(2, 1, 2);
cciim = imagesc(gca, TS_repopulated_cci');
alphaDatacci = TS_repopulated_cci' ~= 0; 
set(cciim, 'AlphaData', alphaDatacci);
title('Filtered Time Series (Top 10% RSS) CCI');
xlabel('Time');
ylabel('Edges');

notsame_edge = find(top_indices_shm ~= top_indices_cci);
disp("Time points that are different for top RSS in sham and cci from edge time series");
disp(top_indices_shm(notsame_edge));
disp(top_indices_cci(notsame_edge));

% get top/bottom FC - cci
cis = double(zscore((mean(TS_cci, 3))>0)+1)';
FC = corr(mean(TS_cci, 3)); % full FC, corr() computes between pairs of columns, each column is an ROI
% select top/bottom 'rssfrac' RSS frames (110 = 10 percent)
% one can use any criterion to select a subset of frames...
rssfrac = 110;
[rsssorted, rssinds] = sort(rss_cci,'descend');
top = rssinds(1:rssfrac);
bot = rssinds(end-rssfrac+1:end);

% FC/A computed from the top/bot frames
mets_top = sum(mean_ets_cci(top,:))./(rssfrac-1);
N = 124;
FCtop = zeros(N);
inds = find(triu(ones(N),1));
FCtop(inds) = mets_top;
mets_bot = sum(mean_ets_cci(bot,:))./(rssfrac-1);
FCbot = zeros(N);
FCbot(inds) = mets_bot;
anulltop = sum(sum([sum(cis(:,top)==1)./N; sum(cis(:,top)==2)./N].*...
   [(sum(cis(:,top)==1)-1)./(N-1); (sum(cis(:,top)==2)-1)./(N-1)],2));
Atop = (agreement(cis(:,top)) - anulltop)./length(top);
anullbot = sum(sum([sum(cis(:,bot)==1)./N; sum(cis(:,bot)==2)./N].*...
   [(sum(cis(:,bot)==1)-1)./(N-1); (sum(cis(:,bot)==2)-1)./(N-1)],2));
Abot = (agreement(cis(:,bot)) - anullbot)./length(bot);

% similarity btw full FC and top/bot FC
corr(FCtop(inds),FC(inds))
corr(FCbot(inds),FC(inds))

% get top/bottom FC - shm
cis = double(zscore((mean(TS_shm, 3))>0)+1)';
FC = corr(mean(TS_shm, 3)); % full FC
% select top/bottom 'rssfrac' RSS frames (110 = 10 percent)
% one can use any criterion to select a subset of frames...
rssfrac = 110;
[rsssorted, rssinds] = sort(rss_shm,'descend');
top = rssinds(1:rssfrac);
bot = rssinds(end-rssfrac+1:end);

% FC/A computed from the top/bot frames
mets_top = sum(mean_ets_shm(top,:))./(rssfrac-1);
FCtop = zeros(N);
FCtop(inds) = mets_top;
mets_bot = sum(mean_ets_shm(bot,:))./(rssfrac-1);
FCbot = zeros(N);
FCbot(inds) = mets_bot;
anulltop = sum(sum([sum(cis(:,top)==1)./N; sum(cis(:,top)==2)./N].*...
   [(sum(cis(:,top)==1)-1)./(N-1); (sum(cis(:,top)==2)-1)./(N-1)],2));
Atop = (agreement(cis(:,top)) - anulltop)./length(top);
anullbot = sum(sum([sum(cis(:,bot)==1)./N; sum(cis(:,bot)==2)./N].*...
   [(sum(cis(:,bot)==1)-1)./(N-1); (sum(cis(:,bot)==2)-1)./(N-1)],2));
Abot = (agreement(cis(:,bot)) - anullbot)./length(bot);

% similarity btw full FC and top/bot FC
corr(FCtop(inds),FC(inds))
corr(FCbot(inds),FC(inds))

% which ets are sig different between cci and shm
num_timepoints = size(mean_ets_cci, 1);
p_values = zeros(1, num_timepoints);

% each ets are they different between groups
for t = 1:num_timepoints
    cci_values = mean_ets_cci(t, :);
    sham_values = mean_ets_shm(t, :);
    [~, p_values(t)] = ttest2(cci_values, sham_values);
end

significant_frames = find(p_values < 0.05); % possibly change alpha to 0.01
fprintf('Significant frames %f\n', significant_frames);
ets_mean_cci_filtered = mean_ets_cci(significant_frames, :);
ets_mean_shm_filtered = mean_ets_shm(significant_frames, :);

cci_filtered = TS_cci(significant_frames, :, :);
shm_filtered = TS_shm(significant_frames, :, :);

% corr w behavior
TS = cat(3, TS_cci, TS_shm);
TS_filtered = cat(3, cci_filtered, shm_filtered);
num_nodes = size(TS, 2);
avg_ts = squeeze(mean(TS, 1));
avg_ts_filtered = squeeze(mean(TS_filtered, 1));
corr_values = zeros(num_nodes, 1);
p_values = zeros(num_nodes, 1);
corr_values_filtered = zeros(num_nodes, 1);
p_values_filtered = zeros(num_nodes, 1);

% get the ts for the node and then get correlation to behavior
for node = 1:num_nodes
    node_data = avg_ts(node, :)';
    node_data_filtered = avg_ts_filtered(node, :)';

    [r, p] = corr(node_data, group_list(:), 'type', 's', 'rows', 'complete');
    corr_values(node) = r;
    p_values(node) = p;

    [r_f, p_f] = corr(node_data_filtered, group_list(:), 'type', 's', 'rows', 'complete'); % -- gets corr between each subj and each group value for each node
    corr_values_filtered(node) = r_f;
    p_values_filtered(node) = p_f;
end
mean_corr = mean(corr_values); % one per node
fprintf('Mean correlation (unfiltered ETS): %.4f\n', mean_corr);
mean_corr_filtered = mean(corr_values_filtered);
fprintf('Mean correlation (filtered ETS): %.4f\n', mean_corr_filtered);

figure;
bar(corr_values)
hold on
bar(corr_values_filtered)
xlabel('Node');
ylabel('Correlation with group');
legend({'All frames', 'Filtered frames'});
hold off