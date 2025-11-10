clearvars
close all

%% Sample analyses of edge time series and bipartitions
%
% This simple script demonstrates most of the main constructs in Sporns et
% al (2021). A single sample time series is supplied. Yeo7 canonical
% networks are ordered as follows: VIS, SOM, DAN, VAN, LIM, FP, DMN.
%
% Papers to cite on bipartitions, edge time series and RSS amplitudes:
%   Esfahlani FZ, et al (2020) PNAS
%   Sporns O, et al (2021) Network Neuroscience 
%
% The script, run with the supplied time series, reproduces several panels
% in Sporns et al (2021). Please cite the above two articles if portions of
% the code below are useful for your own analyses.  You will need some
% utilities that are part of the 'Brain Connectivity Toolbox'
%--------------------------------------------------------------------------

%% Load an example BOLD timeseries (200 nodes, ordered in Yeo7 encoding)

% time series should be [TxN] (T = time steps, N = nodes), and z-scored (all
% node-wise means = 0, std = 1)
%load example_timeseries
ts = load("/Users/rachelfox/dFC_Toolboxes/Leida/CSVs_with_regions_stim_pattern_CIMT/AllTimepoints/CCI_none_07d_002.nii.csv");
ts = zscore(meanTS);


T = size(ts,1);
N = size(ts,2);

% load auxiliary files: Yeo7 node assignments, colormap
load yeo7_200_labels
load mycmap

% indices for upper triangle of NxN matrix
inds = find(triu(ones(N),1));

%% Compute Basic Constructs
addpath('/Users/rachelfox/Downloads/BrainConnectivityToolbox');

% compute edge time series    
ets = fcn_edgets(ts);

% RSS (frame-wise co-fluctuation amplitude)
rss = sum(ets.^2,2).^0.5;
   
% bipartitions, expressed as a set of partitions (communities coded 1, 2)
cis = double((ts>0)+1)';

% FC matrix (computed as Pearson correlation of time series)
FC = corr(ts);

% FC matrix (computed as mean of ets)
mets = sum(ets)/(T-1);
FC2 = zeros(N);
FC2(inds) = mets; FC2 = FC2+FC2';

% similarity btw the two ways of computing FC (equal to one as it is
% mathematically equivalent)
corr(FC(inds),FC2(inds))

% agreement matrix of bipartitions (all frames), including analytic null
% (see Jeub et al, 2018)
anull = sum(sum([sum(cis==1)./N; sum(cis==2)./N].*...
   [(sum(cis==1)-1)./(N-1); (sum(cis==2)-1)./(N-1)],2));
A = (agreement(cis) - anull)./T;

% similarity btw A and FC
corr(A(inds),FC(inds))

% select top/bottom 'rssfrac' RSS frames (110 = 10 percent)
% one can use any criterion to select a subset of frames...
rssfrac = 110;
[rsssorted, rssinds] = sort(rss,'descend');
top = rssinds(1:rssfrac);
bot = rssinds(end-rssfrac+1:end);

% FC/A computed from the top/bot frames
mets_top = sum(ets(top,:))./(rssfrac-1);
FCtop = zeros(N);
FCtop(inds) = mets_top;
mets_bot = sum(ets(bot,:))./(rssfrac-1);
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

% MI btw bipartitions over all frames
[~, MIframes] = partition_distance(cis,cis);

% load template basis set (63 templates)
load template_basis_set_200.mat
cit = cit(1:124, :);

% MI btw each frame and each template
[~, MItemplates] = partition_distance(cis,cit);

% select best matching template
[maxMI, bestmatch] = max(MItemplates');

% determine template frequencies
template_frequency = histcounts(bestmatch,[0.5:1:63.5]);

%% Make Figures

% display FC and A
figure('position',[200 100 800 600])
subplot(2,2,1)
imagesc(FC,[-1 1]); axis square; colorbar
title('functional connectivity')
subplot(2,2,2)
imagesc(A,[-0.3 0.3]); axis square; colorbar
title('agreement')
subplot(2,2,3)
imagesc(Atop,[-0.3 0.3]); axis square; colorbar
title('agreement - top RSS')
subplot(2,2,4)
imagesc(Abot,[-0.3 0.3]); axis square; colorbar
title('agreement - bottom RSS')
colormap(flipud(mycmap))

% display ets and RSS
figure('position',[200 100 800 600])
subplot(2,1,1)
imagesc(ets',[-2 2])
title('edge time series')
xlabel('frames','FontSize',12);
ylabel('edge time series','FontSize',12);
subplot(2,1,2)
plot(rss)
axis([0 T 0 500])
title('co-fluctuation amplitude (RSS)')
xlabel('frames','FontSize',12);
ylabel('RSS amplitude','FontSize',12);
colormap(flipud(mycmap))

% display bipartitions, template basis set, and best matching bipartions
% for template 63 (unrectified)
figure('position',[200 100 800 600])
subplot(2,1,1)
imagesc(cis)
xlabel('frames','FontSize',12)
ylabel('nodes','FontSize',12)
colormap(gray)
subplot(2,2,3)
imagesc(cit)
xlabel('template number','FontSize',12)
ylabel('nodes','FontSize',12)
subplot(2,2,4)
ff = find(bestmatch==63);
imagesc(opt_tmpl);
%imagesc(cis(:,ff));
xlabel('match (optimized template)','FontSize',12)
ylabel('nodes','FontSize',12)
colormap(gray)

% display MI of bipartitions btw frames
figure
imagesc(MIframes,[0 0.5]); axis square;     % scale capped at 0.5
colormap(flipud(gray)); colorbar
xlabel('frames','FontSize',12)
ylabel('frames','FontSize',12)

% display time series of MI of bipartitions for a few sample templates
figure
hold on
pp = plot(MItemplates(:,1)); set(pp,'Color','b')
pp = plot(MItemplates(:,17)); set(pp,'Color','g')
pp = plot(MItemplates(:,63)); set(pp,'Color','r')
axis([0 T 0 0.4])
box on
ylabel('similarity (MI)','FontSize',12)
legend('template 1','template 17','template 63','FontSize',10)

% display template frequency
figure
bar(template_frequency)
xlabel('template basis set','FontSize',12)
ylabel('count','FontSize',12)

%% Template optimization and filtering
addpath('/Users/rachelfox/Downloads/temporal-behavior-optimization-main')
datapath = '/Users/rachelfox/dFC_Toolboxes/Leida/CSVs_with_regions_stim_pattern_CIMT/JustShams';
file_list = dir(fullfile(datapath, '*.csv'));
first_file = fullfile(datapath, file_list(1).name);
[len, w] = size(readmatrix(first_file));
num_files = length(file_list);
TS = zeros(len, w, num_files);

for i = 1:num_files
    file_path = fullfile(datapath, file_list(i).name);
    TS(:, :, i) = readmatrix(file_path);
end

% use forelimb stim for behavior 
behav = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, ...
                     1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, ...
                     1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, ...
                     1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...
                     0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...
                     0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, ...
                     0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, ...
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, ...
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
                     1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, ...
                     1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, ...
                     1, 1, 1, 1, 1, 1, 1];
behav = behav';
numfr = 25; % top frames to select
%[ini_tmpl,opt_tmpl,perf] = optim_template(TS,beh,numfr,inds_cv,ini_tmpl,H,hfrac,T0);

random_behav = rand(24, 1) * 100; % random score
[ini_tmpl,opt_tmpl,perf] = optim_template(TS,behav,numfr, 'H',500); % creates optimized template to correlate with behavior

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

% mi between frames and opt template (make it 1 and 2 instead of 0 and 1)
opt_template = opt_tmpl + 1;
[~, MItemplates] = partition_distance(cis,opt_template);

%[frames] = framewise_filter(filter,numbins,selfr,order,strat)
[frames] = framewise_filter(opt_tmpl,'descend'); % takes TS and binning strategy to select frames for each bin

%frames = true(250, 1); % change to logical vector of certain frames based on filtering
ftype = 1; % 1 from ets 2 from bipartition
fig = true;
meanTS = mean(TS, 3);
%bipart = double((meanTS>0)+1)';
[Xc,moments] = aggregate_component(meanTS,frames,ftype,fig); % can only do 1 subject at time or average subject

corrs = [];
[rsssorted, rssinds] = sort(rss,'descend');
bin_edges = linspace(min(rsssorted), max(rsssorted), 10+1);
bin_indices = discretize(rsssorted, bin_edges); % doesn't match
for i=1:10
    c = corr(opt_tmpl, frames(:, i));
    corrs = [corrs; c];
end
imagesc(corrs); colormap('jet'); colorbar;

ets = fcn_edgets(meanTS);
rss = sum(ets.^2,2).^0.5;
bipart = double((zscore(meanTS)>0)+1);
top_indices = find(rss >= prctile(rss, 90));
TS_repopulated = zeros(size(bipart));
%TS_repopulated(top_indices, :, :, :) = meanTS(top_indices, :, :, :);
TS_repopulated(top_indices, :) = bipart(top_indices, :);
TS_repopulated(TS_repopulated == 0) = Inf;
figure;
imagesc(TS_repopulated');  % Averaging over subjects & runs
colorbar;
colormap('gray');
title('Filtered Bipartition Time Series (Top 10% RSS)');
xlabel('Time');
ylabel('Edges');
