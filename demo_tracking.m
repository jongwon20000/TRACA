%% Libarary Path
MATCONVNET_PATH = '../matconvnet/';
PIOTR_PATH = '../piotr_toolbox/';

%% Parameter Setting
opt.orth_lambda = 1000;
opt.finetune_iter = 10;
opt.finetune_rate = 0.000000001;

opt.scale_ratio = 1.015;
opt.scale_variation = 3; %always odd number
opt.val_min = 25;
opt.val_lambda = 50.0;

opt.output_sigma_factor = 0.05;
opt.lambda = 1.0;
opt.gamma = 0.025;

opt.redetect_n_frame = 50;
opt.redetect_eps = 0.7;
opt.redetect_gamma = 0.0025;

opt.visualization = 1;

%% Data Path
BENCHMARK_PATH = './sequence/';


%% Run tracker
[output_perf, online_time] = tracker(BENCHMARK_PATH, opt, MATCONVNET_PATH, PIOTR_PATH);


%% Disp. output
disp(['perf = ' num2str(output_perf*100) ' (' num2str(online_time*1000) 'ms)']);

