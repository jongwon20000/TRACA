function [performance, online_time] = tracker(BENCHMARK_PATH, opt, MATCONVNET_PATH, PIOTR_PATH)

% library load
run([MATCONVNET_PATH 'matlab/vl_setupnn.m']);
addpath(genpath([PIOTR_PATH 'toolbox/']));
addpath('./tools');
addpath('./ae_train');
addpath('./cf_tracker');


% tracking dataset path
scene_list = dir(BENCHMARK_PATH);
scene_list = scene_list(3:end);

% init
online_time = 0;
performance = 0;

% parameters
target_layers = 7;

reduced_dim = [128, 64];

image_size = [224, 224];

feat_size = [26, 26];

orth_lambda = opt.orth_lambda;
finetune_epoch = opt.finetune_iter;
learning_rate = opt.finetune_rate;

val_min = opt.val_min;
val_lambda = opt.val_lambda;

roi_resize_factor = 2.5;
scale_ratio = opt.scale_ratio;
scale_variation = opt.scale_variation;


% cf parameters
output_sigma_factor = opt.output_sigma_factor;
lambda = opt.lambda;
gamma = opt.gamma;
epsilon = 0.00001;
fftw('planner', 'estimate');

% redetection parameters
redetect_n_frame = opt.redetect_n_frame;
redetect_eps = opt.redetect_eps;

gpus = 1;


% vgg network load
full_vggnet = load('./network/imagenet-vgg-m-2048');
full_vggnet.layers = full_vggnet.layers(1:(end-4)); % trimming the network
vggnet = full_vggnet;
vggnet.layers = vggnet.layers(1:max(target_layers)); % trimming the network

% pretrained dAE load
multi_daenet = load('./network/multi_daenet');
multi_daenet = multi_daenet.multi_dae;
prior_net = load('./network/prior_network');
prior_net = prior_net.prior_net;
prior_net.layers = prior_net.layers(1:(end-1));

% correlation filter initialization
output_sigma = sqrt(prod(ceil(feat_size/roi_resize_factor))) * output_sigma_factor;
bbox_bmap = zeros(feat_size);
bbox_bmap(ceil(feat_size(1)/2-feat_size(1)/2/roi_resize_factor):floor(feat_size(1)/2+feat_size(1)/2/roi_resize_factor) ...
    , ceil(feat_size(2)/2-feat_size(2)/2/roi_resize_factor):floor(feat_size(2)/2+feat_size(2)/2/roi_resize_factor)) = 1;
yv = single(zeros(feat_size(1), feat_size(1)));
yv_temp = gaussian_shaped_labels(output_sigma, feat_size);
yv_temp = yv_temp(1:end, 1);
for jj = 1:length(yv_temp)
    yv(:, jj) = single(circshift(yv_temp, [jj-1,0]));
end
yf = single(fft(vec(gaussian_shaped_labels(output_sigma, feat_size))));
yf2 = single(fft2(gaussian_shaped_labels(output_sigma, feat_size)));
cos_window = single(hann(feat_size(1)) * hann(feat_size(2))');
if(gpus > 0)
    yf = gpuArray(yf);
    bbox_bmap = gpuArray(bbox_bmap);
end
cf_params.yv = yv;
cf_params.yf = yf;
cf_params.yf2 = repmat(yf2, [1,1,val_min]);
cf_params.bbox_bmap = bbox_bmap;
cf_params.cos_window = cos_window;
cf_params.gamma = gamma;
cf_params.lambda = lambda;
cf_params.epsilon = epsilon;


% for scenes,
for scene_idx = 1:length(scene_list)
    
    fprintf('%s start', scene_list(scene_idx).name);
    
    %finetune    
    if(gpus > 0)
        vggnet = vl_simplenn_move(vggnet, 'gpu');
        prior_net = vl_simplenn_move(prior_net, 'gpu');
        full_vggnet = vl_simplenn_move(full_vggnet, 'gpu');
    end
    
    % roi extraction
    [img_files, pos, target_sz, ground_truth, video_path] = load_video_info(BENCHMARK_PATH, scene_list(scene_idx).name);
        
    im = imread([BENCHMARK_PATH, scene_list(scene_idx).name, '/img/', img_files{1}]);
    window_sz = round(target_sz.*roi_resize_factor);
    init_window_sz = window_sz;
    
    % finetune augmentation
    patch = get_subwindow(im, pos, window_sz);    
    patch = cat(4, patch, imgaussfilt(patch(:,:,:,1), 0.5));
    patch = cat(4, patch, imgaussfilt(patch(:,:,:,1), 1.0));
    patch = cat(4, patch, imgaussfilt(patch(:,:,:,1), 1.5));
    patch = cat(4, patch, imgaussfilt(patch(:,:,:,1), 2.0));  
    patch = cat(4, patch, patch(:, end:-1:1, :, 1));
    patch = cat(4, patch, patch(end:-1:1, :, :, 1));
    
    patch = imresize(patch, image_size);
    
    
    % dAE finetuning
    mbatch = single(patch);
    
    if(gpus > 0)
        mbatch = gpuArray(mbatch);
    end
    
    mbatch = bsxfun(@minus, mbatch, full_vggnet.meta.normalization.averageImage);
    
    vgg_res = vl_simplenn(full_vggnet, mbatch);
    
    % multi-daenet selection
    sel_res = vl_simplenn(prior_net, vgg_res(end).x);
    [~, dae_idx] = max(sel_res(end).x(:,:,:,1));
    
    daenet = multi_daenet{dae_idx}{1};
    if(gpus > 0)
        for jj = 1:size(daenet,1)
            daenet(jj,1) = vl_simplenn_move(daenet(jj,1), 'gpu');
            daenet(jj,2) = vl_simplenn_move(daenet(jj,2), 'gpu');
        end
    end
    dae_res = cell(size(daenet,1), 2);
    
    mfeat = vgg_res(target_layers+1).x;
    
    % selected expert ae fine-tuning
    l2_error_stack = zeros(1, finetune_epoch);
    orth_error_stack = zeros(1, finetune_epoch);
    for epoch_idx = 1:finetune_epoch
                
        % multi-stage forward        
        for jj = 1:size(daenet,1)
            dae_res{jj,1} = vl_simplenn(daenet(jj,1), mfeat, [], dae_res{jj,1});
            dae_res{jj,2} = vl_simplenn(daenet(jj,2), dae_res{jj,1}(end).x, [], dae_res{jj,2});
        end
        
        % multi-stage backward
        [dae_res, w] = multi_stage_backward_finetune(daenet, dae_res, mfeat, mfeat, orth_lambda, cf_params);
        
        % multi-stage gradient update
        % encoder update
        for jj = 1:size(daenet,1)
            for kk = 1:size(dae_res{jj,1}, 2)
                if(~isempty(dae_res{jj,1}(kk).dzdw))
                    daenet(end,1).layers{kk}.weights{1} = ...
                        daenet(end,1).layers{kk}.weights{1} - learning_rate*dae_res{jj,1}(kk).dzdw{1};
                    daenet(end,1).layers{kk}.weights{2} = ...
                        daenet(end,1).layers{kk}.weights{2} - learning_rate*dae_res{jj,1}(kk).dzdw{2};
                end
            end
            for kk = 1:size(daenet(jj,1).layers, 2)
                if(~isempty(daenet(jj,1).layers{kk}.weights))
                    daenet(jj,1).layers{kk}.weights =  daenet(end,1).layers{kk}.weights;
                end
            end
        end
        
        % decoder update
        for jj = 1:size(daenet,1)
            for kk = 1:size(dae_res{jj,2}, 2)
                if(~isempty(dae_res{jj,2}(end-kk+1).dzdw))
                    daenet(end,2).layers{end-kk+2}.weights{1} = ...
                        daenet(end,2).layers{end-kk+2}.weights{1} - learning_rate*dae_res{jj,2}(end-kk+1).dzdw{1};
                    daenet(end,2).layers{end-kk+2}.weights{2} = ...
                        daenet(end,2).layers{end-kk+2}.weights{2} - learning_rate*dae_res{jj,2}(end-kk+1).dzdw{2};
                end
            end
            for kk = 1:size(daenet(jj,2).layers, 2)
                if(~isempty(daenet(jj,2).layers{end-kk+1}.weights))
                    daenet(jj,2).layers{end-kk+1}.weights =  daenet(end,2).layers{end-kk+1}.weights;
                end
            end
            
        end

        % loss estimation
        l2_err = 0;
        orth_err = 0;
        for jj = 1:size(daenet,1)
            % 2norm
            l2_err = l2_err + mean((dae_res{jj,2}(end).x(:) - mfeat(:)).*(dae_res{jj,2}(end).x(:) - mfeat(:)));
            % orth loss
            if(orth_lambda > 0)
                ww = bsxfun(@times, sum(bsxfun(@times, permute(w{jj}, [1,2,4,3,5]), w{jj}),1), ...
                    permute(1-eye(size(w{jj},3), size(w{jj},3)), [3,4,1,2]));
                wi2 = pagefun(@mtimes, permute(w{jj}, [2,1,3,4,5]), w{jj}) + epsilon;
                wk2 = permute(wi2, [1,2,4,3,5]);
                l2w = mean(vec(bsxfun(@times, bsxfun(@times, ww.*ww, 1./wi2),1./wk2)));
            else
                l2w = 0;
            end
            orth_err = orth_err + l2w;
        end
        epoch_l2_err = l2_err;
        epoch_orth_err = orth_err;

        % Loss visualization
        l2_error_stack(epoch_idx) = gather(epoch_l2_err);
        orth_error_stack(epoch_idx) = gather(epoch_orth_err);
        if(opt.visualization > 0)
            figure(1000); 
            subplot(1,3,1); plot(l2_error_stack(1:epoch_idx)); title('l2 loss');
            subplot(1,3,2); plot(orth_error_stack(1:epoch_idx)); title('orth loss');
            subplot(1,3,3); plot(l2_error_stack(1:epoch_idx) + orth_lambda*orth_error_stack(1:epoch_idx)); title('entire loss');
            drawnow;
        end
        
    end
    
    
    if(opt.visualization > 0)
        update_visualization = show_video(img_files, video_path);
    end
    
    
    %% tracking sequence
    encoding_number = length(reduced_dim);
    positions = zeros(size(ground_truth,1), 4);
    positions(1,1:2) = pos;
    positions(1,3:4) = target_sz;
    redetection = 0;
    
    % for first frame
    mfeat_origin = mfeat(:,:,:,1);
    if(encoding_number < 1)
        x = mfeat_origin;
    else
        curr_res = vl_simplenn(daenet(encoding_number,1), mfeat_origin);
        x = curr_res(end).x;
    end
    ocp_ratio = sum(sum(bsxfun(@times, abs(x), cf_params.bbox_bmap), 1), 2) ./ (sum(sum(abs(x), 1), 2) + epsilon);

    % foreground layer selecting
    sorted_ocp = sort(ocp_ratio, 'descend');
    sel_layers = find(ocp_ratio >= sorted_ocp(val_min));
        
    xf = fft2(bsxfun(@times, cf_params.cos_window, x(:,:,sel_layers)));    
    wf = gather(bsxfun(@times, cf_params.yf2, xf ./ (xf.*conj(xf) + cf_params.lambda)));
    
    % integrate the vgg-net and the fine-tuned encoder
    feat_net = vggnet;
    for ii = 1:numel(daenet(encoding_number, 1).layers)
        feat_net.layers(end + 1) = daenet(encoding_number, 1).layers(ii);
    end
    feat_net.layers{end-1}.weights{1} = feat_net.layers{end-1}.weights{1}(:,:,:,sel_layers);
    feat_net.layers{end-1}.weights{2} = feat_net.layers{end-1}.weights{2}(1,sel_layers);
        
    
    % time check
    timeh = 0;   
    feat_res = [];
    for frame_idx = 2:length(img_files(:))
        
        
       %% tracking
        % ROI extraction
        prev_im = im;
        
        im = imread([BENCHMARK_PATH, scene_list(scene_idx).name, '/img/', img_files{frame_idx}]);   
                
        tic;
                
        if(redetection > 0)
            patches_test = zeros(image_size(1), image_size(2), size(im, 3), scale_variation+1, 'single');
        else
            patches_test = zeros(image_size(1), image_size(2), size(im, 3), scale_variation, 'single');
        end
        
        % multi-scale sample extraction
        for iSc = -floor(scale_variation/2):floor(scale_variation/2)
            patch = get_subwindow(im, pos, round(window_sz*(scale_ratio.^iSc)));       
            patches_test(:,:,:,iSc+ceil(scale_variation/2)) = single(imResample_fast(patch, image_size));     
        end
        
        % redetection ROI extraction
        if(redetection > 0)
            red_patch = get_subwindow(im, prev_pos(1:2), round(prev_pos(3:4)));
            patches_test(:,:,:,end) = single(imResample_fast(red_patch, image_size));
        end
        
        % update ROI extraction (from the previous image)
        patch2 = get_subwindow(prev_im, pos, window_sz);        
        patches_update = single(imResample_fast(patch2, image_size));
        
        % integrate all the samples
        patches = cat(4, patches_test, patches_update);
        
        % feature extraction
        if(gpus > 0)
            mbatch = gpuArray(patches);
        else
            mbatch = patches;
        end
        mbatch = bsxfun(@minus, mbatch, feat_net.meta.normalization.averageImage);
        feat_res = vl_simplenn_fast(feat_net, mbatch, [], feat_res);
                
        mfeat = gather(feat_res(end).x);
                        
        xf = fft2(bsxfun(@times, cf_params.cos_window, mfeat));
        xf_update = xf(:,:,:,end);
        xf_test = xf(:,:,:,1:(end-1));
        
                
        % correlation filter estimation (update)
        wf_curr = cf_params.yf2.* xf_update ./ (xf_update.*conj(xf_update) + cf_params.lambda);
        wf = (1-cf_params.gamma)*wf + cf_params.gamma*wf_curr;  
                                
        % when the redetction scheme is running,
        if(redetection > 0)
            % normal response
            res = real(ifft2(bsxfun(@times, wf, conj(xf_test(:,:,:,1:scale_variation)))));
            
            % validate the CFs
            [r_max_v, r_max_idx] = max(res, [], 1);
            [~, c_max_idx] = max(r_max_v, [], 2);
            temp_ideal_y = bsxfun(@times, permute(yv(:, r_max_idx(c_max_idx)), [1,3,4,2]),...
                permute(yv(:, c_max_idx), [3,1,4,2]));
            ideal_y = reshape(temp_ideal_y, size(res));
            
            % integrate the multiple responses from the channel-wise CFs
            val = mean(mean((res - ideal_y).^2, 1), 2);
            response = permute(mean(bsxfun(@times, exp(-val_lambda*val), res), 3), [1,2,4,3]);
            
            % redetection response & validation
            red_res = real(ifft2(prev_wf.*conj(xf_test(:,:,:,end))));
            [red_r_max_v, red_r_max_idx] = max(red_res, [], 1);
            [~, red_c_max_idx] = max(red_r_max_v, [], 2);            
            red_ideal_y = bsxfun(@times, permute(yv(:, red_r_max_idx(red_c_max_idx)), [1,3,2]),...
                permute(yv(:, red_c_max_idx), [3,1,2]));
            
            red_val = mean(mean((red_res - red_ideal_y).^2, 1), 2);
            red_response = mean(bsxfun(@times, exp(-val_lambda*red_val), red_res), 3);   
            
        % Otherwise,
        else
            
            % response
            res = real(ifft2(bsxfun(@times, wf, conj(xf_test))));
                        
            % validation
            [r_max_v, r_max_idx] = max(res, [], 1);
            [~, c_max_idx] = max(r_max_v, [], 2);
            temp_ideal_y = bsxfun(@times, permute(yv(:, r_max_idx(c_max_idx)), [1,3,4,2]),...
                permute(yv(:, c_max_idx), [3,1,4,2]));                        
            ideal_y = reshape(temp_ideal_y, size(res));            
            
            % integrate the multiple responses from the channel-wise CFs
            val = mean(mean((res - ideal_y).^2, 1), 2);
            response = permute(mean(bsxfun(@times, exp(-val_lambda*val), res), 3), [1,2,4,3]);  
           
        end
        
        
        % Validation score comparison
        if(redetection > 0)
            cf_params.gamma = opt.redetect_gamma;
            redetect_success = max(red_response(:)) > max(response(:));
            redetection = redetection - 1;
        else
            cf_params.gamma = opt.gamma;
            redetect_success = 0;
        end
        
        % redetection update
        if(frame_idx > redetect_n_frame)
            if(max(response(:)) < max_res*redetect_eps && redetection == 0)
                redetection = redetect_n_frame;
                prev_wf = wf;
                prev_pos = [pos(1:2), window_sz];
            end
        end
        if(frame_idx == 2)
            max_res = max(response(:));
            % When the initial maximum response is too small, turn off the
            % redetection module
            if(max_res < 0.35)
                redetection = -1;
            end
        else
            max_res = (1 - cf_params.gamma)*max_res + cf_params.gamma*max(response(:));
        end
               
        
        % the redetection CFs wins the normal CFs
        if(redetect_success)
            
            redetection = 0;
            
            %find the target position            
            [vert_delta, horiz_delta] = find_peak(permute(red_response, [1,2,4,3]), feat_size, prev_pos(3:4));
                        
            %exception
            if(isnan(vert_delta) || isnan(horiz_delta))
                vert_delta = 0; horiz_delta = 0;
            end
            
            wf = prev_wf;
            pos = prev_pos(1:2) - round([vert_delta, horiz_delta]);
            window_sz = round(prev_pos(3:4));
                        
        else
            
            %find the target position
            scale_delta = find(max(max(response,[],1),[],2) == max(response(:)),1);            
            [vert_delta, horiz_delta] = find_peak(response(:,:,2), feat_size, window_sz);
            
            %exception handling
            if(isnan(vert_delta) || isnan(horiz_delta))
                vert_delta = 0; horiz_delta = 0; scale_delta = ceil(scale_variation/2);
            elseif(length(scale_delta) > 1)
                scale_delta = ceil(scale_variation/2);
            end
            
            % LImit the size of the target
            if(init_window_sz/window_sz > 10 || init_window_sz/window_sz < 1/10)
                scale_delta = ceil(scale_variation/2);
            end
            
            pos = pos - round([vert_delta, horiz_delta]);
            window_sz = round(window_sz * scale_ratio^(scale_delta-ceil(scale_variation/2)));
                        
        end
        
        % time check
        timeh = timeh + toc;
                
        % stack the position & target size
        target_sz = round(window_sz / roi_resize_factor);
        positions(frame_idx,1:2) = pos;
        positions(frame_idx,3:4) = target_sz;
                                
        
        
        %% visualization
        if(opt.visualization > 0)
            box = [pos([2,1]) - target_sz([2,1])/2, target_sz([2,1])];
            stop = update_visualization(frame_idx, box);
            drawnow;
        end
        
    end
    
    len = min(size(ground_truth,1), size(positions,1));
    distance = sqrt((positions(1:len,1)-ground_truth(1:len,1)).^2 + (positions(1:len,2)-ground_truth(1:len,2)).^2);
    temp_perf = sum(distance < 20) / size(distance,1);
    if(opt.visualization > 0)
        disp([scene_list(scene_idx).name, ' (' num2str(length(sel_layers)), ' mods.)']);
        disp(['20px precision = ', num2str(temp_perf)]);
        disp([num2str(dae_idx) '-th dAE was selected']);
    end
    performance = performance + temp_perf;
    online_time = online_time + timeh / (length(img_files(:)) - 1);    
    
    save(['result/' scene_list(scene_idx).name], 'positions');
    
end

performance = performance / length(scene_list);
online_time = online_time / length(scene_list);
