function net = init_denoising_autoencoder_encoder(prev_net, feat_size, layer_dims, reduced_dim, f)

target_dims = [layer_dims, reduced_dim];

% Meta parameters
net.meta.inputSize = [feat_size(1) feat_size(2) layer_dims] ;
net.meta.trainOpts.learningRate = 0.001 ;
net.meta.trainOpts.numEpochs = 20 ;
net.meta.trainOpts.batchSize = 100 ;


if(isempty(prev_net))
    net.layers = {} ;
else
    net.layers = prev_net.layers;
end

% encoder
for ii = 1:(length(target_dims) - 1)
    net.layers{end+1} = struct('type', 'conv', ...
                               'weights', {{f*randn(3,3,target_dims(ii),target_dims(ii+1), 'single'), zeros(1, target_dims(ii+1), 'single')}}, ...
                               'stride', 1, ...
                               'pad', [1,1,1,1]) ;
    net.layers{end+1} = struct('type', 'relu') ;
end


% Fill in defaul values
net = vl_simplenn_tidy(net) ;