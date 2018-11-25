function net = init_denoising_autoencoder_decoder(prev_net, feat_size, layer_dims, reduced_dim, f)

target_dims = [layer_dims, reduced_dim];

% Meta parameters
net.meta.inputSize = [feat_size(1) feat_size(2) layer_dims] ;
net.meta.trainOpts.learningRate = 0.001 ;
net.meta.trainOpts.numEpochs = 20 ;
net.meta.trainOpts.batchSize = 100 ;


net.layers = {} ;

% decoder
for ii = length(target_dims):-1:2
    net.layers{end+1} = struct('type', 'conv', ...
                               'weights', {{f*randn(3,3,target_dims(ii),target_dims(ii-1), 'single'), zeros(1, target_dims(ii-1), 'single')}}, ...
                               'stride', 1, ...
                               'pad', [1,1,1,1]) ;
   if(~isempty(prev_net))
        net.layers{end+1} = struct('type', 'relu') ;
   end
end

if(~isempty(prev_net))
    net.layers = cat(2, net.layers, prev_net.layers);
end

% Fill in defaul values
net = vl_simplenn_tidy(net) ;