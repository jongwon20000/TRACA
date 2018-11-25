function daenet = init_denoising_autoencoder(feat_size, target_layers, layer_dims, reduced_dim, ff)

rng('default');
rng(10) ;

daenet = cell(length(target_layers),1);
for ii = 1:length(target_layers)
    daenet{ii}(1,1) = init_denoising_autoencoder_encoder([], feat_size{ii}, layer_dims(ii), reduced_dim{ii}(1), ff);
    daenet{ii}(1,2) = init_denoising_autoencoder_decoder([], feat_size{ii}, layer_dims(ii), reduced_dim{ii}(1), ff);
    for jj = 2:size(reduced_dim{ii},2)
        daenet{ii}(jj,1) = init_denoising_autoencoder_encoder(daenet{ii}(jj-1,1), feat_size{ii}, reduced_dim{ii}(jj-1), reduced_dim{ii}(jj), ff);
        daenet{ii}(jj,2) = init_denoising_autoencoder_decoder(daenet{ii}(jj-1,2), feat_size{ii}, reduced_dim{ii}(jj-1), reduced_dim{ii}(jj), ff);
    end
end
 