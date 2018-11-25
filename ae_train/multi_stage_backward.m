function [dae_res, w] = multi_stage_backward(daenet, ii, dae_res, mfeat, n_mfeat, orth_lambda, cf_params)



%estimate the correlation filters (for the orthogonality term)
w = cell(size(daenet{ii},1),1);
dEdx = cell(size(daenet{ii},1),1);
if(orth_lambda > 0.0)
    epsilon = cf_params.epsilon;
    yf = cf_params.yf{ii};
    lambda = cf_params.lambda;
    
    for jj = 1:size(daenet{ii},1)
        xf = fft(vectorize(dae_res{ii}{jj,1}(end).x));
        w{jj} = real(permute(ifft(bsxfun(@times, xf, yf) ./ (conj(xf).*xf + lambda)), [1,2,3,5,4]));
        ww = bsxfun(@times, sum(bsxfun(@times, permute(w{jj}, [1,2,4,3,5]), w{jj}),1), ...
            permute(1-eye(size(w{jj},3), size(w{jj},3)), [3,4,1,2]));
        wi2 = pagefun(@mtimes, permute(w{jj}, [2,1,3,4,5]), w{jj}) + epsilon;
        wk2 = permute(wi2, [1,2,4,3,5]);
        dEdw = sum(bsxfun(@times, bsxfun(@times, bsxfun(@times, ww, 1./wi2), 1./wk2), w{jj}),3) ...
            - bsxfun(@times, sum(bsxfun(@times, bsxfun(@times, ww.*ww, 1./wi2),1./wk2./wk2), 3), permute(w{jj}, [1,2,4,3,5]));
        dEdwf = permute(fft(dEdw), [1,2,4,5,3]);
        dEdx1f = -2*conj(xf).*real(xf.*bsxfun(@times, yf, conj(dEdwf))) ./ (xf.*conj(xf)+lambda) ./ (xf.*conj(xf)+lambda);
        dEdx2f = bsxfun(@times, conj(yf), dEdwf) ./ (xf.*conj(xf)+lambda);
        dEdx{jj} = reshape(real(ifft(dEdx1f + dEdx2f)), size(dae_res{ii}{jj,1}(end).x));
    end
end

for jj = 1:size(daenet{ii},1)
    % 1norm
%     dEdz = max(min(dae_res{ii}{jj,2}(end).x - mfeat{ii}, 1), -1);

    % 2norm
    dEdz = dae_res{ii}{jj,2}(end).x - mfeat{ii};
    dae_res{ii}{jj,2} = vl_simplenn(daenet{ii}(jj,2), dae_res{ii}{jj,1}(end).x, dEdz, dae_res{ii}{jj,2}, 'SkipForward', true);
    
    % orthogonality loss
    if(orth_lambda > 0.0)
        orth_der = dEdx{jj};
    else
        orth_der = 0;
    end
    dae_res{ii}{jj,1} = vl_simplenn(daenet{ii}(jj,1), n_mfeat, dae_res{ii}{jj,2}(1).dzdx + orth_lambda*orth_der, dae_res{ii}{jj,1}, 'SkipForward', true);
    
end

