function B = imResample_fast( A, scale, method, norm )
% Fast bilinear image downsampling/upsampling.
%
% Gives similar results to imresize with the bilinear option and
% antialiasing turned off if scale is near 1, except sometimes the final
% dims are off by 1 pixel. For very small values of the scale imresize is
% faster but only looks at subset of values of original image.
%
% This code requires SSE2 to compile and run (most modern Intel and AMD
% processors support SSE2). Please see: http://en.wikipedia.org/wiki/SSE2.
%
% USAGE
%  B = imResample( A, scale, [method], [norm] )
%
% INPUT
%  A        - input image (2D or 3D single, double or uint8 array)
%  scale    - scalar resize factor [s] of target height and width [h w]
%  method   - ['bilinear'] either 'bilinear' or 'nearest'
%  norm     - [1] optionally multiply every output pixel by norm
%
% OUPUT
%   B       - resampled image
%
% EXAMPLE
%  I=single(imread('cameraman.tif')); n=100; s=1/2; method='bilinear';
%  tic, for i=1:n, I1=imresize(I,s,method,'Antialiasing',0); end; toc
%  tic, for i=1:n, I2=imResample(I,s,method); end; toc
%  figure(1); im(I1); figure(2); im(I2);
%
% See also imresize
%
% Piotr's Computer Vision Matlab Toolbox      Version 3.24
% Copyright 2014 Piotr Dollar.  [pdollar-at-gmail.com]
% Licensed under the Simplified BSD License [see external/bsd.txt]

% figure out method and get target dimensions

% use bilinear interpolation
B=imResampleMex(A,scale(1),scale(2),1);
