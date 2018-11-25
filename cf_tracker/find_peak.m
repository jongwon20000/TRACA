function [vert_delta, horiz_delta] = find_peak(response, res_sz, window_sz)

[vert_delta, horiz_delta] = find(response == max(response(:)), 1);

%Delta interpolation!
interp_size = 5;

ys = floor(vert_delta) + (1:interp_size) - ceil(interp_size/2);
xs = floor(horiz_delta) + (1:interp_size) - ceil(interp_size/2);

%check for out-of-bounds coordinates, and set them to the values at
%the borders
xs(xs < 1) = xs(xs < 1) + size(response,2);
ys(ys < 1) = ys(ys < 1) + size(response,1);
xs(xs > size(response,2)) = xs(xs > size(response,2)) - size(response,2);
ys(ys > size(response,1)) = ys(ys > size(response,1)) - size(response,1);

temp = (1:interp_size) - ceil(interp_size/2);
[szx, szy] = meshgrid(temp, temp);

weight_sum = permute(sum(sum(bsxfun(@times, response(ys, xs), cat(3, szx, szy)), 1), 2), [1,3,2]);
res_sum = sum(sum(response(ys, xs)));


vert_delta = vert_delta + weight_sum(2) / res_sum;
horiz_delta = horiz_delta + weight_sum(1) / res_sum;

if vert_delta > res_sz(1) / 2,  %wrap around to negative half-space of vertical axis
    vert_delta = vert_delta - res_sz(1);
end
if horiz_delta > res_sz(2) / 2,  %same for horizontal axis
    horiz_delta = horiz_delta - res_sz(2);
end
vert_delta = (vert_delta-1) / res_sz(1) * window_sz(1);
horiz_delta = (horiz_delta-1) / res_sz(2) * window_sz(2);