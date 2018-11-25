function output = swap(input)

sz = size(input);

output = input(sz(1):-1:1, :, :, :, :);