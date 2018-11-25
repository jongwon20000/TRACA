function output = vectorize(input)

output = reshape(input, [size(input,1)*size(input,2), 1, size(input,3), size(input,4)]);