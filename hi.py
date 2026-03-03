from itertools import combinations_with_replacement

augmentations = ['gaussian', 'normal', 'scale', 'randomstrech', 'randomcrop', 'fft']

pairs = list(combinations_with_replacement(augmentations, 2))

print(len(pairs))  # 21
print(pairs)