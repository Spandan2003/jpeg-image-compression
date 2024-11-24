import cv2
import os
import numpy as np
import math
# import scipy
# from scipy import ndimage, signal
import heapq
from collections import defaultdict, Counter
import matplotlib.pyplot as plt

Q_50_matrix = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                        [12, 12, 14, 19, 26, 58, 60, 55],
                        [14, 13, 16, 24, 40, 57, 69, 56],
                        [14, 17, 22, 29, 51, 87, 80, 62],
                        [18, 22, 37, 56, 68, 109, 103, 77],
                        [24, 25, 55, 64, 81, 104, 113, 92],
                        [49, 64, 78, 87, 103, 121, 120, 101],
                        [72, 92, 95, 98, 112, 100, 103, 99]], dtype=float)

##############  HUFFMAN ENCODING ##############
# Helper to compute size of a value
def compute_size(value):
    if value == 0:
        return 0
    return int(np.floor(np.log2(abs(value))) + 1)

# Zigzag order for an 8x8 block
zigzag_order = [
    0,  1,  5,  6, 14, 15, 27, 28,
    2,  4,  7, 13, 16, 26, 29, 42,
    3,  8, 12, 17, 25, 30, 41, 43,
    9, 11, 18, 24, 31, 40, 44, 53,
   10, 19, 23, 32, 39, 45, 52, 54,
   20, 22, 33, 38, 46, 51, 55, 60,
   21, 34, 37, 47, 50, 56, 59, 61,
   35, 36, 48, 49, 57, 58, 62, 63,
]

# Flatten 8x8 block in zigzag order
def zigzag_flatten(block):
    flat = block.flatten()
    return [flat[idx] for idx in zigzag_order]                                       

# Huffman tree node and generate codes
class HuffmanNode:
    def __init__(self, symbol, freq):
        self.symbol = symbol
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq

# Build the Huffman tree
def build_huffman_tree(frequencies):
    heap = [HuffmanNode(symbol, freq) for symbol, freq in frequencies.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        node1 = heapq.heappop(heap)
        node2 = heapq.heappop(heap)
        merged = HuffmanNode(None, node1.freq + node2.freq)
        merged.left = node1
        merged.right = node2
        heapq.heappush(heap, merged)
    return heap[0]

# Generate Huffman codes from the tree
def generate_huffman_codes(node, code="", codebook={}):
    if node is not None:
        if node.symbol is not None:
            codebook[node.symbol] = code
        generate_huffman_codes(node.left, code + "0", codebook)
        generate_huffman_codes(node.right, code + "1", codebook)
    return codebook

# Encode quantized coefficients
def jpeg_huffman_encode(quantized_matrix):
    height, width = quantized_matrix.shape
    dc_differences = []
    ac_symbols = []
    ac_all_values = []

    previous_dc = 0

    # Process each 8x8 block
    for i in range(0, height, 8):
        for j in range(0, width, 8):
            block = quantized_matrix[i:i+8, j:j+8]
            flat = zigzag_flatten(block)

            # DC coefficient
            dc = flat[0]
            dc_diff = dc - previous_dc
            dc_differences.append(dc_diff)
            previous_dc = dc

            # AC coefficients
            ac_all_values.extend(flat[1:])

    # Create frequency tables
    dc_frequencies = Counter(dc_differences)
    ac_frequencies = Counter(ac_all_values)

    # Build Huffman trees
    dc_tree = build_huffman_tree(dc_frequencies)
    ac_tree = build_huffman_tree(ac_frequencies)

    # Generate Huffman codes (dc_codes and ac_codes are codebook dictionaries)
    dc_codes = generate_huffman_codes(dc_tree)
    ac_codes = generate_huffman_codes(ac_tree)

    # print(ac_codes)

    # Make data tuples:
    # Process each 8x8 block to tuples
    for i in range(0, height, 8):
        for j in range(0, width, 8):
            ac = flat[1:]
            run_length = 0
            for coef in ac:
                if coef == 0:
                    run_length += 1
                    if run_length == 16:  # Maximum run-length encoding
                        ac_symbols.append((15, 0, 0))                        
                        run_length = 0
                else:
                    size = len(ac_codes[coef])
                    ac_symbols.append((run_length, size, ac_codes[coef]))
                    run_length = 0
            if run_length > 0:  # Add end-of-block symbol
                ### NOTE: Eliminate (15, 0, 0) redundancy.
                ac_symbols.append((0, 0))                                   

    # print(ac_symbols)

    # Encode data
    encoded_dc = "".join(dc_codes[diff] for diff in dc_differences)
    ### NEED to Find a way to encode AC data
    for symbol in ac_symbols:
        if symbol == (0, 0):
            continue
        else:
            run_length, size, bit_string = symbol

    return dc_codes, ac_codes, encoded_dc, encoded_ac


##############  HUFFMAN DECODING ##############
# Reverse zigzag to reconstruct 8x8 blocks
def reverse_zigzag(flattened_block):
    block = np.zeros(64, dtype=int)
    for i, idx in enumerate(zigzag_order):
        block[idx] = flattened_block[i]
    return block.reshape(8, 8)

# Build a Huffman decoding map from the codes
def build_huffman_decoding_map(codes):
    decoding_map = {}
    for symbol, code in codes.items():
        decoding_map[code] = symbol
    return decoding_map

# Decode Huffman-encoded data using the decoding map
def decode_huffman_data(encoded_data, decoding_map):
    current_code = ""
    decoded_symbols = []
    for bit in encoded_data:
        current_code += bit
        if current_code in decoding_map:
            decoded_symbols.append(decoding_map[current_code])
            current_code = ""
    return decoded_symbols

# Decode DC coefficients
def decode_dc_coefficients(encoded_dc, dc_decoding_map):
    dc_differences = decode_huffman_data(encoded_dc, dc_decoding_map)
    dc_coefficients = []
    previous_dc = 0
    for diff in dc_differences:
        current_dc = previous_dc + diff
        dc_coefficients.append(current_dc)
        previous_dc = current_dc
    return dc_coefficients

# Decode AC coefficients
def decode_ac_coefficients(encoded_ac, ac_decoding_map):
    ac_symbols = decode_huffman_data(encoded_ac, ac_decoding_map)
    ac_coefficients = []
    for symbol in ac_symbols:
        print(symbol)
        if (len(symbol) == 2):  # End-of-block (EOB)
            while len(ac_coefficients) % 64 != 0:
                ac_coefficients.append(0)
        else:
            run_length, size, value = symbol
            ac_coefficients.extend([0] * run_length)
            ac_coefficients.append(value)
    return ac_coefficients

# Reconstruct the quantized coefficients matrix
def reconstruct_quantized_matrix(dc_coefficients, ac_coefficients, height, width):
    matrix = np.zeros((height, width), dtype=int)
    block_index = 0
    ac_index = 0

    for i in range(0, height, 8):
        for j in range(0, width, 8):
            # Initialize block with zeros
            block = np.zeros(64, dtype=int)

            # Set DC coefficient
            block[0] = dc_coefficients[block_index]

            # Set AC coefficients
            for k in range(1, 64):
                block[k] = ac_coefficients[ac_index]
                ac_index += 1

            # Reverse zigzag to reconstruct the block
            block_2d = reverse_zigzag(block)
            matrix[i:i+8, j:j+8] = block_2d

            block_index += 1

    return matrix

# Main function to decode the Huffman-encoded data
def jpeg_huffman_decode(dc_codes, ac_codes, encoded_dc, encoded_ac, height, width):
    # Build Huffman decoding maps
    dc_decoding_map = build_huffman_decoding_map(dc_codes)
    ac_decoding_map = build_huffman_decoding_map(ac_codes)

    # Decode DC coefficients
    dc_coefficients = decode_dc_coefficients(encoded_dc, dc_decoding_map)

    # Decode AC coefficients
    ac_coefficients = decode_ac_coefficients(encoded_ac, ac_decoding_map)

    # Reconstruct quantized coefficients matrix
    quantized_matrix = reconstruct_quantized_matrix(dc_coefficients, ac_coefficients, height, width)

    return quantized_matrix


# image_add = input('Please specify the image to compress from images directory : ')
image_add = 'house.png'
image_add = 'images\\' + image_add

image = cv2.imread(image_add)
grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
grayscale_image_padded = np.zeros(((np.shape(grayscale_image)[0]//8 + 1) * 8, (np.shape(grayscale_image)[1]//8 + 1) * 8))

# Copy image and apply padding:
for i in range(np.shape(grayscale_image_padded)[0]):
    for j in range(np.shape(grayscale_image_padded)[1]):
        if (np.shape(grayscale_image)[0] > i) and (np.shape(grayscale_image)[1] > j):
            grayscale_image_padded[i, j] = grayscale_image[i, j]
        else:
            grayscale_image_padded[i, j] = 255

# The dimensions of the dct_ouput are that of the original image itself... now we must quantize this dct_output matrix.
# Q_value = float(input('Specific the Q value you wish to compress the data to : '))
Q_value = 50
Q_matrix = (50/Q_value)*Q_50_matrix

# Divide the padded image into 8x8 blocks and apply DCT + Quantization here:
quantized_coef_matrix = np.zeros(np.shape(grayscale_image_padded))

for i in range(0, np.shape(grayscale_image_padded)[0], 8):
    for j in range(0, np.shape(grayscale_image_padded)[1], 8):
        block = grayscale_image_padded[i:i+8, j:j+8]
        dct_coef = cv2.dct(block)
        quantized_coef_matrix[i:i+8, j:j+8] = np.around(dct_coef/Q_matrix)

print(quantized_coef_matrix)

# Huffman encoding
dc_codes, ac_codes, encoded_dc, encoded_ac = jpeg_huffman_encode(quantized_coef_matrix)

print("DC Huffman Codes:", dc_codes)
print("AC Huffman Codes:", ac_codes)
print("Encoded DC Data:", encoded_dc[:100], "...")
print("Encoded AC Data:", encoded_ac[:100], "...")

# Huffman Decoding
reconstructed_matrix = jpeg_huffman_decode(dc_codes, ac_codes, encoded_dc, encoded_ac, np.shape(quantized_coef_matrix)[0], np.shape(quantized_coef_matrix)[1])
print("Reconstructed Quantized Coefficients Matrix:")
print(reconstructed_matrix)

# Image Reconstruction with inverse 
reconstructed_image = np.zeros(np.shape(grayscale_image_padded))

for i in range(0, np.shape(reconstructed_matrix)[0], 8):
    for j in range(0, np.shape(reconstructed_matrix)[1], 8):
        block = reconstructed_matrix[i:i+8, j:j+8]
        iquan_coef = np.zeros((8,8))
        for m in range(8):
            for n in range(8):
                iquan_coef[m, n] = block[m, n]*Q_matrix[m, n]
        reconstructed_image[i:i+8, j:j+8] = cv2.idct(iquan_coef)

# Observe if images are similar:
plt.imshow("image", grayscale_image_padded)
plt.imshow("reconstruct", reconstructed_image)