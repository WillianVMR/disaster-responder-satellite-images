# Image and output sizes
image_height = 224
image_width = 224
image_channels = 3
output_channels = 5
bytes_per_float = 4

# Memory usage per image
image_size = image_height * image_width * image_channels * bytes_per_float
output_size = image_height * image_width * output_channels * bytes_per_float

# Estimate total memory usage per image (in bytes)
memory_per_image = image_size + output_size

# Convert to MB
memory_per_image_MB = memory_per_image / (1024**2)
print(f"Memory per image: {memory_per_image_MB:.2f} MB")

# Total GPU memory (MB) and reserved fraction
total_gpu_memory_MB = 5330  # 5.33 GB available
fraction_reserved_for_batch = 0.8  # Reserve 90% of GPU memory for batch

# Available memory for batch
available_memory_MB = total_gpu_memory_MB * fraction_reserved_for_batch

# Calculate maximum batch size
max_batch_size = available_memory_MB // memory_per_image_MB
print(f"Maximum batch size: {int(max_batch_size)}")
