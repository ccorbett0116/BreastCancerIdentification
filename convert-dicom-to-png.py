# Install dependencies
# !pip install pydicom opencv-python
# uncomment for google colab

import os
import cv2
import pydicom
import numpy as np  # Add this!
from google.colab import drive
import gc

# Mount Google Drive
drive.mount('/content/drive')
base_path = '/content/drive/MyDrive/Cropped Dataset/'
output_drive_base = '/content/drive/MyDrive/Cropped Dataset/Converted PNGs/'

# Define subdirectories
train_subdirs = ['Calc-Training-Final', 'Mass-Training-Final']
test_subdirs = ['Calc-Test-Final', 'Mass-Test-Final']
all_subdirs = train_subdirs + test_subdirs

# Conversion batch size
conversion_batch_size = 500

# Convert DICOM to PNG
def dicom_to_png(dicom_path, output_dir):
    ds = pydicom.dcmread(dicom_path)
    img = ds.pixel_array
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    output_path = os.path.join(output_dir, os.path.basename(dicom_path).replace('.dcm', '.png'))
    if not os.path.exists(output_path):  # Skip if already converted
        cv2.imwrite(output_path, img)
    return output_path

# Collect all DICOM paths
dicom_paths = []
for subdir in all_subdirs:
    full_path = os.path.join(base_path, subdir)
    dicom_files = [os.path.join(full_path, f) for f in os.listdir(full_path) if f.endswith('.dcm')]
    dicom_paths.extend(dicom_files)
print(f"Found {len(dicom_paths)} total DICOM files")

# Convert and save to Google Drive
os.makedirs(output_drive_base, exist_ok=True)
converted_paths = []
for i in range(0, len(dicom_paths), conversion_batch_size):
    batch_paths = dicom_paths[i:i + conversion_batch_size]
    for j, dicom_path in enumerate(batch_paths):
        png_path = dicom_to_png(dicom_path, output_drive_base)
        converted_paths.append(png_path)
        if (j + 1) % 50 == 0:  # Update every 50 files
            print(f"Converted {len(converted_paths)} of {len(dicom_paths)} files ({(len(converted_paths)/len(dicom_paths)*100):.1f}%)")
    print(f"Batch complete: Converted {len(converted_paths)} of {len(dicom_paths)} files")
    gc.collect()

print("Conversion complete!")
print(f"Saved {len(converted_paths)} PNGs to {output_drive_base}")