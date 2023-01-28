# SDTrainPreProc
 
Image batch pre-process for Stable Diffusion traning.

- Categorize images by tag
- Upscale or downscale images to a size suitable for training. ( Keep high quality as possible. )

# Setup
Need python.

Click _install.bat 

# Move files has poor tags
1. Open "1_classify_tags.bat" File.
2. Write Images Directory to first line. - like as "set DIR_IN=C:\Pictures\"
3. Save
4. Run

# Resize Images
1. Open "2_adjust_size.bat" File.
2. Write Images Directory to first line - like as "set DIR_IN=C:\Pictures\"
3. Write resolution you want to train. - like as "set RES=512" or "set RES=768" or "set RES=1024"
4. Save
5. Run
