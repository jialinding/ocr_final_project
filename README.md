# ocr_final_project

Generate data (text -> images) with the following command.
```
python3 gen-images.py <src text file> <output-dir>
```
`<output-dir>` will be populated with a bunch of jpegs and text files (the labels).
Here's naming convention: `xxx.<quality>.png` and `xxx.txt`.

Generate character images (text image -> individual character images) with the following command.
```
python3 segment.py <src text image file> <output-dir>
```
`<output-dir>` will contain an image for each text line of the input image with the format `line<X>.<quality>.jpg`
	and an image for each character of the input image with the format `line<X>.char<Y>.<quality>.jpg`.
