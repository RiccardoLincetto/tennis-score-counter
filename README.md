# tennis-score-counter
Score-board content extractor from Tennis TV broadcast.

## Tasks
- [x] scoreboard localization
- [ ] text extraction

# Considerations
- For a single match, the score-board is expected to be in a fixed position, possibly being shown intermittently, and not changing its aspect. This allows further refinements after the first detection.
- Scoreboards are overlays to the image and thus might be segmented from the background.
- Time variability is almost absent, except for translucient overlays. In the provided examples though the camera is fixed, so this is true for most of the content, making this consideration unusable.
- Color is usally constant (at least within patches), so histograms are peaked (one or more, but with low dispersion)
- Tesseract has different performances when run on entire image or on groundtruth scoreboard. This means preprocessing is important. The images might be reshaped, which means the aspect
ratio of the image could vary. This is important to be checked for a correct OCR.
- As pointed out by [tesseract's guide on improving OCR quality](https://tesseract-ocr.github.io/tessdoc/ImproveQuality.html), version 4.x here used requires dark backround and light text for recognition.

- The scores for the current set have a different background, resulting in a different rectangle. The scoreboard than is made by one rectangle containing the other, with the same height. Sometimes though there is just a cluster of contiguous rectangles which make up the scoreboard parts.

# Ideas
- Find rectangles which have alphanumeric characters within.
- Use LeNet trained on mnist with squares of different sizes to adapt character to expected network input.
- Subtraction of subsequent frames should give a result close to zero for the scoreboard.
- Check OCR libraries (pytesseract, ...).
- Exclude rectangles within the detected court.

# Resources
- https://github.com/gdmurray/ml-scoreboard-extraction
- https://github.com/evansloan/sports.py
- https://github.com/sethah/deeptennis

# Repo structure
- `data/` contains the downloaded zip, with an annotated video.
- `src/` contains the source code, mainly python scripts.