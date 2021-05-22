# tennis-score-counter
Score-board content extractor from Tennis TV broadcast.

## Time logs
In this section is reported the time spent on the task.
- 06/05/21 19:25 - 19:50: first considerations and search on existing resources
- 08/05/21 09:35 - 10:10: tesseract installation tentative
- 09/05/21 16:50 - 17:30: tesseract installation troubleshooting
- 12/05/21 22:00 - 23:30: opencv rectangle extraction
- 13/05/21 21:30 - 22:00: switch to containerised environment
- 18/05/21 08:00 - 08:45: repo reorganization
- 21/05/21 08:00 - 08:30: pipeline validation
- 21/05/21 22:30 - 23:00: pipeline validation
- 22/05/21 10:00 - 10:30: debug bad distance-based rectangle-rating 

# Considerations
- For a single match, the score-board is expected to be in a fixed position, possibly being shown intermittently, and not changing its aspect. This allows further refinements after the first detection.
- Scoreboards are overlays to the image and thus might be segmented from the background.
- Time variability is almost absent, except for translucient overlays. In the provided examples though the camera is fixed, so this is true for most of the content, making this consideration unusable.
- Color is usally constant (at least within patches), so histograms are peaked (one or more, but with low dispersion)
- Tesseract has different performances when run on entire image or on groundtruth scoreboard. This means preprocessing is important. The images might be reshaped, which means the aspect
ratio of the image could vary. This is important to be checked for a correct OCR.

# Ideas
- Find rectangles which have alphanumeric characters within.
- Use LeNet trained on mnist with squares of different sizes to adapt character to expected network input.
- Subtraction of subsequent frames should give a result close to zero for the scoreboard.
- Check OCR libraries (pytesseract, ...).

# Resources
- https://github.com/gdmurray/ml-scoreboard-extraction

# Repo structure
- `data/` contains the downloaded zip, with an annotated video.
- `src/` contains the source code, mainly python scripts.