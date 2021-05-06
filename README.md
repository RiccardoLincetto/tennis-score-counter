# tennis-score-counter
Score-board content extractor from Tennis TV broadcast.

## Time logs
In this section is reported the time spent on the task.
- 06/05/21 19:25 - 19:50: first considerations and search on existing resources

# Considerations
- For a single match, the score-board is expected to be in a fixed position, possibly being shown intermittently, and not changing its aspect. This allows further refinements after the first detection.
- Scoreboards are overlays to the image and thus might be segmented from the background.
- Time variability is almost absent, except for translucient overlays.
- Color is usally constant (at least within patches), so histograms are peaked (one or more, but with low dispersion)

# Ideas
- Find rectangles which have alphanumeric characters within.
- Use LeNet trained on mnist with squares of different sizes to adapt character to expected network input.
- Subtraction of subsequent frames should give a result close to zero for the scoreboard.
- Check OCR libraries (pytesseract, ...).

# Resources
- https://github.com/gdmurray/ml-scoreboard-extraction
