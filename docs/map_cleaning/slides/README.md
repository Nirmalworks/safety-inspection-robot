# Navigation Map Segment: slide sources

Source of the deck for the map-cleaning part of the final presentation (27 Nov 2026), 16 slides: problem, model,
results in nine steps (`results`, `results2` … `results9`: one slide per method, so that stepping through works in
PowerPoint and Google Slides), on the robot, scoreboard for the lab, scoreboard for the unseen building, limits, demo recording.

- `project/deck.json`: slide order. `project/deck.css`: shared styles. `project/slides/*.html`: one file per slide,
  with the speaker notes in `<aside>`.
- `img/`: the pictures the slides use (raw map, each method's cleaned map of test map `lab2d_11_25_7`).
- `video/map_demo.mp4`: 57 s screen recording of the map editor shown on the demo slide.

The slides reference uploaded copies of these files by id (`/_blob/...`), so the HTML does not render on its own.
Results behind the numbers: `../RESULTS.md`.
