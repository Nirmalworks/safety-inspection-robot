# Contributing

## Branches

| Branch | Purpose |
|---|---|
| `main` | Reviewed, working code. Only updated through pull requests from `dev`. |
| `dev` | Integration branch. All feature work is merged and tested together here. |
| `feature/*`, `fix/*`, `nav/*`, `sim/*` | One topic per branch, branched from `dev` and merged back into `dev`. |

```bash
git switch dev && git pull
git switch -c feature/<short-topic>
# ... work, commit ...
git push -u origin feature/<short-topic>   # then open a pull request into dev
```

## Commit messages

Start with the area, then say what changed, in the present tense:

```
Nav2: lower the DWB max velocity for the narrow aisle
Dashboard: show the inspection result image next to the map
ml: depth rule ignores points below the floor plane
```

Keep one logical change per commit. Do not commit build output, bags, model weights or raw screen recordings
(`.gitignore` covers them).

## Before opening a pull request

```bash
ruff check .                              # lint
python -m pytest ml/                      # offline ML tests (pip install -r ml/requirements.txt)
colcon build --symlink-install            # ROS 2 packages build
```

CI runs the same three checks on every pull request. If the change touches navigation, localization or the
patrol, say in the pull request whether it was tested in simulation ([docs/SIMULATION.md](docs/SIMULATION.md))
or on the robot ([docs/ROBOT_SESSION_CHECKLIST.md](docs/ROBOT_SESSION_CHECKLIST.md)).

## Code style

* Python: [ruff](https://docs.astral.sh/ruff/) with the settings in `pyproject.toml`; 4-space indents.
* YAML / XML / launch files: 2-space indents (see `.editorconfig`).
* New ROS parameters go in the YAML under `src/ame505_bringup/config/`, not hard-coded in nodes.
* New docs go in `docs/` and get a row in the README documentation table.
