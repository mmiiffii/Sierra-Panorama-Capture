name: Feratel Camera Watcher

on:
  workflow_dispatch: {}
  schedule:
    - cron: "0 */6 * * *"   # safety restart every 6 hours

permissions:
  contents: write

concurrency:
  group: feratel-watcher
  cancel-in-progress: true

jobs:
  watch:
    runs-on: ubuntu-latest
    timeout-minutes: 360
    env:
      CHECK_INTERVAL: "60"     # seconds between passes
      REQUIRE_OCR: "0"         # set to "1" if you want to skip saves when OCR fails
      OCR_LANG: "eng"          # install extra langs as needed (e.g., 'eng+spa')
    steps:
      - name: Checkout
        uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install system deps (Tesseract for OCR)
        run: |
          sudo apt-get update
          sudo apt-get install -y tesseract-ocr

      - name: Install Python deps
        run: |
          python -m pip install --upgrade pip
          pip install requests numpy opencv-python-headless pytesseract

      - name: Configure git
        run: |
          git config user.name  "github-actions[bot]"
          git config user.email "41898282+github-actions[bot]@users.noreply.github.com"

      - name: Loop: grab once, commit if changed
        env:
          REQUIRE_OCR: ${{ env.REQUIRE_OCR }}
          OCR_LANG: ${{ env.OCR_LANG }}
          CHECK_INTERVAL: ${{ env.CHECK_INTERVAL }}
        run: |
          end=$((SECONDS + 340*60))  # ~5h40m
          while [ $SECONDS -lt $end ]; do
            python feratel_cam_watcher.py  # one pass
            if [ -n "$(git status --porcelain images)" ]; then
              git add images
              git commit -m "feratel: new images $(date -u +'%Y-%m-%d %H:%M:%S UTC')"
              git push
            fi
            sleep "${CHECK_INTERVAL}"
          done
