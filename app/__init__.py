
import os

# Model dependencies belong to installation, never to an API request. Set this before
# lazy Ultralytics imports; operators can explicitly override it in the process environment.
os.environ.setdefault("YOLO_AUTOINSTALL", "false")
