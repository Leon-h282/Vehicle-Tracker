# [PYTHON] VEHICLE TRACKER

## Description
Vehicle Tracker is a machine learning project using `YOLOv8n` model of **Ultralytics** to **detect and track** moving vehicles in the given video and calculate their **speed**.

<img width="426" height="240" alt="VehicleTracker-TestVideo" src="https://github.com/user-attachments/assets/1925581a-1fc4-4236-914c-a39acc42c087" />


## Pipeline
- Extract video's frames.
- Create an `array` of 4 vertex of the tracking area on screen:
```python
def get_src_points(frame: np.ndarray) -> np.float32:
    h, w = frame.shape[:2]
    
    return np.float32([
        [  450,   300],  # Top-left
        [w-450,   300],  # Top-right
        [    w, h-200],  # Bottom-right
        [    0, h-200]   # Bottom-left
    ])
```
- Identify **width** and **length** of the tracking area in real life *(the real width and length in this project were just **estimated**, there are still deviations)*.
- Create flattened real size array of the tracking area vertexes:
```python
dst_point = np.float32([
    [0, 0],                     # Top-left
    [REAL_WIDTH, 0],            # Top-right
    [REAL_WIDTH, REAL_HEIGHT],  # Bottom-right
    [0, REAL_HEIGHT]            # Bottom-left
])
```
- **Detect** and **track** vehicles on screen.
- Check position *(inside or outside tracking area)*.
- Convert units: **pixels** to **meters**.
- Create **label**: `Class` *(vehicle type)*, `Track ID`, `speed`.
- Draw **bounding box** *(only if inside tracking area)*.
---

## Project structure
```text
Project/
│
├── main.py             <- source code
├── yolov8n.pt          <- detection and tracking model
├── requirements.txt    <- environment requirements
└── video.mp4           <- sample video
```

## Requirements

- `python >= 3.10`

- Packages:
    ```text
    opencv-python
    ultralytics
    numpy
    ```

### Install packages
```bash
pip install -r requirements.txt
```

---

## The original video source
https://www.youtube.com/watch?v=wqctLW0Hb_0&list=WL&index=1&t=441s
