import cv2
import numpy as np
import pandas as pd
import math
import argparse
import os

from plotting import plot_angle, fit_amplitude, binning, amplitude_decay

def preprocess_frame(frame, prev_gray, backSub, apply_filter=True, pivot=None, radius=None, arc_tolerance=150):
    """
    Frame preprocessing, optimised for speed using downscaling and reduced kernel sizes.
    """
    if not apply_filter:
        return frame
        
    # downscales frame for heavy calculations
    scale = 0.5 
    small_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
    small_gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
    
    # prev_gray is passed at full resolution, scale it to match
    small_prev_gray = cv2.resize(prev_gray, (0, 0), fx=scale, fy=scale)
    
    # dense optical flow on smaller image
    flow = cv2.calcOpticalFlowFarneback(small_prev_gray, small_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    speed_map = np.clip(mag * 10, 0, 255).astype(np.uint8)
    
    # background subtraction (IMPORTANT)
    blurred = cv2.GaussianBlur(small_gray, (5, 5), 0) # Reduced kernel
    fg_mask = backSub.apply(blurred, learningRate=0.005)
    
    # combine
    motion_blob = cv2.bitwise_and(speed_map, speed_map, mask=fg_mask)
    
    # morphology (reduced kernels for speed on the downscaled frame)
    blob_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)) 
    dilated = cv2.dilate(motion_blob, blob_kernel, iterations=1) 
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, blob_kernel, iterations=1)
    
    # upscale back to original resolution for masking and tracking
    full_closed = cv2.resize(closed, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # masking and floor cutoff
    if pivot is not None and radius is not None:
        mask = np.zeros_like(prev_gray) 
        cv2.circle(mask, pivot, int(radius), 255, thickness=arc_tolerance)
        lowest_point = int(pivot[1] + radius + (arc_tolerance / 2.0))
        cv2.rectangle(mask, (0, lowest_point), (mask.shape[1], mask.shape[0]), 0, -1)
        full_closed = cv2.bitwise_and(full_closed, full_closed, mask=mask)
    
    bgr_closed = cv2.cvtColor(full_closed, cv2.COLOR_GRAY2BGR)
    
    # final blur for easier tracking
    final_blurred = cv2.GaussianBlur(bgr_closed, (9, 9), 0) # Reduced kernel
    
    return final_blurred

def process_video(tracker, video_path, csv_path, ret=False, use_preprocessing=True):
    """
    Process a video to track an object and log its position and angle over time.
    """
    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        print("動画は開けられませんでした。")
        return "QUIT"

    ok, frame = video.read()
    if not ok:
        print('動画ファイルは開けられません。')
        return "QUIT"

    frame_height, frame_width = frame.shape[:2]
    
    # initialises the previous grayscale frame for Optical Flow
    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)
    
    # PIVOT POINT SELECT
    pivot = None
    def select_pivot(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            nonlocal pivot
            print(f"枢軸設定: X = {x}, Y = {y}")
            pivot = (x, y)

    cv2.namedWindow("PIVOT POINT SELECT", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("PIVOT POINT SELECT", min(frame_width, 800), min(frame_height, 600))
    cv2.setMouseCallback("PIVOT POINT SELECT", select_pivot)
    
    pivot_display = frame.copy()
    cv2.putText(pivot_display, "CLICK PIVOT POINT", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    while pivot is None:
        cv2.imshow("PIVOT POINT SELECT", pivot_display)
        k = cv2.waitKey(1) & 0xFF
        if k in [ord('q'), ord('Q'), 27]:
            print("枢軸選択にてトラッキングがキャンセルされた。")
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            video.release()
            return "QUIT"

    cv2.destroyWindow("PIVOT POINT SELECT")
    cv2.waitKey(1)

    # EQUILIBRIUM POINT SELECT
    origin = None
    def select_origin(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            nonlocal origin
            print(f"平衡点{origin}設定: X = {x}, Y = {y}")
            origin = (x, y)

    cv2.namedWindow("EQUILIBRIUM POINT SELECT", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("EQUILIBRIUM POINT SELECT", min(frame_width, 800), min(frame_height, 600))
    cv2.setMouseCallback("EQUILIBRIUM POINT SELECT", select_origin)

    origin_display = frame.copy()
    cv2.circle(origin_display, pivot, 5, (0, 255, 0), -1) 
    cv2.putText(origin_display, "CLICK EQUILIBRIUM POINT", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    while origin is None:
        cv2.imshow("EQUILIBRIUM POINT SELECT", origin_display)
        k = cv2.waitKey(1) & 0xFF
        if k in [ord('q'), ord('Q'), 27]:
            print("平衡点選択にてトラッキングがキャンセルされた。")
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            video.release()
            return "QUIT"

    cv2.destroyWindow("EQUILIBRIUM POINT SELECT")
    cv2.waitKey(1)

    radius = math.hypot(origin[0] - pivot[0], origin[1] - pivot[1])

    # OBJECT SELECT ROI
    cv2.namedWindow("OBJECT SELECT", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("OBJECT SELECT", min(frame_width, 800), min(frame_height, 600))
    
    bounding_box = cv2.selectROI("OBJECT SELECT", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()
    cv2.waitKey(1)

    if bounding_box == (0, 0, 0, 0):
        print("対象物体選択にてトラッキングがキャンセルされた。")
        video.release()
        return "QUIT"

    global trackers
    cvtracker = trackers[tracker]()
    
    if use_preprocessing:
        cv2.namedWindow("VIEWPORT (PREPROCESSED)", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("VIEWPORT (PREPROCESSED)", min(frame_width, 400), min(frame_height, 300))
        cv2.createTrackbar("ARC TOLERANCE", "VIEWPORT (PREPROCESSED)", 150, 500, lambda x: None)
        initial_tolerance = cv2.getTrackbarPos("ARC TOLERANCE", "VIEWPORT (PREPROCESSED)")
    else:
        initial_tolerance = 150

    cv2.namedWindow("TRACKING", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("TRACKING", min(frame_width, 800), min(frame_height, 600))
    cv2.createTrackbar("TRUE VERTICAL (0=OFF, 1=ON)", "TRACKING", 1, 1, lambda x: None)
    
    processed_init_frame = preprocess_frame(frame, prev_gray, backSub, apply_filter=use_preprocessing, pivot=pivot, radius=radius, arc_tolerance=initial_tolerance)
    ok = cvtracker.init(processed_init_frame, bounding_box)

    position_log = []
    recorded_fps = video.get(cv2.CAP_PROP_FPS)
    adj_x, adj_y, angle = 0, 0, 0
    current_frame = 0

    while True:
        ok, frame = video.read()
        if not ok:
            break

        if use_preprocessing:
            arc_tolerance = cv2.getTrackbarPos("ARC TOLERANCE", "VIEWPORT (PREPROCESSED)")
        else:
            arc_tolerance = 150

        use_true_vertical = cv2.getTrackbarPos("TRUE VERTICAL (0=OFF, 1=ON)", "TRACKING")

        processed_frame = preprocess_frame(frame, prev_gray, backSub, apply_filter=use_preprocessing, pivot=pivot, radius=radius, arc_tolerance=arc_tolerance)
        ok, bbox = cvtracker.update(processed_frame)
        
        current_time = (current_frame / recorded_fps)

# anti drift guard
        if ok and pivot is not None and radius is not None:
            x, y, w, h = map(int, bbox)
            meas_cx = x + w / 2.0
            meas_cy = y + h / 2.0
            
            dx = meas_cx - pivot[0]
            dy = meas_cy - pivot[1]
            dist_to_pivot = math.hypot(dx, dy)
            
            max_deviation = arc_tolerance / 2.0
            
            if abs(dist_to_pivot - radius) > max_deviation:
                cv2.putText(frame, "RADIAL BOUNDARY ENFORCED", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255), 2)
                
                if dist_to_pivot > 0:
                    meas_cx = pivot[0] + (dx / dist_to_pivot) * radius
                    meas_cy = pivot[1] + (dy / dist_to_pivot) * radius
                
                x = int(meas_cx - w / 2.0)
                y = int(meas_cy - h / 2.0)
                bbox = (x, y, w, h)
                
                cvtracker = trackers[tracker]()
                cvtracker.init(processed_frame, bbox)

        if ok:
            x, y, w, h = map(int, bbox)
            p1 = (x, y)
            p2 = (x + w, y + h)
            
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
            cv2.circle(frame, origin, 5, (0, 255, 0), -1) 
            
            if use_preprocessing:
                cv2.rectangle(processed_frame, p1, p2, (0, 0, 255), 2, 1)
                cv2.circle(processed_frame, pivot, int(radius), (50, 50, 50), 1)
                
                # visualises the floor cutoff line
                lowest_point = int(pivot[1] + radius + (arc_tolerance / 2.0))
                cv2.line(processed_frame, (0, lowest_point), (frame_width, lowest_point), (0, 255, 255), 1)

            center_x = x + w // 2
            center_y = y + h // 2

            adj_x = center_x - origin[0]
            adj_y = center_y - origin[1]

            if use_true_vertical == 1 and pivot is not None:
                angle_dx = center_x - pivot[0]
                angle_dy = center_y - pivot[1]
            else:
                angle_dx = adj_x
                angle_dy = adj_y

            angle = math.degrees(math.atan2(angle_dx, angle_dy))

            if current_frame % 5 == 0:
                position_log.append([round(current_time, 3), adj_x, adj_y, round(angle, 3)])

            cv2.putText(frame, f"Time: {current_time:.2f}, X: {adj_x:.2f}, Y:{adj_y:.2f}", (100, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (50, 170, 50), 2)
            cv2.putText(frame, f"Angle:{angle:.2f}", (100, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (50, 170, 50), 2)
        else:
            cv2.putText(frame, "TRACKING ERROR", (100, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)

        cv2.imshow("TRACKING", frame)
        
        if use_preprocessing:
            cv2.imshow("VIEWPORT (PREPROCESSED)", processed_frame)

        k = cv2.waitKey(1) & 0xff
        if k == 27: 
            print("次の動画への変更中")
            break
        elif k in [ord('q'), ord('Q')]: 
            print("強制脱出。全トラッキング停止中...")
            video.release()
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            return "QUIT"

        current_frame += 1
        
        # updates previous frame for the next loop's optical flow calculation
        prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    video.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)

    if not position_log:
        if ret:
            return None, 0, 0
        return

    df = pd.DataFrame(position_log, columns=["時刻(s)", "X軸", "Y軸", "角度(deg)"])
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"位置データは{csv_path}に保存された。")

    png_path = os.path.splitext(csv_path)[0] + '.png'
    
    if ret:
        data, q_factor = fit_amplitude(df, output_path=png_path, ret=True)
    else:
        fit_amplitude(df, output_path=png_path) 
    print(f"振幅グラフは{png_path}に保存された。")

    if ret:
        return data, df["時刻(s)"].iloc[-1], q_factor

def process_directory(tracker, directory, ret=False, use_preprocessing=True):
    """
    Processes all video files in a given directory using a specified tracker.
    """
    video_files = [f for f in os.listdir(directory) if f.endswith(('.mov', '.MOV', '.mp4', '.avi', '.mkv'))]

    if ret:
        trials = []
        length = []
        q_factors = []

    if not video_files:
        print("指定された名鑑に動画はありません。")
        return

    os.makedirs("./output", exist_ok=True)

    for video_file in video_files:
        video_path = os.path.join(directory, video_file)
        output_csv_path = os.path.join("./output", f"{os.path.splitext(video_file)[0]}_output.csv")

        if ret:
            result = process_video(tracker, video_path, output_csv_path, ret, use_preprocessing)
            if result == "QUIT":
                print("一括処理がユーザーによって中止された。")
                break
            if result and result[0] is not None:
                data, time, q_factor = result
                trials.append(data)
                length.append(time)
                q_factors.append(q_factor)
        else:
            result = process_video(tracker, video_path, output_csv_path, use_preprocessing=use_preprocessing)
            if result == "QUIT":
                print("一括処理がユーザーによって中止された。")
                break

    if ret and trials:
        times, means, uncertainties = binning(trials, max_time=max(length))
        average_q = sum(q_factors) / len(q_factors) if len(q_factors) > 0 else 0
        amplitude_decay(times, means, uncertainties, average_q, output_path=f"./output/decay_fit_{directory}.png")
    elif ret:
        print("トラッキングデータは集まっていません。最終一括グラフをスキップ中。")

    print(f"一括処理は完了している: {directory}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Specify tracker and source directory.")

    parser.add_argument(
        '--tracker',
        type=str,
        help='Please use capitalised name, see README.md',
        default="CSRT"
    )

    parser.add_argument(
        '--source',
        type=str,
        default="videos",
        help='Source directory path as string.'
    )

    parser.add_argument(
        '--multi-trials', '-m',
        type=str,
        default="False",
        help='Run multiple trials to get mean and error'
    )
    
    parser.add_argument(
        '--no-preprocess', 
        action='store_true',
        help='Disable frame preprocessing (CLAHE and blurring)'
    )

    args = parser.parse_args()

    if not os.path.isdir(args.source):
        raise NotADirectoryError(f"'{args.source}'は有効名鑑ではありません。")

    global trackers
    trackers = {
        "CSRT": cv2.legacy.TrackerCSRT_create,
        "KCF": cv2.legacy.TrackerKCF_create,
        "BOOSTING": cv2.legacy.TrackerBoosting_create,
        "MIL": cv2.legacy.TrackerMIL_create,
        "TLD": cv2.legacy.TrackerTLD_create,
        "MEDIANFLOW": cv2.legacy.TrackerMedianFlow_create,
        "MOSSE": cv2.legacy.TrackerMOSSE_create
    }

    ret = True if args.multi_trials == "True" else False
    use_preprocessing = not args.no_preprocess

    print(f"Using tracker: {args.tracker}")
    print(f"Operating on source directory: {args.source}")
    print(f"Preprocessing enabled: {use_preprocessing}")

    process_directory(args.tracker, args.source, ret, use_preprocessing)
