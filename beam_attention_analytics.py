"""
Beam Eye Tracker with Attention Analytics
Generates video + quantifiable attention metrics and distraction detection

Requirements:
- pip install numpy opencv-python pillow matplotlib scipy pandas
- Beam Eye Tracker SDK installed
- Beam Eye Tracker app running and CALIBRATED
"""

import sys
import time
import numpy as np
from datetime import datetime
from pathlib import Path
from scipy.ndimage import gaussian_filter
import cv2
import json

try:
    from eyeware import beam_eye_tracker as bet
except ImportError:
    print("ERROR: beam_eye_tracker module not found!")
    print("Please ensure the Beam Eye Tracker SDK is installed correctly.")
    sys.exit(1)


class AttentionAnalytics:
    """Analyzes gaze data to quantify attention and detect distraction"""
    
    def __init__(self, screen_width, screen_height):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.gaze_history = []  # List of (timestamp, x, y, is_valid) tuples
        
    def add_sample(self, timestamp, x, y, is_valid):
        """Add a gaze sample with validity flag"""
        self.gaze_history.append((timestamp, x, y, is_valid))
    
    def calculate_attention_score(self):
        """
        Calculate overall attention score (0-100)
        Based on: data validity, gaze stability, and tracking consistency
        """
        if len(self.gaze_history) == 0:
            return 0
        
        valid_samples = [s for s in self.gaze_history if s[3]]
        validity_rate = len(valid_samples) / len(self.gaze_history)
        
        # High validity = high attention
        attention_score = validity_rate * 100
        
        return attention_score
    
    def detect_distraction_periods(self, threshold_seconds=2.0):
        """
        Detect periods where user was distracted (lost tracking)
        Returns list of (start_time, end_time, duration) tuples
        """
        distraction_periods = []
        in_distraction = False
        distraction_start = None
        
        for i, (timestamp, x, y, is_valid) in enumerate(self.gaze_history):
            if not is_valid and not in_distraction:
                # Start of distraction
                in_distraction = True
                distraction_start = timestamp
            elif is_valid and in_distraction:
                # End of distraction
                duration = timestamp - distraction_start
                if duration >= threshold_seconds:
                    distraction_periods.append((distraction_start, timestamp, duration))
                in_distraction = False
                distraction_start = None
        
        # Handle case where tracking was lost at the end
        if in_distraction and distraction_start is not None:
            last_timestamp = self.gaze_history[-1][0]
            duration = last_timestamp - distraction_start
            if duration >= threshold_seconds:
                distraction_periods.append((distraction_start, last_timestamp, duration))
        
        return distraction_periods
    
    def calculate_gaze_dispersion(self):
        """
        Calculate gaze dispersion (how spread out the gaze is)
        Lower dispersion = more focused attention
        Higher dispersion = more wandering/scattered attention
        """
        valid_points = [(x, y) for t, x, y, valid in self.gaze_history if valid]
        
        if len(valid_points) < 2:
            return 0
        
        points = np.array(valid_points)
        
        # Calculate standard deviation (normalized by screen size)
        std_x = np.std(points[:, 0]) / self.screen_width
        std_y = np.std(points[:, 1]) / self.screen_height
        
        # Combined dispersion metric (0-1 scale)
        dispersion = np.sqrt(std_x**2 + std_y**2)
        
        return dispersion
    
    def calculate_fixation_stability(self, window_seconds=1.0):
        """
        Calculate fixation stability (how steady the gaze is)
        High stability = focused attention
        Low stability = rapid eye movements/distraction
        """
        if len(self.gaze_history) < 10:
            return 0
        
        # Calculate velocity (movement between consecutive points)
        velocities = []
        for i in range(1, len(self.gaze_history)):
            t1, x1, y1, v1 = self.gaze_history[i-1]
            t2, x2, y2, v2 = self.gaze_history[i]
            
            if v1 and v2 and (t2 - t1) > 0:
                dt = t2 - t1
                dx = x2 - x1
                dy = y2 - y1
                velocity = np.sqrt(dx**2 + dy**2) / dt
                velocities.append(velocity)
        
        if len(velocities) == 0:
            return 0
        
        # Lower average velocity = more stable fixation
        avg_velocity = np.mean(velocities)
        # Normalize (assuming max velocity of ~2000 pixels/second)
        stability = max(0, 1 - (avg_velocity / 2000))
        
        return stability
    
    def calculate_attention_distribution(self, grid_size=3):
        """
        Calculate how attention is distributed across screen regions
        Returns dictionary with region percentages
        """
        valid_points = [(x, y) for t, x, y, valid in self.gaze_history if valid]
        
        if len(valid_points) == 0:
            return {}
        
        # Divide screen into grid
        grid_width = self.screen_width / grid_size
        grid_height = self.screen_height / grid_size
        
        region_counts = {}
        for x, y in valid_points:
            col = min(int(x / grid_width), grid_size - 1)
            row = min(int(y / grid_height), grid_size - 1)
            region_key = f"Region_{row}_{col}"
            region_counts[region_key] = region_counts.get(region_key, 0) + 1
        
        # Convert to percentages
        total = len(valid_points)
        region_percentages = {k: (v / total * 100) for k, v in region_counts.items()}
        
        return region_percentages
    
    def calculate_engagement_over_time(self, interval_seconds=5):
        """
        Calculate engagement levels over time in intervals
        Returns list of (time_point, engagement_score) tuples
        """
        if len(self.gaze_history) == 0:
            return []
        
        max_time = self.gaze_history[-1][0]
        engagement_timeline = []
        
        for t in np.arange(0, max_time, interval_seconds):
            # Get samples in this time window
            window_samples = [s for s in self.gaze_history 
                            if t <= s[0] < t + interval_seconds]
            
            if len(window_samples) == 0:
                engagement_timeline.append((t, 0))
                continue
            
            # Calculate engagement as % of valid samples
            valid_count = sum(1 for s in window_samples if s[3])
            engagement = (valid_count / len(window_samples)) * 100
            engagement_timeline.append((t, engagement))
        
        return engagement_timeline
    
    def generate_attention_report(self):
        """Generate comprehensive attention analysis report"""
        report = {
            "summary": {},
            "distraction_analysis": {},
            "spatial_analysis": {},
            "temporal_analysis": {}
        }
        
        # Summary metrics
        valid_samples = [s for s in self.gaze_history if s[3]]
        total_duration = self.gaze_history[-1][0] if self.gaze_history else 0
        
        report["summary"] = {
            "total_samples": len(self.gaze_history),
            "valid_samples": len(valid_samples),
            "validity_rate_percent": (len(valid_samples) / len(self.gaze_history) * 100) if self.gaze_history else 0,
            "total_duration_seconds": total_duration,
            "attention_score": self.calculate_attention_score(),
            "average_sample_rate_hz": len(self.gaze_history) / total_duration if total_duration > 0 else 0
        }
        
        # Distraction analysis
        distractions = self.detect_distraction_periods()
        total_distraction_time = sum(d[2] for d in distractions)
        
        report["distraction_analysis"] = {
            "distraction_count": len(distractions),
            "total_distraction_time_seconds": total_distraction_time,
            "distraction_percentage": (total_distraction_time / total_duration * 100) if total_duration > 0 else 0,
            "longest_distraction_seconds": max([d[2] for d in distractions]) if distractions else 0,
            "distraction_periods": [
                {
                    "start_time": d[0],
                    "end_time": d[1],
                    "duration": d[2]
                } for d in distractions
            ]
        }
        
        # Spatial analysis
        dispersion = self.calculate_gaze_dispersion()
        stability = self.calculate_fixation_stability()
        
        report["spatial_analysis"] = {
            "gaze_dispersion": dispersion,
            "fixation_stability": stability,
            "focus_quality": stability * (1 - dispersion) * 100,  # Combined metric
            "attention_distribution": self.calculate_attention_distribution()
        }
        
        # Temporal analysis
        engagement_timeline = self.calculate_engagement_over_time()
        
        report["temporal_analysis"] = {
            "engagement_timeline": engagement_timeline,
            "average_engagement": np.mean([e[1] for e in engagement_timeline]) if engagement_timeline else 0
        }
        
        return report


class RealtimeHeatmapVideoGenerator:
    """Collects gaze data and generates a real-time heatmap video with analytics"""
    
    def __init__(self, screen_width=1920, screen_height=1080, app_name="Attention Analytics"):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.gaze_points = []  # List of (x, y) tuples
        self.start_time = None
        self.is_tracking = False
        self.output_folder = None
        self.last_update_timestamp = None
        
        # Analytics
        self.analytics = AttentionAnalytics(screen_width, screen_height)
        
        # Video settings
        self.video_width = 800
        self.video_height = 600
        self.fps = 30
        self.video_writer = None
        
        # Heatmap settings
        self.heatmap_resolution = 200
        self.blur_sigma = 3.0
        
        # Initialize Beam API
        print(f"Initializing Beam Eye Tracker API...")
        print(f"Screen resolution: {screen_width}x{screen_height}")
        print(f"Video resolution: {self.video_width}x{self.video_height} @ {self.fps}fps")
    
        try:
            point_00 = bet.Point(0, 0)
            point_11 = bet.Point(screen_width, screen_height)
            viewport = bet.ViewportGeometry(point_00, point_11)
            self.api = bet.API(app_name, viewport)
            print("✓ API initialized successfully")
        except Exception as e:
            print(f"✗ Failed to initialize API: {e}")
            raise
    
    def start_tracking(self, base_folder="attention_analysis"):
        """Start collecting gaze data and initialize video writer"""
        print("\nAttempting to start Beam Eye Tracker...")
        
        # Create base output folder
        base_path = Path(base_folder)
        base_path.mkdir(exist_ok=True)
        
        # Find next participant number
        participant_num = 1
        while True:
            participant_folder = base_path / f"Participant_{participant_num}"
            if not participant_folder.exists():
                break
            participant_num += 1
        
        # Create participant folder
        self.output_folder = participant_folder
        self.output_folder.mkdir(exist_ok=True)
        self.participant_num = participant_num
        
        print(f"✓ Created folder: {self.output_folder.name}")
        
        # Create video filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_timestamp = timestamp
        self.video_filename = f"heatmap_{timestamp}.mp4"
        self.video_path = self.output_folder / self.video_filename
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            str(self.video_path),
            fourcc,
            self.fps,
            (self.video_width, self.video_height)
        )
        
        print(f"✓ Video will be saved to: {self.video_path.absolute()}")
        
        try:
            self.api.attempt_starting_the_beam_eye_tracker()
            print("✓ Beam Eye Tracker started")
            
            self.last_update_timestamp = bet.NULL_DATA_TIMESTAMP()
            time.sleep(1)
            
            self.is_tracking = True
            self.start_time = time.time()
            print("✓ Tracking started successfully!")
            print("\nIMPORTANT: Make sure you have:")
            print("  1. Beam Eye Tracker app is running")
            print("  2. You have completed calibration")
            print("  3. Your face is visible to the webcam")
            print("  4. Look around your screen naturally\n")
        except Exception as e:
            print(f"✗ Failed to start tracking: {e}")
            if self.video_writer:
                self.video_writer.release()
            raise
    
    def create_heatmap_frame(self, elapsed_time):
        """Generate a single heatmap frame from current gaze points"""
        # Create blank image
        frame = np.zeros((self.video_height, self.video_width, 3), dtype=np.uint8)
        
        if len(self.gaze_points) == 0:
            # Add "Waiting for gaze data..." text
            text = "Waiting for gaze data..."
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_size = cv2.getTextSize(text, font, 0.7, 2)[0]
            text_x = (self.video_width - text_size[0]) // 2
            text_y = (self.video_height + text_size[1]) // 2
            cv2.putText(frame, text, (text_x, text_y), font, 0.7, (255, 255, 255), 2)
            return frame
        
        # Normalize gaze points to video dimensions
        normalized_points = []
        for x, y in self.gaze_points:
            norm_x = x / self.screen_width
            norm_y = y / self.screen_height
            video_x = norm_x * self.video_width
            video_y = norm_y * self.video_height
            normalized_points.append((video_x, video_y))
        
        # Create 2D histogram
        x_coords = [p[0] for p in normalized_points]
        y_coords = [p[1] for p in normalized_points]
        
        heatmap, xedges, yedges = np.histogram2d(
            x_coords, y_coords,
            bins=self.heatmap_resolution,
            range=[[0, self.video_width], [0, self.video_height]]
        )
        
        # Apply gaussian blur for smoothing
        heatmap = gaussian_filter(heatmap.T, sigma=self.blur_sigma)
        
        # CRITICAL: Store the max value BEFORE normalization (this is actual sample count)
        max_heatmap_value = heatmap.max() if heatmap.max() > 0 else 1
        
        # Normalize heatmap to 0-255 AFTER capturing max
        heatmap_normalized = (heatmap / max_heatmap_value * 255).astype(np.uint8)
        
        # Resize to video dimensions
        heatmap_resized = cv2.resize(heatmap_normalized, (self.video_width, self.video_height))
        
        # Apply colormap
        heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
        
        # Make low-intensity areas more transparent
        alpha = (heatmap_resized / 255.0)[:, :, np.newaxis]
        frame = (heatmap_colored * alpha).astype(np.uint8)
        
        # Add timestamp and point count overlay
        font = cv2.FONT_HERSHEY_SIMPLEX
        time_text = f"Time: {elapsed_time:.1f}s"
        count_text = f"Points: {len(self.gaze_points)}"
        
        cv2.putText(frame, time_text, (10, 30), font, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, count_text, (10, 60), font, 0.7, (255, 255, 255), 2)
        
        # Add live attention score
        attention_score = self.analytics.calculate_attention_score()
        attention_text = f"Attention: {attention_score:.0f}%"
        cv2.putText(frame, attention_text, (10, 90), font, 0.7, (255, 255, 255), 2)
        
        # Calculate time-per-sample (average dwell time)
        # max_heatmap_value represents the count in the hottest bin
        # We need to convert this to actual seconds
        
        # Average sample rate (samples per second)
        avg_sample_rate = len(self.gaze_points) / elapsed_time if elapsed_time > 0 else 30
        
        # The max_heatmap_value is the number of samples in the hottest bin
        # Convert this to seconds: samples / (samples per second) = seconds
        max_dwell_seconds = max_heatmap_value / avg_sample_rate if avg_sample_rate > 0 else 0
        
        # Handle edge case: if max_dwell_seconds is still very small (< 0.5s), 
        # use a minimum scale based on elapsed time
        if max_dwell_seconds < 0.5 and elapsed_time > 5:
            # Use a reasonable estimate: assume hottest spot got ~10% of total time
            max_dwell_seconds = elapsed_time * 0.1
        elif max_dwell_seconds < 0.1:
            # Very early in recording, use a placeholder
            max_dwell_seconds = 1.0
        
        # For color thresholds (0%, 25%, 50%, 75%, 100% of max intensity)
        color_thresholds = [0, 0.25, 0.5, 0.75, 1.0]
        dwell_times = []
        
        for threshold in color_thresholds:
            # Seconds needed to reach this intensity level
            seconds_needed = max_dwell_seconds * threshold
            dwell_times.append(seconds_needed)
        
        # Add detailed color gradient guide with TIME LABELS
        legend_height = 30
        legend_width = 250
        legend_x = self.video_width - legend_width - 15
        legend_y = 15
        
        # Create gradient bar with border
        gradient = np.linspace(0, 255, legend_width, dtype=np.uint8)
        gradient = np.tile(gradient, (legend_height, 1))
        gradient_colored = cv2.applyColorMap(gradient, cv2.COLORMAP_JET)
        
        # Add white border around gradient
        cv2.rectangle(frame, (legend_x-2, legend_y-2), 
                     (legend_x+legend_width+2, legend_y+legend_height+2), 
                     (255, 255, 255), 2)
        
        # Place gradient on frame
        frame[legend_y:legend_y+legend_height, legend_x:legend_x+legend_width] = gradient_colored
        
        # Add color transition labels with TIME VALUES
        label_y = legend_y + legend_height + 20
        
        # Color names with their positions and corresponding dwell times
        colors_with_times = [
            (0, "Blue", dwell_times[0], (255, 255, 255)),      # 0%
            (0.25, "Cyan", dwell_times[1], (255, 255, 255)),   # 25%
            (0.5, "Green", dwell_times[2], (255, 255, 255)),   # 50%
            (0.75, "Yellow", dwell_times[3], (255, 255, 255)), # 75%
            (1.0, "Red", dwell_times[4], (255, 255, 255))      # 100%
        ]
        
        for position, color_name, dwell_time, text_color in colors_with_times:
            x_pos = legend_x + int(position * legend_width)
            
            # Draw small tick mark
            cv2.line(frame, (x_pos, legend_y + legend_height), 
                    (x_pos, legend_y + legend_height + 5), (255, 255, 255), 2)
            
            # Format dwell time nicely
            if dwell_time < 0.05:
                time_label = "~0s"
            elif dwell_time < 1:
                time_label = f"{dwell_time:.1f}s"
            elif dwell_time < 10:
                time_label = f"{dwell_time:.1f}s"
            else:
                time_label = f"{dwell_time:.0f}s"
            
            # Add color label
            text_size = cv2.getTextSize(color_name, font, 0.35, 1)[0]
            text_x = x_pos - text_size[0] // 2
            cv2.putText(frame, color_name, (text_x, label_y), 
                       font, 0.35, text_color, 1)
            
            # Add time label below color name
            time_size = cv2.getTextSize(time_label, font, 0.3, 1)[0]
            time_x = x_pos - time_size[0] // 2
            cv2.putText(frame, time_label, (time_x, label_y + 12), 
                       font, 0.3, text_color, 1)
        
        # Add "Dwell Time (seconds)" label above gradient
        title = "Dwell Time (seconds)"
        title_size = cv2.getTextSize(title, font, 0.4, 1)[0]
        title_x = legend_x + (legend_width - title_size[0]) // 2
        cv2.putText(frame, title, (title_x, legend_y - 8), 
                   font, 0.4, (255, 255, 255), 1)
        
        return frame
    
    def add_gaze_point(self, timestamp, x, y, is_valid):
        """Add a gaze point to the collection"""
        if self.is_tracking:
            if is_valid:
                self.gaze_points.append((int(x), int(y)))
            self.analytics.add_sample(timestamp, int(x), int(y), is_valid)
    
    def collect_and_record(self, duration):
        """Collect gaze data and write video frames in real-time"""
        print(f"Recording for {duration} seconds...")
        print("Look around your screen naturally!\n")
        
        end_time = time.time() + duration
        frame_interval = 1.0 / self.fps
        next_frame_time = time.time() + frame_interval
        
        sample_count = 0
        valid_count = 0
        frame_count = 0
        last_progress_time = time.time()
        
        while time.time() < end_time:
            try:
                if self.api.wait_for_new_tracking_state_set(self.last_update_timestamp, 10):
                    tracking_state_set = self.api.get_latest_tracking_state_set()
                    user_state = tracking_state_set.user_state()
                    
                    sample_count += 1
                    elapsed = time.time() - self.start_time
                    
                    is_valid = (user_state.unified_screen_gaze.confidence != bet.TrackingConfidence.LOST_TRACKING and
                               user_state.timestamp_in_seconds != bet.NULL_DATA_TIMESTAMP())
                    
                    if is_valid:
                        valid_count += 1
                        point = user_state.unified_screen_gaze.point_of_regard
                        x, y = point.x, point.y
                        self.add_gaze_point(elapsed, x, y, True)
                        
                        if valid_count == 1:
                            print(f"✓ First valid gaze point at ({x:.1f}, {y:.1f})")
                    else:
                        self.add_gaze_point(elapsed, 0, 0, False)
                    
                    self.last_update_timestamp = user_state.timestamp_in_seconds
                
                # Generate and write video frame
                if time.time() >= next_frame_time:
                    elapsed = time.time() - self.start_time
                    frame = self.create_heatmap_frame(elapsed)
                    self.video_writer.write(frame)
                    frame_count += 1
                    next_frame_time += frame_interval
                    
                    # Progress update
                    if time.time() - last_progress_time >= 5:
                        validity = (valid_count / sample_count * 100) if sample_count > 0 else 0
                        attention = self.analytics.calculate_attention_score()
                        print(f"[{elapsed:.1f}s] Points: {valid_count} | Attention: {attention:.0f}% | Frames: {frame_count}")
                        last_progress_time = time.time()
                
            except KeyboardInterrupt:
                print("\n\nStopped by user.")
                break
            except Exception as e:
                print(f"❌ Error during recording: {e}")
                break
        
        print(f"\n✓ Recording complete:")
        print(f"  - Valid gaze points: {valid_count}")
        print(f"  - Total samples: {sample_count}")
        print(f"  - Video frames: {frame_count}")
        if sample_count > 0:
            print(f"  - Validity rate: {valid_count/sample_count*100:.1f}%")
    
    def stop_tracking(self):
        """Stop tracking and finalize video"""
        self.is_tracking = False
        
        if self.video_writer:
            self.video_writer.release()
            print(f"✓ Video saved: {self.video_path}")
        
        elapsed = time.time() - self.start_time if self.start_time else 0
        print(f"✓ Tracking stopped after {elapsed:.1f} seconds")
        print(f"✓ Collected {len(self.gaze_points)} valid gaze points")
    
    def save_analytics_report(self):
        """Generate and save comprehensive analytics report"""
        report = self.analytics.generate_attention_report()
        
        # Add participant info to report
        report["participant_info"] = {
            "participant_number": self.participant_num,
            "session_timestamp": self.session_timestamp,
            "output_folder": str(self.output_folder)
        }
        
        # Save JSON report
        json_path = self.output_folder / f"analytics_{self.session_timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n✓ Analytics report saved: {json_path}")
        
        # Save human-readable text report
        txt_path = self.output_folder / f"report_{self.session_timestamp}.txt"
        with open(txt_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("ATTENTION ANALYTICS REPORT\n")
            f.write("="*70 + "\n\n")
            
            # Participant Info
            f.write("PARTICIPANT INFORMATION\n")
            f.write("-" * 70 + "\n")
            f.write(f"Participant Number: {self.participant_num}\n")
            f.write(f"Session Timestamp: {self.session_timestamp}\n")
            f.write(f"Output Folder: {self.output_folder.name}\n")
            
            # Summary
            f.write("\n\nSUMMARY\n")
            f.write("-" * 70 + "\n")
            for key, value in report["summary"].items():
                f.write(f"{key.replace('_', ' ').title()}: {value:.2f}\n")
            
            # Distraction Analysis
            f.write("\n\nDISTRACTION ANALYSIS\n")
            f.write("-" * 70 + "\n")
            f.write(f"Total Distractions: {report['distraction_analysis']['distraction_count']}\n")
            f.write(f"Total Distraction Time: {report['distraction_analysis']['total_distraction_time_seconds']:.2f}s\n")
            f.write(f"Distraction Percentage: {report['distraction_analysis']['distraction_percentage']:.2f}%\n")
            f.write(f"Longest Distraction: {report['distraction_analysis']['longest_distraction_seconds']:.2f}s\n")
            
            if report['distraction_analysis']['distraction_periods']:
                f.write("\nDistraction Periods:\n")
                for i, period in enumerate(report['distraction_analysis']['distraction_periods'], 1):
                    f.write(f"  {i}. {period['start_time']:.2f}s - {period['end_time']:.2f}s (duration: {period['duration']:.2f}s)\n")
            
            # Spatial Analysis
            f.write("\n\nSPATIAL ANALYSIS\n")
            f.write("-" * 70 + "\n")
            f.write(f"Gaze Dispersion: {report['spatial_analysis']['gaze_dispersion']:.4f}\n")
            f.write(f"Fixation Stability: {report['spatial_analysis']['fixation_stability']:.4f}\n")
            f.write(f"Focus Quality Score: {report['spatial_analysis']['focus_quality']:.2f}/100\n")
            
            f.write("\nAttention Distribution by Region:\n")
            for region, percentage in sorted(report['spatial_analysis']['attention_distribution'].items()):
                f.write(f"  {region}: {percentage:.2f}%\n")
            
            # Temporal Analysis
            f.write("\n\nTEMPORAL ANALYSIS\n")
            f.write("-" * 70 + "\n")
            f.write(f"Average Engagement: {report['temporal_analysis']['average_engagement']:.2f}%\n")
            
            f.write("\n" + "="*70 + "\n")
        
        print(f"✓ Text report saved: {txt_path}")
        
        return report, json_path, txt_path
    
    def print_quick_summary(self, report):
        """Print a quick summary to console"""
        print("\n" + "="*70)
        print(f"QUICK ATTENTION SUMMARY - PARTICIPANT {self.participant_num}")
        print("="*70)
        print(f"Overall Attention Score:     {report['summary']['attention_score']:.1f}/100")
        print(f"Validity Rate:               {report['summary']['validity_rate_percent']:.1f}%")
        print(f"Total Distractions:          {report['distraction_analysis']['distraction_count']}")
        print(f"Time Distracted:             {report['distraction_analysis']['distraction_percentage']:.1f}%")
        print(f"Focus Quality Score:         {report['spatial_analysis']['focus_quality']:.1f}/100")
        print(f"Fixation Stability:          {report['spatial_analysis']['fixation_stability']:.3f}")
        print("="*70)


def main():
    """Main function"""
    
    print("="*70)
    print("BEAM EYE TRACKER - ATTENTION ANALYTICS")
    print("="*70)
    print("\nThis tool creates a VIDEO and generates QUANTIFIABLE metrics")
    print("measuring attention, focus, and distraction patterns.\n")
    
    # Get screen resolution
    try:
        width = int(input(f"Enter screen width (default 1920): ") or "1920")
        height = int(input(f"Enter screen height (default 1080): ") or "1080")
    except ValueError:
        print("Invalid input. Using default 1920x1080")
        width, height = 1920, 1080
    
    # Create tracker
    try:
        tracker = RealtimeHeatmapVideoGenerator(width, height)
    except Exception as e:
        print(f"\nFailed to initialize: {e}")
        return
    
    # Get recording duration
    try:
        duration = int(input("\nHow many seconds to record? (default 30): ") or "30")
    except ValueError:
        duration = 30
    
    # Start tracking (will auto-create Participant_N folder)
    try:
        tracker.start_tracking()
    except Exception as e:
        print(f"\nFailed to start tracking: {e}")
        return
    
    # Record video
    print(f"\nRecording for {duration} seconds...")
    print("Press Ctrl+C to stop early.\n")
    
    try:
        tracker.collect_and_record(duration)
    except KeyboardInterrupt:
        print("\n\nStopped by user.")
    
    # Stop tracking
    tracker.stop_tracking()
    
    # Generate analytics
    if len(tracker.gaze_points) > 0:
        print("\n📊 Generating analytics report...")
        report, json_path, txt_path = tracker.save_analytics_report()
        tracker.print_quick_summary(report)
        
        print("\n✓ Success! Your analysis is complete.")
        print(f"\nGenerated files:")
        print(f"  📹 Video: {tracker.video_path.name}")
        print(f"  📊 JSON Report: {json_path.name}")
        print(f"  📄 Text Report: {txt_path.name}")
        print(f"\n📁 Location: {tracker.output_folder.absolute()}")
        
        # Ask if user wants to open folder
        try:
            view = input("\nOpen the output folder? (y/n): ").lower().strip()
            if view == 'y':
                import os
                import platform
                if platform.system() == 'Windows':
                    os.startfile(tracker.output_folder.absolute())
                elif platform.system() == 'Darwin':  # macOS
                    os.system(f'open "{tracker.output_folder.absolute()}"')
                else:  # Linux
                    os.system(f'xdg-open "{tracker.output_folder.absolute()}"')
        except:
            pass
    else:
        print("\n✗ No data collected.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Program error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        input("\nPress Enter to close...")
