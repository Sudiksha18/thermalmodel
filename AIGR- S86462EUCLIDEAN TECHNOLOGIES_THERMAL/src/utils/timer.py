"""
Timer and FPS counter utilities for performance monitoring.
Part of the EuclideanTechnologies Thermal Anomaly Detection System.
"""

import time
from typing import Optional, List
from collections import deque
import statistics


class Timer:
    """High-precision timer for measuring execution time."""
    
    def __init__(self):
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.total_time: float = 0.0
        self.lap_times: List[float] = []
    
    def start(self) -> None:
        """Start the timer."""
        self.start_time = time.perf_counter()
        self.end_time = None
    
    def stop(self) -> float:
        """Stop the timer and return elapsed time in seconds."""
        if self.start_time is None:
            raise ValueError("Timer not started")
        
        self.end_time = time.perf_counter()
        elapsed = self.end_time - self.start_time
        self.total_time += elapsed
        return elapsed
    
    def lap(self) -> float:
        """Record a lap time and return elapsed time since start."""
        if self.start_time is None:
            raise ValueError("Timer not started")
        
        current_time = time.perf_counter()
        lap_time = current_time - self.start_time
        self.lap_times.append(lap_time)
        return lap_time
    
    def reset(self) -> None:
        """Reset the timer."""
        self.start_time = None
        self.end_time = None
        self.total_time = 0.0
        self.lap_times.clear()
    
    def elapsed(self) -> float:
        """Get elapsed time without stopping the timer."""
        if self.start_time is None:
            return 0.0
        
        current_time = time.perf_counter()
        return current_time - self.start_time
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


class FPSCounter:
    """Frame rate counter for real-time performance monitoring."""
    
    def __init__(self, window_size: int = 30):
        """
        Initialize FPS counter.
        
        Args:
            window_size: Number of frames to average over
        """
        self.window_size = window_size
        self.frame_times = deque(maxlen=window_size)
        self.last_frame_time = None
        self.frame_count = 0
    
    def update(self) -> float:
        """
        Update the FPS counter with a new frame.
        
        Returns:
            Current FPS or 0.0 if not enough frames
        """
        current_time = time.perf_counter()
        
        if self.last_frame_time is not None:
            frame_time = current_time - self.last_frame_time
            self.frame_times.append(frame_time)
        
        self.last_frame_time = current_time
        self.frame_count += 1
        
        return self.get_fps()
    
    def get_fps(self) -> float:
        """
        Get current FPS.
        
        Returns:
            Current FPS or 0.0 if not enough frames
        """
        if len(self.frame_times) < 2:
            return 0.0
        
        avg_frame_time = statistics.mean(self.frame_times)
        return 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0
    
    def get_stats(self) -> dict:
        """
        Get detailed FPS statistics.
        
        Returns:
            Dictionary with FPS statistics
        """
        if len(self.frame_times) < 2:
            return {
                'fps': 0.0,
                'min_fps': 0.0,
                'max_fps': 0.0,
                'avg_frame_time_ms': 0.0,
                'frame_count': self.frame_count
            }
        
        frame_times_list = list(self.frame_times)
        fps_values = [1.0 / ft for ft in frame_times_list if ft > 0]
        
        return {
            'fps': self.get_fps(),
            'min_fps': min(fps_values) if fps_values else 0.0,
            'max_fps': max(fps_values) if fps_values else 0.0,
            'avg_frame_time_ms': statistics.mean(frame_times_list) * 1000,
            'frame_count': self.frame_count
        }
    
    def reset(self) -> None:
        """Reset the FPS counter."""
        self.frame_times.clear()
        self.last_frame_time = None
        self.frame_count = 0


class ProfilerContext:
    """Context manager for profiling code sections."""
    
    def __init__(self, name: str, print_result: bool = True):
        """
        Initialize profiler context.
        
        Args:
            name: Name of the section being profiled
            print_result: Whether to print timing result
        """
        self.name = name
        self.print_result = print_result
        self.timer = Timer()
        self.elapsed_time = 0.0
    
    def __enter__(self):
        """Enter the profiler context."""
        self.timer.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the profiler context."""
        self.elapsed_time = self.timer.stop()
        
        if self.print_result:
            print(f"⏱️  {self.name}: {self.elapsed_time * 1000:.2f}ms")


def profile_function(func):
    """Decorator to profile function execution time."""
    def wrapper(*args, **kwargs):
        with ProfilerContext(func.__name__):
            return func(*args, **kwargs)
    return wrapper


# Example usage and testing
if __name__ == "__main__":
    # Test Timer
    print("Testing Timer...")
    timer = Timer()
    timer.start()
    time.sleep(0.1)
    elapsed = timer.stop()
    print(f"Timer test: {elapsed:.3f}s (expected ~0.1s)")
    
    # Test FPS Counter
    print("\nTesting FPS Counter...")
    fps_counter = FPSCounter()
    
    for i in range(10):
        time.sleep(0.016)  # Simulate ~60 FPS
        fps = fps_counter.update()
        if i > 5:  # Let it stabilize
            print(f"Frame {i}: {fps:.1f} FPS")
    
    stats = fps_counter.get_stats()
    print(f"FPS Stats: {stats}")
    
    # Test ProfilerContext
    print("\nTesting ProfilerContext...")
    with ProfilerContext("Test operation"):
        time.sleep(0.05)
    
    print("✅ All timer utilities working correctly!")