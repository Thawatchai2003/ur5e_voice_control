#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
import threading
import time
import queue
import numpy as np
import sounddevice as sd  # เพิ่มตรงนี้
import speech_recognition as sr

RATE = 16000  # sample rate

class GoogleSTTNode(Node):
    def __init__(self):
        super().__init__('google_stt_node')

        # QUEUE สำหรับเก็บเสียง
        self.audio_queue = queue.Queue()

        # STT recognizer
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 300
        self.recognizer.dynamic_energy_threshold = False

        # STATE CONTROL
        self.state = "IDLE"
        self.armed = False
        self.woke_up = False
        self.use_wake_mode = True
        self._need_flush = False
        self._in_recognize = False

        # PUBLISHER
        self.pub = self.create_publisher(String, 'voice/heard_text', 10)
        self.pub_wake = self.create_publisher(Bool, 'voice/wake_detected', 10)

        # THREAD สำหรับ PROCESS LOOP
        self.worker = threading.Thread(target=self.process_loop)
        self.worker.daemon = True
        self.worker.start()

        # THREAD สำหรับ CAPTURE เสียงจากไมโครโฟน
        self.mic_worker = threading.Thread(target=self.capture_audio)
        self.mic_worker.daemon = True
        self.mic_worker.start()

        self.get_logger().info("Google STT Node Ready (Mono Mic Mode)")

    def capture_audio(self):
        """Capture audio from mic and put into queue"""
        def callback(indata, frames, time_info, status):
            if status:
                self.get_logger().warning(f"Mic status: {status}")
            self.audio_queue.put(indata.copy().tobytes())
        with sd.InputStream(samplerate=RATE, channels=1, dtype='int16', callback=callback):
            while rclpy.ok():
                sd.sleep(100)

    def process_loop(self):
        buffer = b""
        while rclpy.ok():
            try:
                data = self.audio_queue.get(timeout=0.1)
                buffer += data
            except queue.Empty:
                continue

            # ตัวอย่างง่าย ๆ ตรวจจับเสียงแล้วส่ง STT
            if len(buffer) > RATE * 2 * 1:  # 1 วินาที buffer
                self.do_stt(buffer)
                buffer = b""

    def do_stt(self, pcm_data: bytes):
        try:
            audio = sr.AudioData(pcm_data, RATE, 2)
            text = self.recognizer.recognize_google(audio, language="th-TH").strip()
            if text:
                msg = String()
                msg.data = text
                self.pub.publish(msg)
                self.get_logger().info(f"Recognized: {text}")
        except Exception as e:
            self.get_logger().warning(f"STT fail: {e}")

def main():
    rclpy.init()
    node = GoogleSTTNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
