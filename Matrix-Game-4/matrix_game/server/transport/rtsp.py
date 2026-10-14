"""Bounded RGB to RTSP publisher. Requires GI only in the sidecar."""
from collections import OrderedDict
import os
import re
import threading
import time
import numpy as np

class RTSPPublisher:
    def __init__(self, width, height, fps, generation_id, on_frame):
        # RTSP is loopback-only. Avoid libproxy's native unwinder conflict with
        # NumPy on the Ubuntu node; no desktop proxy discovery is needed here.
        os.environ.setdefault('GIO_USE_PROXY_RESOLVER', 'dummy')
        import gi
        gi.require_version('Gst', '1.0')
        gi.require_version('GstRtp', '1.0')
        from gi.repository import Gst, GstRtp

        if not re.fullmatch(r'[a-f0-9]{32}', generation_id):
            raise ValueError('Invalid generation ID')
        Gst.init(None)
        self.Gst, self.GstRtp = Gst, GstRtp
        self.fps = fps
        self.on_frame = on_frame
        self.pending = OrderedDict()
        self.lock = threading.Lock()
        self.eos = threading.Event()
        self.closed = False
        self.error = None
        self.encoded_frames = 0
        port = int(os.getenv('STUDIO_WEBRTC_RTSP_PORT', '8555'))
        bitrate = int(os.getenv('STUDIO_MEDIA_MAXRATE_KBPS', '1800'))
        if not 1 <= port <= 65535 or not 128 <= bitrate <= 50000:
            raise ValueError('Invalid publisher port or bitrate')
        self.pipeline = Gst.parse_launch(
            'appsrc name=source is-live=true format=time block=false '
            'max-buffers=8 max-bytes=0 max-time=0 leaky-type=downstream '
            f'caps=video/x-raw,format=RGB,width={width},height={height},framerate={fps}/1 '
            '! videoconvert ! video/x-raw,format=I420 '
            '! x264enc tune=zerolatency speed-preset=veryfast threads=2 bframes=0 '
            f'bitrate={bitrate} vbv-buf-capacity=100 key-int-max={max(1, fps // 2)} '
            '! video/x-h264,profile=constrained-baseline '
            '! h264parse config-interval=-1 '
            f'! rtspclientsink name=sink protocols=tcp latency=0 tcp-timeout=3000000 '
            f'location=rtsp://127.0.0.1:{port}/native-{generation_id}')
        self.source = self.pipeline.get_by_name('source')
        self.pipeline.get_by_name('sink').connect('new-payloader', self._payloader)
        self.bus = self.pipeline.get_bus()
        if self.pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
            self.close()
            raise RuntimeError('GStreamer publisher failed to start')

    def _payloader(self, sink, payloader):
        Gst = self.Gst
        payloader.set_property('timestamp-offset', 0)
        payloader.set_property('mtu', 1200)
        payloader.get_static_pad('src').add_probe(
            Gst.PadProbeType.BUFFER | Gst.PadProbeType.BUFFER_LIST, self._encoded)

    def _encoded(self, pad, info):
        Gst = self.Gst
        try:
            if info.type & Gst.PadProbeType.BUFFER_LIST:
                values = info.get_buffer_list()
                buffers = [values.get(i) for i in range(values.length())]
            else:
                buffers = [info.get_buffer()]
            event = pad.get_sticky_event(Gst.EventType.SEGMENT, 0)
            if event is None:
                raise RuntimeError('RTP encoder has no presentation segment')
            segment = event.parse_segment()
            for buffer in buffers:
                ok, packet = self.GstRtp.RTPBuffer.map(buffer, Gst.MapFlags.READ)
                if not ok:
                    raise RuntimeError('Cannot read encoded RTP timestamp')
                try:
                    marker, timestamp = packet.get_marker(), packet.get_timestamp()
                finally:
                    packet.unmap()
                if not marker:
                    continue
                pts = segment.to_running_time(Gst.Format.TIME, buffer.pts)
                if pts == Gst.CLOCK_TIME_NONE:
                    raise RuntimeError('RTP frame has no valid presentation timestamp')
                index = round(pts * self.fps / Gst.SECOND)
                with self.lock:
                    metadata = self.pending.pop(index, None)
                    closed = self.closed
                    self.encoded_frames += 1
                if closed:
                    continue
                if metadata is None:
                    raise RuntimeError(f'No game state for encoded frame {index}')
                self.on_frame(dict(metadata, rtp_timestamp=int(timestamp), encoded_wall_ms=time.time()*1000))
        except Exception as exc:
            self.error = exc
        return Gst.PadProbeReturn.OK

    def check(self):
        if self.error is not None:
            raise RuntimeError(f'WebRTC publisher failed: {self.error}') from self.error
        message = self.bus.pop_filtered(self.Gst.MessageType.ERROR | self.Gst.MessageType.EOS)
        if message and message.type == self.Gst.MessageType.EOS:
            self.eos.set()
        elif message:
            error, debug = message.parse_error()
            self.error = RuntimeError(f'GStreamer: {error.message}; {debug}')
            raise self.error

    def push(self, rgb, game):
        self.check()
        index = game['frame']
        wall_ms = (time.time() + float(game['capture_monotonic']) - time.monotonic()) * 1000
        with self.lock:
            self.pending[index] = dict(game=game, capture_wall_ms=wall_ms)
            while len(self.pending) > 120:
                self.pending.popitem(last=False)
        Gst = self.Gst
        payload = np.ascontiguousarray(rgb).tobytes()
        buffer = Gst.Buffer.new_allocate(None, len(payload), None)
        buffer.fill(0, payload)
        buffer.pts = buffer.dts = index * Gst.SECOND // self.fps
        buffer.duration = (index+1) * Gst.SECOND // self.fps - buffer.pts
        if self.source.emit('push-buffer', buffer) != Gst.FlowReturn.OK:
            raise RuntimeError('GStreamer refused reference RGB')

    def finish(self):
        self.source.emit('end-of-stream')
        deadline = time.monotonic() + 5
        while not self.eos.is_set():
            self.check()
            if time.monotonic() > deadline:
                raise TimeoutError('GStreamer did not finish encoding')
            self.eos.wait(.01)
        self.check()

    def close(self):
        with self.lock:
            self.closed = True
        if self.pipeline.set_state(self.Gst.State.NULL) == self.Gst.StateChangeReturn.FAILURE:
            raise RuntimeError('GStreamer shutdown failed')
        with self.lock:
            self.pending.clear()

