"""Fail-closed single-cycle controller. Arduino owns its own 3 s timer.

The UI never waits forever: missed OPENED/CLOSED acknowledgements transition to
RECOVERING (CLOSE retry) and finally ERROR with a visible recovery action.
Physical CLOSED is accepted ONLY when acknowledged by the Arduino.
"""
from __future__ import annotations
import threading
import time
from typing import Any

try:
    import serial
    from serial.tools import list_ports
except ImportError:
    serial, list_ports = None, None


class GateController:
    OPEN_ACK_TIMEOUT = 2.0
    CLOSE_ACK_GRACE = 1.25
    CLOSE_ACK_TIMEOUT = 2.0

    def __init__(self, duration_ms: int = 3000):
        self.duration_ms = duration_ms
        self.lock = threading.RLock()
        self.ser: Any = None
        self.port: str | None = None
        self.state = 'CLOSED'
        self.deadline = 0.0
        self.command_started = 0.0
        self.closing_started = 0.0
        self.last_event = 'Ready for first image'
        self.last_error: str | None = None
        self.hardware_warning: str | None = None
        self._ready = threading.Event()
        self._reader: threading.Thread | None = None
        self._running = False

    def ports(self):
        if list_ports is None:
            return []
        return [{'device': p.device, 'description': p.description} for p in list_ports.comports()]

    def _set_error(self, msg: str):
        self.state = 'ERROR'
        self.deadline = 0.0
        self.last_error = msg
        self.last_event = msg

    def _request_close_locked(self, reason: str):
        if self.ser is None:
            self.state = 'CLOSED'
            self.deadline = 0.0
            return
        try:
            self.ser.write(b'CLOSE\n')
            self.ser.flush()
            self.state = 'CLOSING'
            self.deadline = 0.0
            self.closing_started = time.monotonic()
            self.last_event = reason + ' — CLOSE command sent; waiting for Arduino'
            self.last_error = None
        except Exception as e:
            self._set_error(f'Serial CLOSE failed: {e}. Inspect gate and reconnect Arduino.')

    def _refresh_locked(self):
        now = time.monotonic()
        if self.ser is None:
            if self.state == 'OPEN' and now >= self.deadline:
                self.state = 'CLOSED'
                self.deadline = 0.0
                self.last_event = 'Simulation finished: CLOSED — next photo ready'
            return
        if not getattr(self.ser, 'is_open', False):
            self._set_error('Arduino serial port disconnected. Check USB and reconnect.')
        elif self.state == 'OPENING' and now - self.command_started > self.OPEN_ACK_TIMEOUT:
            self._request_close_locked('No OPENED reply within 2 seconds: recovery')
        elif self.state == 'OPEN' and now > self.deadline + self.CLOSE_ACK_GRACE:
            self._request_close_locked('CLOSED acknowledgement missing: recovery')
        elif self.state == 'CLOSING' and now - self.closing_started > self.CLOSE_ACK_TIMEOUT:
            self._set_error('Arduino did not confirm CLOSED. The gate may be moving. Check power, USB, sketch, and servo. Click Retry CLOSE after inspecting it.')

    def status(self):
        with self.lock:
            self._refresh_locked()
            return {
                'connected': bool(self.ser and getattr(self.ser, 'is_open', False) and self._ready.is_set()),
                'port': self.port,
                'state': self.state,
                'remaining_ms': max(0, round((self.deadline - time.monotonic()) * 1000)) if self.state == 'OPEN' else 0,
                'duration_ms': self.duration_ms,
                'mode': 'HARDWARE' if self.ser is not None else 'SIMULATION',
                'last_event': self.last_event,
                'error': self.last_error,
                'hardware_warning': self.hardware_warning,
                'can_accept_image': self.state == 'CLOSED',
                'needs_attention': self.state == 'ERROR',
            }

    def connect(self, port: str):
        if serial is None:
            raise RuntimeError('pyserial missing. Install requirements.txt first.')
        if port not in {x['device'] for x in self.ports()}:
            raise RuntimeError('Select a listed COM port. Plug in your Arduino and refresh ports.')
        with self.lock:
            self._refresh_locked()
            if self.state != 'CLOSED' or self.ser is not None:
                raise RuntimeError('Gate must be CLOSED and disconnected first.')
            self._ready.clear()
            self.last_error = None
            try:
                device = serial.Serial(port=port, baudrate=9600, timeout=.15, write_timeout=.5)
            except Exception as e:
                raise RuntimeError(f'Cannot open {port}: {e}. Close Arduino Serial Monitor.') from e
            self.ser = device
            self.port = port
            self._running = True
            self.state = 'CONNECTING'
            self._reader = threading.Thread(target=self._read_serial, args=(device,), daemon=True)
            self._reader.start()
        # Uno commonly resets on opening USB: READY arrives after boot.
        # New firmware additionally supports STATUS for no-reset boards.
        if not self._ready.wait(2.5):
            try:
                device.write(b'STATUS\n')
                device.flush()
            except Exception:
                pass
        if not self._ready.wait(3.0):
            self.disconnect(force=True)
            raise RuntimeError('No READY / STATE:CLOSED response. Upload new .ino, verify correct COM and 9600 baud.')
        with self.lock:
            self.hardware_warning = None
            self.last_event = f'Arduino connected on {port}. Gate initialized CLOSED.'
        return self.status()

    def _read_serial(self, device):
        while self._running and device is self.ser:
            try:
                raw = device.readline()
                if not raw:
                    continue
                message = raw.decode('ascii', errors='ignore').strip().upper()
                with self.lock:
                    if message in ('READY', 'STATE:CLOSED'):
                        self.state = 'CLOSED'
                        self.deadline = 0.0
                        self.last_error = None
                        self._ready.set()
                        self.last_event = 'Arduino confirms CLOSED; ready for next image'
                    elif message == 'STATE:OPEN':
                        self._set_error('Arduino reports OPEN at connection. Press Retry CLOSE and confirm physical position.')
                        self._ready.set()
                    elif message == 'OPENED':
                        if self.state == 'OPENING':
                            self.state = 'OPEN'
                            self.deadline = time.monotonic() + self.duration_ms / 1000
                            self.last_event = 'Arduino confirmed OPENED. Closing automatically after 3 seconds.'
                    elif message == 'CLOSED':
                        if self.state in ('OPENING', 'OPEN', 'CLOSING', 'ERROR'):
                            self.state = 'CLOSED'
                            self.deadline = 0.0
                            self.last_error = None
                            self.last_event = 'Arduino confirmed CLOSED — next photo ready'
                    elif message == 'BUSY':
                        self.last_event = 'Arduino rejected duplicate OPEN: busy'
                    elif message == 'ERROR':
                        self._set_error('Arduino reported unknown command. Verify latest sketch and baud rate.')
            except Exception as exc:
                with self.lock:
                    if device is self.ser and self._running:
                        self._set_error(f'Serial read failed: {exc}. Check USB and reconnect.')
                break

    def trigger(self):
        with self.lock:
            self._refresh_locked()
            if self.state != 'CLOSED':
                return False, 'Gate not ready. Check status; do not send another OPEN.'
            if self.ser is None:
                self.state = 'OPEN'
                self.deadline = time.monotonic() + self.duration_ms / 1000
                self.last_event = 'SIMULATION: gate OPEN for 3 seconds'
                return True, 'Simulation — no physical servo connected'
            if not self._ready.is_set():
                return False, 'Arduino not ready'
            try:
                self.ser.write(b'OPEN\n')
                self.ser.flush()
            except Exception as e:
                self._set_error(f'Could not send OPEN: {e}')
                return False, self.last_error
            self.state = 'OPENING'
            self.command_started = time.monotonic()
            self.deadline = 0.0
            self.last_event = 'OPEN sent: waiting for Arduino OPENED reply'
            return True, 'OPEN sent. Arduino will close itself after 3 seconds.'

    def retry_close(self):
        """Recovery requires an actual CLOSED serial acknowledgement, never a blind reset."""
        with self.lock:
            self._refresh_locked()
            if self.ser is None:
                raise RuntimeError('No Arduino connected; simulation auto-closes in 3 seconds.')
            if self.state == 'CLOSED':
                return self.status()
            self._request_close_locked('Manual recovery')
            return self.status()

    def disconnect(self, force: bool = False):
        with self.lock:
            self._refresh_locked()
            if self.ser is None:
                return self.status()
            if self.state != 'CLOSED' and not force:
                raise RuntimeError('Wait for CLOSED, or inspect hardware then use Force CLOSE + Disconnect.')
            device = self.ser
            try:
                device.write(b'CLOSE\n')
                device.flush()
            except Exception:
                pass
            self._running = False
            self.ser = None
            self.port = None
            self._ready.clear()
            try:
                device.close()
            except Exception:
                pass
            self.state = 'CLOSED'  # explicitly labeled SIMULATION after disconnection
            self.deadline = 0.0
            self.last_error = None
            if force:
                self.hardware_warning = 'Hardware closure NOT confirmed. Check the real gate before re-enabling the Arduino.'
                self.last_event = 'Disconnected into SIMULATION; physical gate position unverified'
            else:
                self.hardware_warning = None
                self.last_event = 'Arduino disconnected; SIMULATION mode enabled'
            return self.status()

    def reset_error(self):
        with self.lock:
            if self.ser is not None:
                raise RuntimeError('Use Retry CLOSE. It must receive CLOSED before new image can open the servo.')
            self.state = 'CLOSED'
            self.last_error = None
            return self.status()


controller = GateController()
