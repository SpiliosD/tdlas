from __future__ import annotations

# =========================================================
# - Standard library
# =========================================================
import time
import socket
import struct
import subprocess
from typing import Optional, Dict, List, Tuple

# =========================================================
# - Third-party
# =========================================================
import numpy as np

# =========================================================
# - Red Pitaya SCPI helper (generator control)
# =========================================================
import redpitaya_scpi as scpi

# =========================================================
# - ADC CONVERSION CONSTANTS (counts -> volts)
# =========================================================
ADC_BITS = 14
ADC_MAX_COUNT = 2 ** (ADC_BITS - 1)  # 8192


def raw_to_volts(raw_counts: np.ndarray, voltage_range: float) -> np.ndarray:
    return (raw_counts.astype(np.float64) / ADC_MAX_COUNT) * voltage_range


# =========================================================
# - CONFIG FILE PARSER (key=value)
# =========================================================
def load_config_txt(path: str) -> Dict[str, str]:
    cfg: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for raw in f.readlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            cfg[k.strip()] = v.strip()
    return cfg


def get_str(cfg: Dict[str, str], key: str, default: str) -> str:
    return cfg.get(key, default)


def get_float(cfg: Dict[str, str], key: str, default: float) -> float:
    try:
        return float(cfg.get(key, default))
    except Exception:
        return default


def get_int(cfg: Dict[str, str], key: str, default: int) -> int:
    try:
        return int(float(cfg.get(key, default)))
    except Exception:
        return default


def get_bool01(cfg: Dict[str, str], key: str, default: bool) -> bool:
    v = cfg.get(key, None)
    if v is None:
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "on")


# =========================================================
# - STREAMING THROUGHPUT / DECIMATION
# =========================================================
def sampling_to_decimation(fs_req: float, adc_base_hz: float) -> Tuple[int, float]:
    """
    Pick decimation such that Fs_eff ~ fs_req:
        Fs_eff = adc_base_hz / round(adc_base_hz / fs_req)
    """
    if fs_req <= 0:
        raise ValueError("sampling_frequency_hz must be > 0")

    dec = int(round(adc_base_hz / fs_req))
    if dec < 1:
        dec = 1

    fs_eff = adc_base_hz / dec
    return dec, fs_eff


def clamp_rate_for_link(fs_eff: float, bytes_per_sample: int, max_MB_s: float, safety: float) -> float:
    """
    Ensure Fs does not exceed link budget:
        bytes/s = Fs * bytes_per_sample
        Fs_max = (max_MB_s * 1e6) / bytes_per_sample
    safety < 1 keeps headroom for TCP/IP overhead + Python processing.
    """
    fs_max = (max_MB_s * 1e6) / float(bytes_per_sample) * float(safety)
    return min(fs_eff, fs_max)


# =========================================================
# - SSH helpers (auto-start/stop streaming-server)
# =========================================================
def run_ssh_command(
    host: str,
    user: str,
    port: int,
    cmd: str,
    key_path: str = "",
    connect_timeout_s: float = 5.0,
) -> None:
    """
    Uses local 'ssh' executable. Recommended: key-based auth.
    Windows: ensure OpenSSH Client is installed and 'ssh' is in PATH.
    """
    ssh_cmd = [
        "ssh",
        "-p",
        str(port),
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-o",
        f"ConnectTimeout={int(max(1.0, connect_timeout_s))}",
    ]

    if key_path.strip():
        ssh_cmd += ["-i", key_path.strip()]

    ssh_cmd += [f"{user}@{host}", cmd]

    p = subprocess.run(
        ssh_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if p.returncode != 0:
        raise RuntimeError(
            "SSH command failed.\n"
            f"Host: {user}@{host}:{port}\n"
            f"Cmd : {cmd}\n"
            f"RC  : {p.returncode}\n"
            f"STDOUT:\n{p.stdout}\n"
            f"STDERR:\n{p.stderr}\n"
        )


# =========================================================
# - STREAMING CLIENT (TCP port 8900 by default)
# =========================================================
class RPStreamClient:
    """
    Reads continuous ADC samples from the Red Pitaya streaming server.

    Resync strategy:
      - We search for a known 16-byte magic header.
      - If garbage/partial data appears, we discard until we find the magic header again.

    Packet layout used here (commonly observed in Python clients):
      - 16 bytes: magic
      - uint64: index
      - uint64: lost_rate
      - uint32: rate
      - uint32: buffer_size
      - uint32: chan1_size_bytes
      - uint32: chan2_size_bytes
      - uint32: resolution_marker
      - payload: chan1 bytes then chan2 bytes
    """

    MAGIC = b"\x00\x00\x00\x00\xa0\xa0\xa0\xa0\xff\xff\xff\xff\xa0\xa0\xa0\xa0"
    META_FMT = "<QQIIIII"  # little-endian
    META_LEN = struct.calcsize(META_FMT)  # 36
    FIXED_LEN = 16 + META_LEN  # 52

    def __init__(self, host: str, port: int, sock_timeout_s: float = 2.0):
        self.host = host
        self.port = port
        self.sock_timeout_s = sock_timeout_s
        self.s: Optional[socket.socket] = None
        self._buf = bytearray()

    def connect(self) -> None:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(self.sock_timeout_s)
        s.connect((self.host, self.port))
        self.s = s
        self._buf.clear()

    def close(self) -> None:
        if self.s is not None:
            try:
                self.s.close()
            except Exception:
                pass
        self.s = None

    def _recv_into_buffer(self, n_min: int) -> None:
        assert self.s is not None
        while len(self._buf) < n_min:
            chunk = self.s.recv(65536)
            if not chunk:
                raise ConnectionError("Streaming socket closed by peer.")
            self._buf.extend(chunk)

    def _find_magic(self) -> int:
        return self._buf.find(self.MAGIC)

    def read_packet(self) -> Tuple[dict, bytes, bytes]:
        while True:
            if len(self._buf) < self.FIXED_LEN:
                self._recv_into_buffer(self.FIXED_LEN)

            idx = self._find_magic()
            if idx < 0:
                # keep tail in case magic spans boundary
                self._buf[:] = self._buf[-15:]
                self._recv_into_buffer(65536)
                continue

            if idx > 0:
                del self._buf[:idx]

            self._recv_into_buffer(self.FIXED_LEN)

            meta_bytes = self._buf[16:self.FIXED_LEN]
            pkt_index, lost_rate, rate, buffer_size, ch1_sz, ch2_sz, resolution = struct.unpack(
                self.META_FMT, meta_bytes
            )

            total_needed = self.FIXED_LEN + ch1_sz + ch2_sz
            self._recv_into_buffer(total_needed)

            ch1 = bytes(self._buf[self.FIXED_LEN:self.FIXED_LEN + ch1_sz])
            ch2 = bytes(self._buf[self.FIXED_LEN + ch1_sz:total_needed])

            del self._buf[:total_needed]

            meta = {
                "index": int(pkt_index),
                "lost_rate": int(lost_rate),
                "rate": int(rate),
                "buffer_size": int(buffer_size),
                "ch1_size": int(ch1_sz),
                "ch2_size": int(ch2_sz),
                "resolution": int(resolution),
            }
            return meta, ch1, ch2


# =========================================================
# - LIVE PLOT (continuous sliding window)
# =========================================================
class LivePlotIN1:
    def __init__(self, window_s: float):
        import matplotlib.pyplot as plt

        self.plt = plt
        plt.ion()

        self.window_s = float(window_s)
        self.t0: Optional[float] = None

        self.t: List[float] = []
        self.y: List[float] = []

        self.fig, self.ax = plt.subplots(1, 1, figsize=(11, 5))
        self.fig.suptitle("Red Pitaya IN1 - Continuous Stream", fontsize=14, fontweight="bold")
        self.line, = self.ax.plot([], [], linewidth=0.7)
        self.ax.set_xlabel("Relative Time (s)")
        self.ax.set_ylabel("Voltage (V)")
        self.ax.grid(True, alpha=0.3)

    def add_samples(self, t_abs: np.ndarray, y: np.ndarray) -> None:
        if self.t0 is None:
            self.t0 = float(t_abs[0])

        t_rel = (t_abs - self.t0).astype(np.float64)
        self.t.extend(t_rel.tolist())
        self.y.extend(y.astype(np.float64).tolist())

        # Trim to sliding window
        t_now = self.t[-1] if self.t else 0.0
        t_min = t_now - self.window_s

        # Find first index >= t_min
        k = 0
        for i, tv in enumerate(self.t):
            if tv >= t_min:
                k = i
                break
        if k > 0:
            del self.t[:k]
            del self.y[:k]

    def draw(self) -> None:
        self.line.set_data(self.t, self.y)
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        self.plt.pause(0.001)


# =========================================================
# - MAIN
# =========================================================
def main(config_path: str = "config.txt") -> None:
    cfg = load_config_txt(config_path)

    # -----------------------------------------------------
    # - Network / RP
    # -----------------------------------------------------
    ip = get_str(cfg, "ip", "169.254.7.90")
    timeout_s = get_float(cfg, "timeout_s", 5.0)

    # -----------------------------------------------------
    # - Generator settings (SCPI)
    # -----------------------------------------------------
    signal_type = get_str(cfg, "signal_type", "SAWU")
    signal_freq_hz = get_float(cfg, "signal_frequency_hz", 1000.0)
    amplitude_v = get_float(cfg, "amplitude_v", 0.1)
    offset_v = get_float(cfg, "offset_v", 0.2)

    # -----------------------------------------------------
    # - Continuous streaming settings
    # -----------------------------------------------------
    fs_req = get_float(cfg, "sampling_frequency_hz", 122070.3125)
    adc_base_hz = get_float(cfg, "adc_base_rate_hz", 250_000_000.0)

    stream_port = get_int(cfg, "stream_port", 8900)
    max_tcp_MB_s = get_float(cfg, "max_tcp_rate_MB_s", 62.5)
    link_safety = get_float(cfg, "link_safety", 0.75)

    adc_voltage_range = get_float(cfg, "adc_voltage_range", 1.0)

    enable_plot = get_bool01(cfg, "enable_plot", True)
    plot_window_s = get_float(cfg, "plot_window_s", 2.0)
    plot_update_hz = get_float(cfg, "plot_update_hz", 20.0)

    print_every_s = get_float(cfg, "print_every_s", 1.0)

    # -----------------------------------------------------
    # - SSH autostart options
    # -----------------------------------------------------
    enable_ssh_autostart = get_bool01(cfg, "enable_ssh_autostart", True)
    ssh_user = get_str(cfg, "ssh_user", "root")
    ssh_port = get_int(cfg, "ssh_port", 22)
    ssh_key_path = get_str(cfg, "ssh_key_path", "")
    ssh_connect_timeout_s = get_float(cfg, "ssh_connect_timeout_s", 5.0)

    # Recommended default: load FPGA overlay then start streaming-server in background
    ssh_start_stream_cmd = get_str(
        cfg,
        "ssh_start_stream_cmd",
        "bash -lc 'overlay.sh stream_app && streaming-server --background'",
    )

    # Optional stop command
    stop_stream_on_exit = get_bool01(cfg, "stop_stream_on_exit", True)
    ssh_stop_stream_cmd = get_str(
        cfg,
        "ssh_stop_stream_cmd",
        "bash -lc 'killall streaming-server 2>/dev/null || true'",
    )

    # -----------------------------------------------------
    # - Decide effective Fs considering link budget
    # -----------------------------------------------------
    dec, fs_eff = sampling_to_decimation(fs_req, adc_base_hz)

    # Highest resolution transfer case: int16 -> 2 bytes/sample (per channel)
    bytes_per_sample = 2

    fs_link_clamped = clamp_rate_for_link(fs_eff, bytes_per_sample, max_tcp_MB_s, link_safety)
    if fs_link_clamped < fs_eff:
        dec, fs_eff = sampling_to_decimation(fs_link_clamped, adc_base_hz)

    dt = 1.0 / fs_eff

    # -----------------------------------------------------
    # - Connect SCPI (generator only)
    # -----------------------------------------------------
    rp = scpi.scpi(ip, timeout=timeout_s)

    # -----------------------------------------------------
    # - Optional plotting
    # -----------------------------------------------------
    plots = LivePlotIN1(plot_window_s) if enable_plot else None

    # -----------------------------------------------------
    # - Streaming socket client
    # -----------------------------------------------------
    stream = RPStreamClient(ip, stream_port, sock_timeout_s=max(1.0, min(5.0, timeout_s)))

    try:
        # Identify device
        rp.tx_txt("*IDN?")
        idn = rp.rx_txt().strip()

        print("\n- === Red Pitaya run (SCPI GEN + TCP STREAM IN1 + SSH autostart) ===")
        print(f"- IDN: {idn}")
        print(f"- RP IP: {ip}")
        print("- Generator (OUT1):")
        print(f"  - type={signal_type}")
        print(f"  - frequency={signal_freq_hz} Hz")
        print(f"  - amplitude={amplitude_v} V")
        print(f"  - offset={offset_v} V")
        print("- Streaming (IN1):")
        print(f"  - requested Fs={fs_req} Hz")
        print(f"  - adc_base_rate_hz={adc_base_hz:.0f} Hz")
        print(f"  - chosen decimation={dec}")
        print(f"  - effective Fs={fs_eff:.6f} Hz")
        print(f"  - link budget: max_tcp={max_tcp_MB_s} MB/s, safety={link_safety}")
        print(f"  - bytes/sample={bytes_per_sample} (16-bit)")
        print(f"  - approx throughput={fs_eff*bytes_per_sample/1e6:.2f} MB/s")
        print(f"  - ADC conversion: ±{ADC_MAX_COUNT} counts = ±{adc_voltage_range} V")
        print(f"- Plotting: {'ON' if enable_plot else 'OFF'} (window={plot_window_s}s)")
        print("- IN1 is NOT saved to disk.")
        print("- Press Ctrl-C to stop.\n")

        # -------------------------------------------------
        # - Start streaming-server via SSH (optional)
        # -------------------------------------------------
        if enable_ssh_autostart:
            print("- SSH: starting streaming server...", flush=True)
            run_ssh_command(
                host=ip,
                user=ssh_user,
                port=ssh_port,
                cmd=ssh_start_stream_cmd,
                key_path=ssh_key_path,
                connect_timeout_s=ssh_connect_timeout_s,
            )
            # Give server a moment to bind the port
            time.sleep(0.5)
            print("- SSH: streaming server start command sent.", flush=True)

        # -------------------------------------------------
        # - Reset generator baseline then start it
        # -------------------------------------------------
        rp.tx_txt("GEN:RST")
        time.sleep(0.1)

        rp.tx_txt("SOUR1:BURS:STAT CONTINUOUS")
        rp.tx_txt("SOUR1:TRig:SOUR INTERNAL")
        rp.tx_txt(f"SOUR1:FUNC {signal_type}")
        rp.tx_txt(f"SOUR1:FREQ:FIX {signal_freq_hz}")
        rp.tx_txt(f"SOUR1:VOLT {amplitude_v}")
        rp.tx_txt(f"SOUR1:VOLT:OFFS {offset_v}")
        rp.tx_txt("OUTPUT1:STATE ON")
        rp.tx_txt("SOUR1:TRig:INT")

        # -------------------------------------------------
        # - Connect streaming socket
        # -------------------------------------------------
        stream.connect()

        # -------------------------------------------------
        # - Continuous receive loop
        # -------------------------------------------------
        t_last_print = time.time()
        t_last_plot = time.time()
        plot_period = 1.0 / max(1e-6, plot_update_hz)

        # Local timebase based on streaming index
        sample_index0: Optional[int] = None
        wall0: Optional[float] = None

        while True:
            meta, ch1, _ch2 = stream.read_packet()

            # bytes -> int16 (little-endian is typical for streaming clients)
            # If values look wrong, change dtype to '>i2'
            y_raw = np.frombuffer(ch1, dtype="<i2").astype(np.int16, copy=False)
            y_v = raw_to_volts(y_raw, adc_voltage_range)

            pkt_index = meta["index"]
            n = y_v.size

            if sample_index0 is None:
                sample_index0 = pkt_index
                wall0 = time.time()

            # Interpret pkt_index as a sample counter (common)
            start_sample = pkt_index
            t_rel0 = (start_sample - sample_index0) * dt
            t_rel = t_rel0 + (np.arange(n, dtype=np.float64) * dt)

            # Map to wallclock for plots (so first sample ~= "now")
            assert wall0 is not None
            t_abs = wall0 + t_rel

            now = time.time()

            # Plot throttled
            if plots and (now - t_last_plot) >= plot_period:
                plots.add_samples(t_abs, y_v)
                plots.draw()
                t_last_plot = now

            # Console stats throttled
            if (now - t_last_print) >= print_every_s:
                ystd = float(y_v.std())
                ymin = float(y_v.min())
                ymax = float(y_v.max())

                # crude receive rate estimate for this print interval
                dt_print = max(1e-6, (now - t_last_print))
                approx_mbps = (n * bytes_per_sample) / dt_print / 1e6

                print(
                    f"- IN1 stream: idx={pkt_index} samples={n} "
                    f"std={ystd:.4f} V (min={ymin:.4f}, max={ymax:.4f}) "
                    f"| meta_rate={meta['rate']} | rx~{approx_mbps:.1f} MB/s",
                    flush=True,
                )
                t_last_print = now

    except KeyboardInterrupt:
        print("\n- Ctrl-C received; stopping.", flush=True)

    finally:
        # Stop generator outputs
        try:
            print("- Cleanup: turning OFF outputs...", flush=True)
            rp.tx_txt("OUTPUT1:STATE OFF")
            rp.tx_txt("OUTPUT2:STATE OFF")
        except Exception:
            pass

        # Close stream socket
        try:
            stream.close()
        except Exception:
            pass

        # Stop streaming server over SSH (optional)
        if enable_ssh_autostart and stop_stream_on_exit:
            try:
                print("- SSH: stopping streaming server...", flush=True)
                run_ssh_command(
                    host=ip,
                    user=ssh_user,
                    port=ssh_port,
                    cmd=ssh_stop_stream_cmd,
                    key_path=ssh_key_path,
                    connect_timeout_s=ssh_connect_timeout_s,
                )
                print("- SSH: stop command sent.", flush=True)
            except Exception as e:
                print(f"- [WARN] SSH stop failed: {e}", flush=True)

        # Close SCPI
        try:
            rp.close()
        except Exception:
            pass

        print("- Done.", flush=True)


if __name__ == "__main__":
    main("config_continuous_flashing.txt")
