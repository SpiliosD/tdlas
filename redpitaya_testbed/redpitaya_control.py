from __future__ import annotations

# - Standard library
import csv
import os
import time
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple

# - Third-party
import numpy as np

# - Red Pitaya SCPI helper (you already have this)
import redpitaya_scpi as scpi

# =========================================================
# - ADC CONVERSION CONSTANTS
# =========================================================
# Red Pitaya STEMlab 125-14 specifications:
# - ADC resolution: 14 bits
# - ADC range: ±1V (when using HV jumpers) or ±20V (when using LV jumpers)
# - For standard configuration (HV): ±1V corresponds to ±8192 counts (2^13)
ADC_BITS = 14
ADC_MAX_COUNT = 2 ** (ADC_BITS - 1)  # 8192


def raw_to_volts(raw_counts: np.ndarray, voltage_range: float) -> np.ndarray:
    """
    Convert raw ADC counts to voltage.

    For Red Pitaya STEMlab 125-14:
    - ADC range: ±8192 counts = ±1V (HV jumper) or ±20V (LV jumper)
    - Conversion: V = (raw_count / 8192) * voltage_range

    Args:
        raw_counts: Raw ADC values from Red Pitaya
        voltage_range: Voltage range setting (1.0 for HV, 20.0 for LV)

    Returns:
        Voltage values in Volts
    """
    return (raw_counts.astype(np.float64) / ADC_MAX_COUNT) * voltage_range


# =========================================================
# - CONFIG FILE PARSER
# =========================================================
# - We use a simple "key = value" format.
# - Lines starting with '#' are comments and ignored.
# - Empty lines are ignored.
# - Values are read as strings and converted with helper functions.
def load_config_txt(path: str) -> Dict[str, str]:
    """
    - Reads config.txt and returns a dictionary of {key: value_string}.

    - Supported line format:
        key = value

    - Examples:
        signal_type = SAWU
        sampling_frequency_hz = 488281.25
        enable_plot = 1
    """
    cfg: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for raw in f.readlines():
            line = raw.strip()

            # - Skip blank lines and comment lines
            if not line or line.startswith("#"):
                continue

            # - Skip lines not containing '='
            if "=" not in line:
                continue

            # - Split at first '=' only
            k, v = line.split("=", 1)
            cfg[k.strip()] = v.strip()

    return cfg


# - Typed getters with defaults:
# - These prevent crashes if a key is missing or malformed.
def get_str(cfg: Dict[str, str], key: str, default: str) -> str:
    return cfg.get(key, default)


def get_float(cfg: Dict[str, str], key: str, default: float) -> float:
    try:
        return float(cfg.get(key, default))
    except Exception:
        return default


def get_int(cfg: Dict[str, str], key: str, default: int) -> int:
    try:
        # - Allow "100.0" to be interpreted as 100
        return int(float(cfg.get(key, default)))
    except Exception:
        return default


def get_bool01(cfg: Dict[str, str], key: str, default: bool) -> bool:
    """
    - Accepts:
        1 / 0
        true / false
        yes / no
        on / off
    """
    v = cfg.get(key, None)
    if v is None:
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "on")


# =========================================================
# - ACQUISITION HELPERS
# =========================================================

def parse_scpi_binblock(payload: bytes) -> bytes:
    """
    - SCPI binary replies often use "definite-length blocks":
        #<ndigits><length><binary-data>

    - Example:
        b'#512345' + <12345 bytes of data>

    - This function strips the header and returns only the raw binary data.
    """
    if not payload:
        return payload

    # - If it's not a binblock header, return as-is
    if payload[:1] != b"#":
        return payload

    # - The next byte tells how many digits describe the length
    if len(payload) < 3:
        return payload

    nd_chr = payload[1:2]
    if not (b"0" <= nd_chr <= b"9"):
        return payload

    nd = int(nd_chr.decode("ascii"))

    # - nd == 0 means "no length field" (rare)
    if nd == 0:
        return payload[2:]

    header_len = 2 + nd
    if len(payload) < header_len:
        return payload

    # - Length field is ASCII digits, describing the data length
    data_len = int(payload[2:header_len].decode("ascii"))

    start = header_len
    end = start + data_len

    # - If incomplete, return what's available
    if len(payload) < end:
        return payload[start:]

    return payload[start:end]


def read_rf_bin_raw_int16(rp: scpi.scpi, n_samples: int) -> np.ndarray:
    """
    - Reads IN1 acquisition buffer using:
        ACQ:DATA:FORMAT BIN
        ACQ:DATA:UNITS RAW

    - Returns:
        numpy array of int16 values (ADC counts)

    - Notes:
        • RAW units are NOT volts; they are ADC counts.
        • Endianness can vary by firmware.
          We use big-endian (>i2) because it is common for Red Pitaya SCPI examples.
          If you see nonsense values, try changing to '<i2'.
    """
    rp.tx_txt("ACQ:SOUR1:DATA?")
    payload = rp.rx_arb()
    if payload is None:
        raise RuntimeError("rx_arb returned None (no payload).")

    payload = parse_scpi_binblock(payload)

    data = np.frombuffer(payload, dtype=">i2")

    # - Ensure correct size
    if data.size < n_samples:
        raise RuntimeError(f"Short payload: got {data.size}, expected {n_samples}.")
    if data.size > n_samples:
        data = data[:n_samples]

    return data.astype(np.int16, copy=False)


def wait_trig_td(rp: scpi.scpi, timeout_s: float) -> str:
    """
    - Wait until acquisition is complete.
    - Red Pitaya returns trigger status via:
        ACQ:TRIG:STAT?

    - We wait until it becomes "TD" (triggered + data ready).
    - Timeout is critical so we don't hang forever.
    """
    t0 = time.time()
    last = ""
    while True:
        rp.tx_txt("ACQ:TRIG:STAT?")
        last = rp.rx_txt().strip()

        # - TD = trigger detected + buffer ready
        if last == "TD":
            return last

        # - timeout
        if time.time() - t0 > timeout_s:
            return last

        time.sleep(0.002)


def read_ain(rp: scpi.scpi, name: str) -> float:
    """
    - Reads slow ADC pins on E2 connector:
        AIN0, AIN1

    - Returns voltage as float.

    - If nothing connected:
        • The pins float and you will read drifting/random-ish values.
    """
    rp.tx_txt(f"ANALOG:PIN? {name}")
    return float(rp.rx_txt().strip())


# =========================================================
# - UNIFIED PLOTTING (3 subplots in 1 figure)
# =========================================================
class LivePlotsUnified:
    """
    - Single figure with three vertically stacked subplots:
        • IN1 (Voltage in V) - full accumulated history
        • AIN0 (V) - full accumulated history
        • AIN1 (V) - full accumulated history

    - Relative time:
        • All plots show time relative to acquisition start (t=0)
        • Actual timestamps are preserved in saved data

    - All channels accumulate data continuously
    """

    def __init__(self):
        import matplotlib.pyplot as plt

        self.plt = plt
        plt.ion()

        # - Store initial timestamp for relative time calculation
        self.t_start: Optional[float] = None

        # - Full history buffers for all channels
        self.t_in1: List[float] = []
        self.y_in1: List[float] = []
        self.t_ain0: List[float] = []
        self.y_ain0: List[float] = []
        self.t_ain1: List[float] = []
        self.y_ain1: List[float] = []

        # - Create single figure with 3 vertically stacked subplots
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(3, 1, figsize=(10, 8))
        self.fig.suptitle("Red Pitaya Real-Time Acquisition", fontsize=14, fontweight='bold')

        # - IN1 subplot (full history, in Volts)
        self.line1, = self.ax1.plot([], [], 'b-', linewidth=0.5)
        self.ax1.set_title("IN1 (High-speed ADC)")
        self.ax1.set_xlabel("Relative Time (s)")
        self.ax1.set_ylabel("Voltage (V)")
        self.ax1.grid(True, alpha=0.3)

        # - AIN0 subplot
        self.line2, = self.ax2.plot([], [], 'g-o', markersize=3)
        self.ax2.set_title("AIN0 (Slow ADC)")
        self.ax2.set_xlabel("Relative Time (s)")
        self.ax2.set_ylabel("Voltage (V)")
        self.ax2.grid(True, alpha=0.3)

        # - AIN1 subplot
        self.line3, = self.ax3.plot([], [], 'r-o', markersize=3)
        self.ax3.set_title("AIN1 (Slow ADC)")
        self.ax3.set_xlabel("Relative Time (s)")
        self.ax3.set_ylabel("Voltage (V)")
        self.ax3.grid(True, alpha=0.3)

        # - Adjust layout to prevent overlap
        self.fig.tight_layout()

    def add_in1_block(self, t_block: np.ndarray, y_block: np.ndarray) -> None:
        """
        Add IN1 data block (accumulates all data)

        Args:
            t_block: Absolute timestamps for this block
            y_block: Voltage values (already in Volts)
        """
        # - Set start time on first call
        if self.t_start is None:
            self.t_start = t_block[0]

        # - Convert to relative time and accumulate
        t_relative = (t_block - self.t_start).tolist()
        self.t_in1.extend(t_relative)
        self.y_in1.extend(y_block.tolist())

    def add_ain0(self, t: float, v: float) -> None:
        """Add AIN0 data point (accumulates all data)"""
        if self.t_start is None:
            self.t_start = t

        t_relative = t - self.t_start
        self.t_ain0.append(t_relative)
        self.y_ain0.append(v)

    def add_ain1(self, t: float, v: float) -> None:
        """Add AIN1 data point (accumulates all data)"""
        if self.t_start is None:
            self.t_start = t

        t_relative = t - self.t_start
        self.t_ain1.append(t_relative)
        self.y_ain1.append(v)

    def draw(self) -> None:
        # - IN1 update (full accumulated history)
        self.line1.set_data(self.t_in1, self.y_in1)
        self.ax1.relim()
        self.ax1.autoscale_view()

        # - AIN0 update (full accumulated history)
        self.line2.set_data(self.t_ain0, self.y_ain0)
        self.ax2.relim()
        self.ax2.autoscale_view()

        # - AIN1 update (full accumulated history)
        self.line3.set_data(self.t_ain1, self.y_ain1)
        self.ax3.relim()
        self.ax3.autoscale_view()

        # - Redraw the single figure
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        self.plt.pause(0.001)


# =========================================================
# - SAMPLING FREQUENCY -> DECIMATION
# =========================================================
def sampling_to_decimation(fs_req: float) -> Tuple[int, float]:
    """
    - Red Pitaya ADC base rate = 125 MHz
    - Effective rate:
        Fs_eff = 125e6 / decimation

    - We choose:
        decimation = round(125e6 / fs_req)

    - Returns:
        (decimation, Fs_eff)
    """
    base = 125_000_000.0
    if fs_req <= 0:
        raise ValueError("sampling_frequency_hz must be > 0")

    dec = int(round(base / fs_req))
    if dec < 1:
        dec = 1

    fs_eff = base / dec
    return dec, fs_eff


# =========================================================
# - SAVING HELPERS
# =========================================================
def write_csv_header(writer: csv.writer) -> None:
    # - Exactly 2 columns as requested
    writer.writerow(["timestamp_s", "value"])


def save_block_to_npy_two_cols(path: str, t: np.ndarray, y: np.ndarray) -> None:
    """
    - NPY is not good for "append", so we save per-block files:
        prefix_in1_000001.npy, etc.

    - Format saved:
        2-column array [timestamp_s, value]
    """
    arr = np.column_stack((t.astype(np.float64), y.astype(np.float64)))
    np.save(path, arr)


# =========================================================
# - MAIN
# =========================================================
def main(config_path: str = "config.txt") -> None:
    # - Load config text file
    txt = load_config_txt(config_path)

    # -----------------------------------------------------
    # - REQUIRED USER INPUTS (as you requested)
    # -----------------------------------------------------
    # - Signal generation inputs
    ip = get_str(txt, "ip", "169.254.9.103")

    signal_type = get_str(txt, "signal_type", "SAWU")
    signal_freq_hz = get_float(txt, "signal_frequency_hz", 100.0)
    amplitude_v = get_float(txt, "amplitude_v", 0.5)
    offset_v = get_float(txt, "offset_v", 0.5)

    # - Signal recording input
    fs_req = get_float(txt, "sampling_frequency_hz", 488_281.25)

    # - ADC conversion setting
    adc_voltage_range = get_float(txt, "adc_voltage_range", 1.0)

    # - Saving input
    file_type = get_str(txt, "file_type", "csv").lower().strip()

    # -----------------------------------------------------
    # - OPTIONAL SETTINGS
    # -----------------------------------------------------
    out_dir = get_str(txt, "out_dir", "./rp_capture")
    prefix = get_str(txt, "prefix", "gen_and_noise")
    enable_plot = get_bool01(txt, "enable_plot", True)

    timeout_s = get_float(txt, "timeout_s", 5.0)
    n_samples = get_int(txt, "n_samples", 16384)
    flush_every_rows = get_int(txt, "flush_every_rows", 20000)
    plot_update_every_blocks = get_int(txt, "plot_update_every_blocks", 1)
    print_every_blocks = get_int(txt, "print_every_blocks", 1)

    max_blocks_raw = get_int(txt, "max_blocks", 0)
    max_blocks: Optional[int] = None if max_blocks_raw == 0 else max_blocks_raw

    # -----------------------------------------------------
    # - Convert requested sampling frequency to decimation
    # -----------------------------------------------------
    decimation, fs_eff = sampling_to_decimation(fs_req)
    dt = 1.0 / fs_eff

    # -----------------------------------------------------
    # - Setup filesystem
    # -----------------------------------------------------
    os.makedirs(out_dir, exist_ok=True)

    # - File targets
    # - For CSV: one growing file per channel
    # - For NPY: these are base names; we save per-block files
    in1_path = os.path.join(out_dir, f"{prefix}_in1.{file_type}")
    ain0_path = os.path.join(out_dir, f"{prefix}_ain0.{file_type}")
    ain1_path = os.path.join(out_dir, f"{prefix}_ain1.{file_type}")

    # -----------------------------------------------------
    # - Connect to Red Pitaya
    # -----------------------------------------------------
    rp = scpi.scpi(ip, timeout=timeout_s)

    # - Optional plotting (now unified in single figure)
    plots = LivePlotsUnified() if enable_plot else None

    # -----------------------------------------------------
    # - Setup saving
    # -----------------------------------------------------
    if file_type == "csv":
        in1_f = open(in1_path, "w", newline="", encoding="utf-8")
        a0_f = open(ain0_path, "w", newline="", encoding="utf-8")
        a1_f = open(ain1_path, "w", newline="", encoding="utf-8")

        in1_w = csv.writer(in1_f)
        a0_w = csv.writer(a0_f)
        a1_w = csv.writer(a1_f)

        write_csv_header(in1_w)
        write_csv_header(a0_w)
        write_csv_header(a1_w)

        in1_rows_since_flush = 0
        a0_rows_since_flush = 0
        a1_rows_since_flush = 0

    elif file_type == "npy":
        # - NPY is per-block; no open file handle needed
        in1_f = a0_f = a1_f = None
        in1_w = a0_w = a1_w = None
        in1_rows_since_flush = a0_rows_since_flush = a1_rows_since_flush = 0

    else:
        raise ValueError("file_type must be 'csv' or 'npy'")

    try:
        # -------------------------------------------------
        # - Identify device (sanity check)
        # -------------------------------------------------
        rp.tx_txt("*IDN?")
        idn = rp.rx_txt().strip()

        print("\n- === Red Pitaya run ===")
        print(f"- IDN: {idn}")
        print(f"- IP: {ip}")
        print(f"- Generator settings:")
        print(f"  - type={signal_type}")
        print(f"  - frequency={signal_freq_hz} Hz")
        print(f"  - amplitude={amplitude_v} V")
        print(f"  - offset={offset_v} V")
        print(f"- Recording settings:")
        print(f"  - requested Fs={fs_req} Hz")
        print(f"  - chosen decimation={decimation}")
        print(f"  - effective Fs={fs_eff:.6f} Hz")
        print(f"  - ADC conversion: ±{ADC_MAX_COUNT} counts = ±{adc_voltage_range} V")
        print(f"- Saving: file_type={file_type}")
        print(f"  - IN1 : {os.path.abspath(in1_path)}")
        print(f"  - AIN0: {os.path.abspath(ain0_path)}")
        print(f"  - AIN1: {os.path.abspath(ain1_path)}")
        print(f"- Plotting: {'ON (unified figure)' if enable_plot else 'OFF'}")
        print("- Press Ctrl-C to stop.\n")

        # -------------------------------------------------
        # - Reset generator + acquisition (clean baseline)
        # -------------------------------------------------
        rp.tx_txt("GEN:RST")
        rp.tx_txt("ACQ:RST")
        time.sleep(0.1)

        # -------------------------------------------------
        # - Configure generator (OUT1) and start it
        # -------------------------------------------------
        rp.tx_txt("SOUR1:BURS:STAT CONTINUOUS")  # - not burst (no single pulse)
        rp.tx_txt("SOUR1:TRig:SOUR INTERNAL")  # - free running source

        rp.tx_txt(f"SOUR1:FUNC {signal_type}")
        rp.tx_txt(f"SOUR1:FREQ:FIX {signal_freq_hz}")
        rp.tx_txt(f"SOUR1:VOLT {amplitude_v}")
        rp.tx_txt(f"SOUR1:VOLT:OFFS {offset_v}")

        rp.tx_txt("OUTPUT1:STATE ON")  # - enable physical output
        rp.tx_txt("SOUR1:TRig:INT")  # - start generation

        # -------------------------------------------------
        # - Configure acquisition for IN1 noise recording
        # -------------------------------------------------
        rp.tx_txt(f"ACQ:DEC:Factor {decimation}")
        rp.tx_txt("ACQ:DATA:FORMAT BIN")
        rp.tx_txt("ACQ:DATA:UNITS RAW")

        # - NOW trigger does not require external input
        # - IMPORTANT: do not query ACQ:TRIG? (can timeout)
        rp.tx_txt("ACQ:TRIG NOW")

        # -------------------------------------------------
        # - Loop: acquire IN1 block + read AIN0/AIN1 + save + plot
        # -------------------------------------------------
        blocks = 0
        while True:
            if max_blocks is not None and blocks >= max_blocks:
                print("- max_blocks reached; stopping.")
                break

            # - Arm acquisition buffer
            rp.tx_txt("ACQ:START")

            # - Fire immediate trigger (noise mode)
            rp.tx_txt("ACQ:TRIG NOW")

            # - Wait for data ready
            st = wait_trig_td(rp, timeout_s=2.0)
            if st != "TD":
                print(f"- [WARN] Block {blocks}: trigger timeout (state={st}). Try lower Fs.")
                continue

            # - Timestamp block start
            t0 = time.time()

            # - Read IN1 samples (raw ADC counts)
            y_raw = read_rf_bin_raw_int16(rp, n_samples)

            # - Convert raw counts to Volts using config voltage range
            y_volts = raw_to_volts(y_raw, adc_voltage_range)

            # - Build per-sample timestamps
            t = t0 + (np.arange(n_samples, dtype=np.float64) * dt)

            # - Read AIN0/AIN1 once per block
            v0 = read_ain(rp, "AIN0")
            v1 = read_ain(rp, "AIN1")

            # -------------------------
            # - Save data (IN1 saved as continuous stream, not blocks)
            # -------------------------
            if file_type == "csv":
                # - IN1: Write each sample individually (continuous stream)
                for t_sample, v_sample in zip(t.tolist(), y_volts.tolist()):
                    in1_w.writerow([t_sample, v_sample])
                in1_rows_since_flush += n_samples

                # - AIN0/AIN1 (one row each)
                a0_w.writerow([t0, v0])
                a1_w.writerow([t0, v1])
                a0_rows_since_flush += 1
                a1_rows_since_flush += 1

                # - Flush periodically
                if in1_rows_since_flush >= flush_every_rows:
                    in1_f.flush()
                    in1_rows_since_flush = 0
                if a0_rows_since_flush >= flush_every_rows:
                    a0_f.flush()
                    a0_rows_since_flush = 0
                if a1_rows_since_flush >= flush_every_rows:
                    a1_f.flush()
                    a1_rows_since_flush = 0

            else:
                # - NPY: Save each sample individually like analog inputs
                # - For IN1, we could either:
                #   (a) Save one file per sample (inefficient, many files)
                #   (b) Save per-block but as 2-column array (efficient, fewer files)
                # - We'll use option (b) for NPY format
                in1_blk = os.path.join(out_dir, f"{prefix}_in1_{blocks:06d}.npy")
                a0_blk = os.path.join(out_dir, f"{prefix}_ain0_{blocks:06d}.npy")
                a1_blk = os.path.join(out_dir, f"{prefix}_ain1_{blocks:06d}.npy")

                # - Save 2-column arrays (timestamp, value in Volts)
                save_block_to_npy_two_cols(in1_blk, t, y_volts)
                save_block_to_npy_two_cols(a0_blk, np.array([t0]), np.array([v0]))
                save_block_to_npy_two_cols(a1_blk, np.array([t0]), np.array([v1]))

            # -------------------------
            # - Plot (keeps all data, now IN1 is in Volts)
            # -------------------------
            if plots and (blocks % plot_update_every_blocks == 0):
                plots.add_in1_block(t, y_volts)
                plots.add_ain0(t0, v0)
                plots.add_ain1(t0, v1)
                plots.draw()

            # -------------------------
            # - Console progress (now shows voltage statistics)
            # -------------------------
            if print_every_blocks and (blocks % print_every_blocks == 0):
                print(
                    f"- [BLOCK {blocks:06d}] IN1 std={float(y_volts.std()):.4f} V "
                    f"(min={float(y_volts.min()):.4f} V, max={float(y_volts.max()):.4f} V) | "
                    f"AIN0={v0:.3f} V | AIN1={v1:.3f} V",
                    flush=True
                )

            blocks += 1

    except KeyboardInterrupt:
        print("\n- Ctrl-C received; stopping.", flush=True)

    finally:
        # - Always turn outputs off (good safety)
        try:
            print("- Cleanup: turning OFF outputs...", flush=True)
            rp.tx_txt("OUTPUT1:STATE OFF")
            rp.tx_txt("OUTPUT2:STATE OFF")
        except Exception:
            pass

        try:
            rp.close()
        except Exception:
            pass

        # - Close CSV files if used
        if file_type == "csv":
            try:
                in1_f.flush()
                a0_f.flush()
                a1_f.flush()
            except Exception:
                pass

            in1_f.close()
            a0_f.close()
            a1_f.close()

        print("- Done.", flush=True)


if __name__ == "__main__":
    # - Run using config.txt settings
    main("config.txt")