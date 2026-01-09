import time
import redpitaya_scpi as scpi

def out1_triangle_0_to_0p2V_100Hz_continuous_verbose(ip="169.254.7.90"):
    rp = scpi.scpi(ip)

    def q(cmd: str) -> str:
        rp.tx_txt(cmd)
        return rp.rx_txt().strip()

    def send(cmd: str, query: str | None = None):
        print(f"[SEND] {cmd}")
        rp.tx_txt(cmd)
        if query:
            print(f"[ASK ] {query}")
            rp.tx_txt(query)
            ans = rp.rx_txt().strip()
            print(f"[RESP] {ans}")
            return ans
        return None

    try:
        print("\n=== OUT1 continuous triangle 0..0.2 V @ 100 Hz ===")
        print("[ASK ] *IDN?")
        print("[RESP]", q("*IDN?"))

        # Reset generator (good practice)
        send("GEN:RST")
        time.sleep(0.1)

        # Ensure NOT in burst mode (continuous output)
        send("SOUR1:BURS:STAT OFF", "SOUR1:BURS:STAT?")

        # If supported, ensure free-run / internal trigger source (not required on all builds)
        try:
            send("SOUR1:TRIG:SOUR INTERNAL", "SOUR1:TRIG:SOUR?")
        except Exception:
            print("[WARN] SOUR1:TRIG:SOUR not supported (continuing).")

        # Configure waveform
        send("SOUR1:FUNC SAWU", "SOUR1:FUNC?")
        send("SOUR1:FREQ:FIX 10000", "SOUR1:FREQ:FIX?")

        # 0..0.2 V => amp=0.1, offset=0.1
        send("SOUR1:VOLT 0.1", "SOUR1:VOLT?")
        send("SOUR1:VOLT:OFFS 0.1", "SOUR1:VOLT:OFFS?")

        # Enable output
        send("OUTPUT1:STATE ON", "OUTPUT1:STATE?")

        # --- Start generation NOW (this is the key step) ---
        # Docs command: SOUR<n>:TRig:INT  (starts the waveform from the beginning) :contentReference[oaicite:2]{index=2}
        send("SOUR1:TRig:INT")

        print("\n[OK] You should now see a CONTINUOUS triangle on OUT1.")
        print("     Scope tips: DC coupling, 2 ms/div or 5 ms/div, 50–100 mV/div.\n")

        # Keep the connection open briefly while you observe
        time.sleep(10)

    finally:
        try:
            send("OUTPUT1:STATE OFF", "OUTPUT1:STATE?")
        except Exception:
            pass
        try:
            rp.close()
        except Exception:
            pass
        print("[DONE] Output OFF and SCPI closed.")


if __name__== '__main__':
    out1_triangle_0_to_0p2V_100Hz_continuous_verbose()