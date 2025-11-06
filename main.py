import argparse
import numpy as np
from pythonosc.udp_client import SimpleUDPClient

from config import (
    FORCE_MOCK, ORACLE_WHEN_MOCK,
    FS, N_WIN, N_HOP, CH_NAMES,
    OSC_IP, OSC_PORT, OSC_PARAMS,
    ADDR_A, ADDR_V, ADDR_ART,
    BASELINE_SEC, EMA_ALPHA,
    ADAPT_ALPHA, TARGET, DEFAULT_OPTIMIZER
)
from util import bandpass_filter, ema, distance, squash_tanh
from features import compute_features
from plotting import LivePlot
from sources import UnicornSource, MockEEGSource, mock_av_response, HAVE_UNICORN
from optimizers import get_optimizer


def run(optimizer_name: str):
    client = SimpleUDPClient(OSC_IP, OSC_PORT)

    # Decide data source mode
    if not FORCE_MOCK and HAVE_UNICORN:
        try:
            src = UnicornSource()
            mode = "EEG_REAL"
        except Exception as e:
            print(f"Unicorn init failed ({e}). Falling back to mock.")
            src = None
            mode = "MOCK_ORACLE" if ORACLE_WHEN_MOCK else "MOCK_EEG"
    else:
        src = None
        mode = "MOCK_ORACLE" if ORACLE_WHEN_MOCK else "MOCK_EEG"

    # Initialize music params at center and send
    current_params = {name: 0.5 for name in OSC_PARAMS}
    for k, v in current_params.items():
        client.send_message(f"/music/{k}", float(v))

    if mode == "MOCK_EEG":
        src = MockEEGSource()
        src.set_current_params(current_params)

    A_out = 0.5
    V_out = 0.5
    art = 0

    # Oracle initialization (no EEG)
    if mode == "MOCK_ORACLE":
        A_out, V_out = mock_av_response(current_params)
        client.send_message(ADDR_A, float(A_out))
        client.send_message(ADDR_V, float(V_out))
        client.send_message(ADDR_ART, 0)

    # Baseline for EEG modes (centering/scaling)
    if mode in ("EEG_REAL", "MOCK_EEG"):
        print("Collecting baseline...")
        buf = np.zeros((len(CH_NAMES), N_WIN))
        write_idx = 0
        A_hist, V_hist = [], []
        baseline_samples = int(BASELINE_SEC * FS)
        samples_collected, hop_accum = 0, 0

        while samples_collected < baseline_samples:
            chunk = src.pull_chunk(min(N_HOP, baseline_samples - samples_collected))
            for i in range(chunk.shape[1]):
                buf[:, write_idx % N_WIN] = chunk[:, i]
                write_idx += 1
                samples_collected += 1
                hop_accum += 1
                if hop_accum == N_HOP:
                    idx = np.arange(write_idx - N_WIN, write_idx) % N_WIN
                    win = buf[:, idx]
                    # win: np.ndarray [n_channels, n_samples]
                    win_filt = bandpass_filter(win, FS, 1, 20)
                    a_raw, v_raw, art = compute_features(win_filt)
                    A_hist.append(a_raw)
                    V_hist.append(v_raw)
                    hop_accum = 0

        A_center = float(np.mean(A_hist))
        V_center = float(np.mean(V_hist))
        # robust scale (MAD around median)
        A_mad = float(np.median(np.abs(A_hist - np.median(A_hist)))) or 1e-3
        V_mad = float(np.median(np.abs(V_hist - np.median(V_hist)))) or 1e-3
        ema_A, ema_V = float(A_hist[-1]), float(V_hist[-1])
        print("Baseline ready. Center A =", round(A_center, 4), "V =", round(V_center, 4))
    else:
        # dummy initializers for oracle mode
        buf = None
        write_idx = 0
        A_center = V_center = 0.0
        A_mad = V_mad = 1.0
        ema_A = ema_V = 0.0

    # Plotting and optimizer
    live = LivePlot(A_star=TARGET["A_star"], V_star=TARGET["V_star"])
    opt = get_optimizer(optimizer_name)

    # Share targets with optimizers that use them for reward shaping (e.g., BO)
    setattr(opt, "A_star", TARGET["A_star"])
    setattr(opt, "V_star", TARGET["V_star"])

    # Per-phase sample buffers (for 3s averaging @ 0.25s hop)
    plus_A, plus_V = [], []
    minus_A, minus_V = [], []
    settle_hops = 2          # ignore ~0.5s after param change
    phase_hop_index = 0

    hop_counter = 0
    epoch_hops = 12          # ~3s per epoch
    phase = 0                # 0=idle, 1=plus, 2=minus (SPSA uses both, BO only uses 1)

    try:
        while True:
            # ==== Ingest A/V ====
            if mode != "MOCK_ORACLE":
                hop = src.pull_chunk(N_HOP)
                for i in range(hop.shape[1]):
                    buf[:, write_idx % N_WIN] = hop[:, i]
                    write_idx += 1

                idx = np.arange(write_idx - N_WIN, write_idx) % N_WIN
                win = buf[:, idx]
                a_raw, v_raw, art = compute_features(win)

                # EMA in feature space, then squash to [0,1]
                ema_A = ema(ema_A, a_raw, EMA_ALPHA)
                ema_V = ema(ema_V, v_raw, EMA_ALPHA)
                A_scaled = squash_tanh(ema_A, A_center, A_mad)
                V_scaled = squash_tanh(ema_V, V_center, V_mad)

                # Only update outputs & adaptation if no artifact
                if art == 0:
                    A_out, V_out = A_scaled, V_scaled
                    # slow adaptation to drift (centers and MADs)
                    A_center = (1 - ADAPT_ALPHA) * A_center + ADAPT_ALPHA * ema_A
                    V_center = (1 - ADAPT_ALPHA) * V_center + ADAPT_ALPHA * ema_V
                    A_mad = (1 - ADAPT_ALPHA) * A_mad + ADAPT_ALPHA * abs(ema_A - A_center)
                    V_mad = (1 - ADAPT_ALPHA) * V_mad + ADAPT_ALPHA * abs(ema_V - V_center)

                # Emit OSC
                if A_out is not None and V_out is not None:
                    client.send_message(ADDR_A, float(A_out))
                    client.send_message(ADDR_V, float(V_out))
                client.send_message(ADDR_ART, int(art))
            else:
                # Oracle mode only recomputes A/V when params change (below)
                pass

            # ==== Collect A/V samples for current phase (after settle) ====
            if A_out is not None and V_out is not None and (art == 0 or mode == "MOCK_ORACLE"):
                if phase == 1 and phase_hop_index >= settle_hops:
                    plus_A.append(A_out)
                    plus_V.append(V_out)
                elif phase == 2 and phase_hop_index >= settle_hops:
                    minus_A.append(A_out)
                    minus_V.append(V_out)

            hop_counter += 1

            # ==== Epoch controller ====
            if hop_counter % epoch_hops == 1:
                # Ensure we have a current A,V and start a new epoch for optimizer
                if mode == "MOCK_ORACLE":
                    A_out, V_out = mock_av_response(current_params)
                    client.send_message(ADDR_A, float(A_out))
                    client.send_message(ADDR_V, float(V_out))

                if A_out is not None and V_out is not None:
                    d0 = distance(A_out, V_out)
                    opt.start_epoch(d0)

                    # '+' proposal from optimizer (SPSA/BO)
                    p_plus = opt.propose(phase=1)
                    if p_plus is not None:
                        current_params = p_plus
                        for k, v in current_params.items():
                            client.send_message(f"/music/{k}", float(v))
                        if mode == "MOCK_EEG":
                            src.set_current_params(current_params)
                        if mode == "MOCK_ORACLE":
                            A_out, V_out = mock_av_response(current_params)
                            client.send_message(ADDR_A, float(A_out))
                            client.send_message(ADDR_V, float(V_out))

                # reset '+' buffers
                plus_A.clear()
                plus_V.clear()
                phase_hop_index = 0
                phase = 1

            # End of '+' phase every epoch_hops
            if hop_counter % epoch_hops == 0 and phase == 1:
                # Report to optimizer
                if opt.name() == "spsa":
                    # SPSA consumes averaged distance as reward for '+'
                    if plus_A and plus_V:
                        d_plus = float(np.mean([distance(a, v) for a, v in zip(plus_A, plus_V)]))
                    else:
                        d_plus = d0
                    opt.observe_reward_plus(d_plus)
                else:
                    # BO consumes A/V lists and computes reward internally
                    opt.report(phase=1, A_list=plus_A, V_list=plus_V)

                # Plot '+' averages
                A_mean_plus = float(np.mean(plus_A)) if plus_A else (A_out if A_out is not None else 0.5)
                V_mean_plus = float(np.mean(plus_V)) if plus_V else (V_out if V_out is not None else 0.5)
                live.update(A_mean_plus, V_mean_plus, current_params)

                # Ask optimizer for '-' phase (SPSA only). BO returns None.
                p_minus = opt.propose(phase=2)
                if p_minus is not None:
                    current_params = p_minus
                    for k, v in current_params.items():
                        client.send_message(f"/music/{k}", float(v))
                    if mode == "MOCK_EEG":
                        src.set_current_params(current_params)
                    if mode == "MOCK_ORACLE":
                        A_out, V_out = mock_av_response(current_params)
                        client.send_message(ADDR_A, float(A_out))
                        client.send_message(ADDR_V, float(V_out))

                # prepare '-' buffers
                minus_A.clear()
                minus_V.clear()
                phase_hop_index = 0
                # If BO: skip '-' and jump to epoch-end step; else continue to phase 2
                phase = 2 if p_minus is not None else 0

            # End of '-' phase (SPSA only) every 2*epoch_hops
            if hop_counter % (2 * epoch_hops) == 0 and phase == 2:
                # Reward for '-'
                if minus_A and minus_V:
                    d_minus = float(np.mean([distance(a, v) for a, v in zip(minus_A, minus_V)]))
                else:
                    d_minus = d0
                opt.observe_reward_minus(d_minus)

                # Plot '-' averages
                A_mean_minus = float(np.mean(minus_A)) if minus_A else (A_out if A_out is not None else 0.5)
                V_mean_minus = float(np.mean(minus_V)) if minus_V else (V_out if V_out is not None else 0.5)
                live.update(A_mean_minus, V_mean_minus, current_params)

                # Apply SPSA update, push new center
                opt.step()
                current_params = opt.current_params_dict()
                for k, v in current_params.items():
                    client.send_message(f"/music/{k}", float(v))
                if mode == "MOCK_EEG":
                    src.set_current_params(current_params)
                if mode == "MOCK_ORACLE":
                    A_out, V_out = mock_av_response(current_params)
                    client.send_message(ADDR_A, float(A_out))
                    client.send_message(ADDR_V, float(V_out))
                phase = 0

            # End-of-epoch step for BO (no '-' phase)
            if opt.name() == "bo" and (hop_counter % epoch_hops == 0) and phase == 0:
                opt.step()
                current_params = opt.current_params_dict()
                for k, v in current_params.items():
                    client.send_message(f"/music/{k}", float(v))
                if mode == "MOCK_EEG":
                    src.set_current_params(current_params)
                if mode == "MOCK_ORACLE":
                    A_out, V_out = mock_av_response(current_params)
                    client.send_message(ADDR_A, float(A_out))
                    client.send_message(ADDR_V, float(V_out))

            # Status every second
            if hop_counter % int(1.0 / (N_HOP / FS)) == 0:
                ps = ", ".join([f"{k}={v:.2f}" for k, v in current_params.items()]) if 'current_params' in locals() else ""
                a_show = A_out if A_out is not None else float('nan')
                v_show = V_out if V_out is not None else float('nan')
                d_show = distance(0.5 if A_out is None else A_out, 0.5 if V_out is None else V_out)
                print(f"[{mode}|{opt.name()}] A={a_show:.3f} V={v_show:.3f} dist={d_show:.3f} | {ps}", end="\r", flush=True)

            # Advance phase hop index
            if phase in (1, 2):
                phase_hop_index += 1

    except KeyboardInterrupt:
        print("\nStopped by user.")
        if src is not None and mode in ("EEG_REAL", "MOCK_EEG"):
            src.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizer", "-o", default=DEFAULT_OPTIMIZER, choices=["spsa", "bo"])
    args = parser.parse_args()
    run(args.optimizer)
