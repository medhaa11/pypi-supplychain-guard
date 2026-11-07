import pandas as pd


def apply_rules(df):
    net = df.c_setup_net + df.c_pkg_net
    proc = df.c_setup_proc + df.c_pkg_proc
    ex = df.c_setup_exec_eval + df.c_pkg_exec_eval
    dec = df.c_setup_decode + df.c_pkg_decode
    exfil = df.c_setup_exfil + df.c_pkg_exfil
    sens = df.c_setup_sens + df.c_pkg_sens
    custom_cmd = (df.c_setup_cmdclass + df.c_setup_cmd_subclass) > 0
    R = pd.DataFrame({
        "R1 setup.py makes network calls": df.c_setup_net > 0,
        "R2 custom install cmd + proc/net/exec/decode": custom_cmd & ((proc + net + ex + dec) > 0),
        "R3 exfil-service string": exfil > 0,
        "R4 decode + exec/eval": (dec > 0) & (ex > 0),
        "R5 setup.py spawns process + network": (df.c_setup_proc > 0) & (df.c_setup_net > 0),
        "R6 sensitive paths + network": (sens > 0) & (net > 0),
    })
    R["any"] = R.any(axis=1)
    return R
