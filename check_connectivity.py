#!/usr/bin/env python3
"""Check TCP/UDP connectivity using Python std libraries."""

import socket
import sys


DISCORD = "https://discord.gg/JMWxhgmVzf"
TIMEOUT = 3
DEFAULT_CHECKS = [
    ("Direct connection", "benji.unaiverse.io", "TCP", 33000),
    ("Direct connection by IP", "159.195.199.26", "TCP", 33000),
    ("Relay connection", "misc.unaiverse.io", "TCP", 30060),
    ("Relay connection", "misc.unaiverse.io", "TCP", 30070),
    ("Relay connection by IP", "159.195.197.171", "TCP", 30060),
    ("Relay connection by IP", "159.195.197.171", "TCP", 30070),
    ("UDP connection", "159.195.199.26", "UDP", 33001),
]


def result(status, message):
    color = {"OK": "32", "WARNING": "33", "ERROR": "31"}[status]
    label = f"[{status}]"
    if sys.stdout.isatty():
        label = f"\033[{color}m{label}\033[0m"
    print(f"{label} {message}", flush=True)


def probe(family, kind, address):
    try:
        with socket.socket(family, kind) as sock:
            sock.settimeout(TIMEOUT)
            sock.connect(address)
            if kind == socket.SOCK_DGRAM:
                sock.send(b"connectivity-check")
                sock.recv(4096)
                return "OK", "UDP reply received"
            return "OK", "TCP connection established"
    except socket.timeout:
        if kind == socket.SOCK_DGRAM:
            return "UNKNOWN", "no UDP reply. Check your firewall, routing or server configuration"
        return "TIMEOUT", "no TCP connection. Check your firewall, routing or server configuration"
    except ConnectionRefusedError:
        return "OK", "Even if it is counterintuitive, a TCP connection refused is a valid response. The server is reachable and working."
    except OSError as exc:
        return "ERROR", str(exc)


def main():
    print(f"UNaIVERSE connection check - {socket.gethostname()}", flush=True)
    print("Please wait while we check your connections...\n", flush=True)
    counts = {"OK": 0, "WARNING": 0, "ERROR": 0}
    for index, (name, host, protocol, port) in enumerate(DEFAULT_CHECKS, 1):
        print(f"{index}/{len(DEFAULT_CHECKS)} {name} ({host}:{port}, {protocol})", flush=True)
        kind = socket.SOCK_STREAM if protocol == "TCP" else socket.SOCK_DGRAM
        try:
            addresses = socket.getaddrinfo(host, port, type=kind)
        except OSError as exc:
            result("ERROR", f"Cannot look up this server: {exc}")
            counts["ERROR"] += 1
            continue
        seen = set()
        for family, _, _, _, address in addresses:
            if (family, address) in seen:
                continue
            seen.add((family, address))
            status, detail = probe(family, kind, address)
            status = "WARNING" if status == "UNKNOWN" else "OK" if status == "OK" else "ERROR"
            result(status, f"{detail} ({address[0]})")
            counts[status] += 1

    print(f"\nDone: {counts['OK']} OK, {counts['WARNING']} warnings, {counts['ERROR']} errors.")
    if counts["ERROR"] or counts["WARNING"]:
        status = "ERROR" if counts["ERROR"] else "WARNING"
        result(status, "")
        print(f"Contact us on Discord: {DISCORD}")
        print("Copy this full output or send a screenshot so we can help.")
        return 1
    result("OK", "All checks received a connection, reply or explicit refusal.")
    print(f"If your agent still cannot connect, contact us on Discord: {DISCORD}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
        raise SystemExit(130)
