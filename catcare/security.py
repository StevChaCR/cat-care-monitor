"""Validate local camera connections without logging or persisting credentials."""
import ipaddress
import socket
from urllib.parse import urlsplit

PRIVATE_NETWORKS = tuple(ipaddress.ip_network(network) for network in (
    "10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "fc00::/7",
))


def is_local(address):
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        address = address.ipv4_mapped
    return (not address.is_unspecified and not address.is_multicast and
            (address.is_loopback or address.is_link_local or any(
                address.version == network.version and address in network
                for network in PRIVATE_NETWORKS)))


def validate_rtsp(value, resolver=None):
    value = value.strip()
    if not value or any(c.isspace() or ord(c) < 32 for c in value):
        raise ValueError("Paste a complete RTSP URL without spaces or control characters.")
    try:
        parts = urlsplit(value)
        port = parts.port
    except ValueError:
        raise ValueError("The RTSP address or port is invalid.") from None
    if parts.scheme.lower() not in ("rtsp", "rtsps") or not parts.hostname:
        raise ValueError("Use an rtsp:// or rtsps:// camera URL.")
    if port is not None and not 1 <= port <= 65535:
        raise ValueError("The camera port must be between 1 and 65535.")
    if not parts.username or not parts.password:
        raise ValueError("Enable camera authentication and include its RTSP username and password.")
    try:
        addresses = [ipaddress.ip_address(parts.hostname)]
    except ValueError:
        resolver = resolver or socket.getaddrinfo
        try:
            addresses = [ipaddress.ip_address(record[4][0])
                         for record in resolver(parts.hostname, None)]
        except (OSError, ValueError):
            raise ValueError("Could not resolve the camera's local address.") from None
    if not addresses or not all(is_local(address) for address in addresses):
        raise ValueError("Only cameras on a private local network are supported.")
    return value
