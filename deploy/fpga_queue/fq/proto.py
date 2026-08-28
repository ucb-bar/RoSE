"""Wire protocol between clients and the daemon.

Transport: a **unix domain stream socket**.

Why a socket and not a shared spool directory or a shared SQLite file
(which is what firesim-queue used)?

  * **Unforgeable identity.**  ``SO_PEERCRED`` gives the daemon the kernel's
    own view of the client's uid/gid/pid.  A submitter cannot claim to be
    someone else, cannot cancel someone else's job, and cannot edit a job
    record after submission.  With a group-writable spool or DB, every
    submitter can rewrite every other submitter's state; the only thing
    stopping them is politeness.  This pool drives machines that cost
    $2/hour each, so politeness is not the control we want.
  * **Validation happens once, in one place.**  A spec is checked by the
    daemon before it ever becomes a job, so there is no such thing as a
    half-written or malformed job record.
  * **Liveness is obvious.**  ``connect()`` either works or it does not.  No
    heartbeat-age heuristics.

The cost is that submission requires the daemon to be up.  That is an
acceptable trade for a resource that cannot be used without the daemon
anyway, and the CLI says so plainly instead of silently queueing into a void.

Framing: newline-delimited JSON.  ``json.dumps`` never emits a raw newline
(it escapes them), so this is safe for arbitrary payloads including log text.
One request, one response, then the connection closes -- no multiplexing,
nothing to get out of sync.
"""

from __future__ import annotations

import json
import socket
import struct
from typing import Any, Optional

# Refuse absurd frames rather than trying to buffer them.
MAX_FRAME = 8 * 1024 * 1024

PROTOCOL_VERSION = 1


class ProtocolError(Exception):
    pass


def send_msg(sock: socket.socket, obj: Any) -> None:
    data = json.dumps(obj, separators=(",", ":")).encode() + b"\n"
    if len(data) > MAX_FRAME:
        raise ProtocolError(f"frame too large: {len(data)} bytes")
    sock.sendall(data)


class LineReader:
    """Buffered newline-framed reader over a socket."""

    def __init__(self, sock: socket.socket):
        self.sock = sock
        self.buf = bytearray()

    def read_msg(self) -> Optional[Any]:
        while True:
            nl = self.buf.find(b"\n")
            if nl >= 0:
                line = bytes(self.buf[:nl])
                del self.buf[:nl + 1]
                if not line.strip():
                    continue
                try:
                    return json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ProtocolError(f"bad JSON frame: {exc}") from exc
            if len(self.buf) > MAX_FRAME:
                raise ProtocolError("frame too large")
            chunk = self.sock.recv(65536)
            if not chunk:
                return None
            self.buf.extend(chunk)


def recv_msg(sock: socket.socket) -> Optional[Any]:
    return LineReader(sock).read_msg()


def peer_credentials(sock: socket.socket) -> tuple[int, int, int]:
    """(pid, uid, gid) of the peer, straight from the kernel.

    This is the authentication mechanism.  It cannot be spoofed by the client
    (SO_PEERCRED is filled in by the kernel at connect() time), and it needs
    no shared secret, no config and no key management.
    """
    creds = sock.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED,
                            struct.calcsize("3i"))
    pid, uid, gid = struct.unpack("3i", creds)
    return pid, uid, gid


def ok(result: Any = None) -> dict:
    return {"ok": True, "result": result, "v": PROTOCOL_VERSION}


def err(message: str, code: str = "error") -> dict:
    return {"ok": False, "error": message, "code": code,
            "v": PROTOCOL_VERSION}
