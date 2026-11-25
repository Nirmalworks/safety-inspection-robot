"""Minimal ROS 1 subscriber in plain Python (no ROS 1 install needed).

The robot's other container runs ROS 1 with the Astra camera driver; this side runs ROS 2.
This speaks just enough of ROS 1's protocol to read camera topics: ask the master who
publishes a topic (XML-RPC), ask that node for a TCP port (requestTopic), then read
length-prefixed messages over TCPROS. Decoders cover Image, CompressedImage, CameraInfo.
"""
import os
import socket
import struct
import threading
import time
import xmlrpc.client

CALLER = '/jetauto_dashboard'


class Reader:
    def __init__(self, b):
        self.b, self.o = b, 0

    def u8(self):
        self.o += 1
        return self.b[self.o - 1]

    def u32(self):
        self.o += 4
        return struct.unpack_from('<I', self.b, self.o - 4)[0]

    def f64s(self, n):
        self.o += 8 * n
        return list(struct.unpack_from(f'<{n}d', self.b, self.o - 8 * n))

    def string(self):
        n = self.u32()
        self.o += n
        return bytes(self.b[self.o - n:self.o]).decode(errors='replace')

    def blob(self):
        n = self.u32()
        self.o += n
        return memoryview(self.b)[self.o - n:self.o]

    def header(self):
        self.u32()                               # seq
        sec, nsec = self.u32(), self.u32()
        return sec + nsec * 1e-9, self.string()


def decode_image(b):
    r = Reader(b)
    stamp, frame = r.header()
    h, w = r.u32(), r.u32()
    enc = r.string()
    r.u8()                                       # is_bigendian
    step = r.u32()
    return dict(stamp=stamp, frame=frame, height=h, width=w, encoding=enc, step=step, data=r.blob())


def decode_compressed(b):
    r = Reader(b)
    stamp, frame = r.header()
    fmt = r.string()
    return dict(stamp=stamp, frame=frame, format=fmt, data=r.blob())


def decode_camera_info(b):
    r = Reader(b)
    stamp, frame = r.header()
    h, w = r.u32(), r.u32()
    r.string()                                   # distortion model
    r.f64s(r.u32())                              # D
    return dict(stamp=stamp, frame=frame, height=h, width=w, K=r.f64s(9))


def decode_scan(b):
    r = Reader(b)
    stamp, frame = r.header()
    amin, amax, inc, tinc, stime, rmin, rmax = struct.unpack_from('<7f', r.b, r.o)
    r.o += 28
    n = r.u32()
    ranges = struct.unpack_from(f'<{n}f', r.b, r.o)
    r.o += 4 * n
    n = r.u32()
    intens = struct.unpack_from(f'<{n}f', r.b, r.o)
    return dict(stamp=stamp, frame=frame, angle_min=amin, angle_max=amax, angle_increment=inc,
                time_increment=tinc, scan_time=stime, range_min=rmin, range_max=rmax,
                ranges=ranges, intensities=intens)


def decode_imu(b):
    r = Reader(b)
    stamp, frame = r.header()
    q = r.f64s(4)
    r.f64s(9)
    w = r.f64s(3)
    r.f64s(9)
    a = r.f64s(3)
    return dict(stamp=stamp, frame=frame, orientation=q, angular_velocity=w, linear_acceleration=a)


def decode_uint16(b):
    return dict(data=struct.unpack_from('<H', b, 0)[0])


TYPES = {
    'sensor_msgs/LaserScan': decode_scan,
    'sensor_msgs/Imu': decode_imu,
    'std_msgs/UInt16': decode_uint16,
    'sensor_msgs/Image': decode_image,
    'sensor_msgs/CompressedImage': decode_compressed,
    'sensor_msgs/CameraInfo': decode_camera_info,
}


class Subscriber(threading.Thread):
    """Keeps the latest decoded message of one ROS 1 topic; reconnects if the link drops."""

    def __init__(self, master, topic, typ, on_msg=None):
        super().__init__(daemon=True)
        self.master, self.topic, self.typ, self.on_msg = master, topic, typ, on_msg
        self.latest, self.count, self.error = None, 0, None
        self.stamps = []
        self.start()

    def connect(self):
        m = xmlrpc.client.ServerProxy(self.master)
        code, _, state = m.getSystemState(CALLER)
        pubs = dict(state[0]).get(self.topic)
        if not pubs:
            raise RuntimeError(f'no publisher for {self.topic}')
        _, _, uri = m.lookupNode(CALLER, pubs[0])
        code, msg, proto = xmlrpc.client.ServerProxy(uri).requestTopic(CALLER, self.topic, [['TCPROS']])
        if code != 1:
            raise RuntimeError(msg)
        host = proto[1]
        try:
            socket.gethostbyname(host)
        except OSError:
            host = '127.0.0.1'                   # container hostnames may not resolve here
        s = socket.create_connection((host, proto[2]), timeout=5)
        fields = [f'callerid={CALLER}', f'topic={self.topic}', f'type={self.typ}', 'md5sum=*', 'tcp_nodelay=1']
        body = b''.join(struct.pack('<I', len(f)) + f.encode() for f in fields)
        s.sendall(struct.pack('<I', len(body)) + body)
        self.recv_exact(s, struct.unpack('<I', self.recv_exact(s, 4))[0])   # publisher's header
        s.settimeout(10)
        return s

    @staticmethod
    def recv_exact(s, n):
        buf = bytearray(n)
        view, got = memoryview(buf), 0
        while got < n:
            k = s.recv_into(view[got:], n - got)
            if not k:
                raise ConnectionError('closed')
            got += k
        return buf

    def run(self):
        decode = TYPES[self.typ]
        while True:
            try:
                s = self.connect()
                self.error = None
                while True:
                    n = struct.unpack('<I', self.recv_exact(s, 4))[0]
                    msg = decode(self.recv_exact(s, n))
                    self.latest = msg
                    self.count += 1
                    now = time.time()
                    self.stamps = [t for t in self.stamps if now - t < 3.0] + [now]
                    if self.on_msg:
                        self.on_msg(msg)
            except Exception as e:  # camera container restarting, master gone, etc.
                self.error = str(e)
                time.sleep(2)

    def hz(self):
        now = time.time()
        recent = [t for t in self.stamps if now - t < 3.0]
        return round(len(recent) / 3.0, 1) if len(recent) > 1 else 0.0


def call_service(master, srv, req=b''):
    """Call a ROS 1 service; returns (ok, raw response bytes). md5sum '*' skips the type check."""
    from urllib.parse import urlparse
    _, _, uri = xmlrpc.client.ServerProxy(master).lookupService(CALLER, srv)
    u = urlparse(uri)
    host = u.hostname
    try:
        socket.gethostbyname(host)
    except OSError:
        host = '127.0.0.1'
    s = socket.create_connection((host, u.port), timeout=5)
    fields = [f'callerid={CALLER}', f'service={srv}', 'md5sum=*']
    body = b''.join(struct.pack('<I', len(f)) + f.encode() for f in fields)
    s.sendall(struct.pack('<I', len(body)) + body)
    rx = Subscriber.recv_exact
    rx(s, struct.unpack('<I', rx(s, 4))[0])
    s.sendall(struct.pack('<I', len(req)) + req)
    ok = rx(s, 1)[0]
    payload = bytes(rx(s, struct.unpack('<I', rx(s, 4))[0]))
    s.close()
    return bool(ok), payload


class Publisher:
    """Minimal ROS 1 publisher: registers with the master, serves requestTopic over XML-RPC,
    and sends length-prefixed messages to every subscriber over TCPROS."""

    MD5 = {'geometry_msgs/Twist': '9f195f881246fdfa2798d1d3eebca84a'}

    def __init__(self, master, topic, typ, caller=CALLER + '_pub'):
        from xmlrpc.server import SimpleXMLRPCServer
        self.master, self.topic, self.typ, self.caller = master, topic, typ, caller
        self.conns, self.lock = [], threading.Lock()
        self.tcp = socket.socket()
        self.tcp.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.tcp.bind(('0.0.0.0', 0))
        self.tcp.listen(5)
        self.rpc = SimpleXMLRPCServer(('0.0.0.0', 0), logRequests=False, allow_none=True)
        self.rpc.register_function(lambda c, t, p: [1, 'ok', ['TCPROS', 'localhost', self.tcp.getsockname()[1]]], 'requestTopic')
        self.rpc.register_function(lambda c: [1, 'ok', os.getpid()], 'getPid')
        self.rpc.register_function(lambda c, t, p: [1, 'ok', 0], 'publisherUpdate')
        self.rpc.register_function(lambda c, *a: [1, 'ok', [[1, 'sub', 'o', 'TCPROS', self.topic]]], 'getBusInfo')
        self.rpc.register_function(lambda c, m='': [1, 'ok', 0], 'shutdown')
        self.uri = f'http://localhost:{self.rpc.server_address[1]}/'
        threading.Thread(target=self.rpc.serve_forever, daemon=True).start()
        threading.Thread(target=self.accept, daemon=True).start()
        xmlrpc.client.ServerProxy(master).registerPublisher(caller, topic, typ, self.uri)

    def accept(self):
        while True:
            s, _ = self.tcp.accept()
            try:
                Subscriber.recv_exact(s, struct.unpack('<I', Subscriber.recv_exact(s, 4))[0])
                fields = [f'callerid={self.caller}', f'topic={self.topic}', f'type={self.typ}',
                          f'md5sum={self.MD5[self.typ]}', 'latching=0']
                body = b''.join(struct.pack('<I', len(f)) + f.encode() for f in fields)
                s.sendall(struct.pack('<I', len(body)) + body)
                with self.lock:
                    self.conns.append(s)
            except Exception:
                s.close()

    def subscribers(self):
        with self.lock:
            return len(self.conns)

    def publish_raw(self, payload):
        frame = struct.pack('<I', len(payload)) + payload
        with self.lock:
            for s in list(self.conns):
                try:
                    s.sendall(frame)
                except OSError:
                    self.conns.remove(s)

    def publish_twist(self, vx=0.0, vy=0.0, wz=0.0):
        self.publish_raw(struct.pack('<6d', vx, vy, 0.0, 0.0, 0.0, wz))

    def close(self):
        try:
            xmlrpc.client.ServerProxy(self.master).unregisterPublisher(self.caller, self.topic, self.uri)
        except Exception:
            pass
