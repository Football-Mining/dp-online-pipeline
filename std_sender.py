import sys

def send_data(*args):
    sys.stdout.write(" ".join(map(str, args)) + "\n")
    sys.stdout.flush()

def send_bytes(data_bytes):
    # Send metadata and data bytes
    sys.stdout.buffer.write(data_bytes)
    sys.stdout.flush()