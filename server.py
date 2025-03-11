import bitalino
import threading
import time
import socket

from CONFIGURATION import *

# Shared data
SHARED_server_running = True
SHARED_server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
SHARED_latest_sampling: list[int] = None
SHARED_sample_ready_sem = threading.BoundedSemaphore(1)

# Consume the initial semaphore
SHARED_sample_ready_sem.acquire()

def server_thread_func():
    global SHARED_latest_sampling
    global SHARED_sample_ready_sem
    global SHARED_server_running

    # Create a socket
    SHARED_server_socket.bind(('0.0.0.0', SERVER_PORT))
    SHARED_server_socket.listen(1)

    print(f"Listening on port {SERVER_PORT}...")

    try:
        while SHARED_server_running:
            client_socket, client_address = SHARED_server_socket.accept()
            print(f"New client: {client_address}")

            while True:
                # Wait for client to send something
                data = client_socket.recv(1)
                if not data:
                    break

                # Send the latest sampling to the client
                SHARED_sample_ready_sem.acquire()

                data_to_send = SHARED_latest_sampling.copy()
                serialized_data = DATA_PACKET_FORMAT.pack(
                    data_to_send[0], data_to_send[1], data_to_send[2],
                    data_to_send[3], data_to_send[4], data_to_send[5]
                )
                client_socket.sendall(serialized_data)

            print(f"Bye client: {client_address}")
            client_socket.close()
    except OSError as e:
        if e.winerror == 10038:
            print("Server socket closed.")
        else:
            raise e
    finally:
        print("Stopping server...")
        SHARED_server_socket.close()
        SHARED_server_running = False

def main():
    global SHARED_server_running
    global SHARED_server_socket
    global SHARED_latest_sampling
    global SHARED_sample_ready_sem

    try:
        device = bitalino.BITalino(MAC_ADDRESS)
        print(f"Device version: {device.version()}")
        dev_state = device.state()
        print(f"Battery state = {dev_state['battery']} @ threshold {dev_state['batteryThreshold']}")
        device.start(SAMPLING_RATE, CHANNELS)
    except Exception as e:
        print(str(e))
        print("Aborting...")
        return

    server_thread = threading.Thread(target=server_thread_func, name="ServerThread")
    server_thread.start()

    try:
        while SHARED_server_running:
            one_sec_start = time.time()
            num_acquisitions = 0
            while True:
                samples = device.read(1)[0, 5:]
                SHARED_latest_sampling = samples

                # Release the semaphore to allow server thread
                # to send data. We have to wrap this into a
                # try-except block because the semaphore might
                # already be released, and Python decided that
                # this should result in an exception...
                try:
                    SHARED_sample_ready_sem.release()
                except ValueError:
                    pass

                num_acquisitions += 1

                end = time.time()
                if (end - one_sec_start) >= 1:
                    break

            elapsed = time.time() - one_sec_start
            # print warning if we can't keep up (>5% samples lost)
            if num_acquisitions < (0.95 * SAMPLING_RATE):
                print(f"Can't keep up! Sampled {num_acquisitions} in {elapsed:0.2f}s (wanted {SAMPLING_RATE})")

    except KeyboardInterrupt:
        pass

    finally:
        try:
            device.stop()
            device.close()
        except:
            pass

        SHARED_server_running = False
        SHARED_server_socket.close()
        server_thread.join()

main()